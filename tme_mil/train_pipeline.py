import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from .calibration import calibrated_probabilities, fit_temperature, softmax_np
from .config import ModelConfig, TrainConfig
from .data import BagDataset, build_eval_loader, build_patient_maps, build_train_loader, require_keys
from .metrics import compute_metrics, save_summary_artifacts
from .model import FourBranchGatedMIL, forward_bag_all_cells
from .utils import get_grad_scaler, save_json, set_seed


def maybe_stratified_group_kfold():
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold
    except Exception:
        return None


def find_best_group_stratified_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    num_classes: int,
    min_class_per_fold: int,
    max_tries: int,
):
    sgkf = maybe_stratified_group_kfold()
    if sgkf is None:
        raise ImportError("scikit-learn does not provide StratifiedGroupKFold")

    best_splits = None
    best_info = None
    best_score = None

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)

    for t in range(int(max_tries)):
        rs = int(seed) + t
        splitter = sgkf(n_splits=int(n_splits), shuffle=True, random_state=rs)
        splits = list(splitter.split(X=np.zeros(len(y)), y=y, groups=groups))

        ok = True
        counts_per_fold = []
        fold_sizes = []
        for _, va in splits:
            counts = np.bincount(y[va], minlength=int(num_classes)).astype(int).tolist()
            counts_per_fold.append(counts)
            fold_sizes.append(len(va))
            if any(c < int(min_class_per_fold) for c in counts):
                ok = False

        fold_sizes = np.asarray(fold_sizes, dtype=float)
        size_cv = float(np.std(fold_sizes, ddof=0) / (np.mean(fold_sizes) + 1e-8))
        score = -size_cv

        if ok and (best_score is None or score > best_score):
            best_score = score
            best_splits = splits
            best_info = {
                "random_state": int(rs),
                "split_size_cv": float(size_cv),
                "val_counts_per_fold": counts_per_fold,
            }

    if best_splits is None or best_info is None:
        raise ValueError(
            "Failed to find group-stratified folds where every validation fold contains all classes"
        )

    return best_splits, best_info


@torch.no_grad()
def predict_logits_loader(
    model: FourBranchGatedMIL,
    loader,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
):
    model.eval()
    y_true = []
    logits_all = []

    for (go, tf, kegg, reactome), y, _pid in loader:
        logits, _ = forward_bag_all_cells(
            model,
            go[0],
            tf[0],
            kegg[0],
            reactome[0],
            device=device,
            chunk_size=int(chunk_size),
            use_amp=bool(use_amp),
        )
        logits_all.append(logits.detach().cpu().numpy())
        y_true.append(int(y.item()))

    return np.asarray(y_true, dtype=int), np.vstack(logits_all)


def train_one_fold(
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    train_pids,
    val_pids,
    patient_to_indices,
    go_scores,
    tf_scores,
    kegg_scores,
    reactome_scores,
    patient_label,
    device: torch.device,
    fold_index: int,
):
    train_ds = BagDataset(
        train_pids,
        patient_to_indices,
        go_scores,
        tf_scores,
        kegg_scores,
        reactome_scores,
        patient_label,
    )
    val_ds = BagDataset(
        val_pids,
        patient_to_indices,
        go_scores,
        tf_scores,
        kegg_scores,
        reactome_scores,
        patient_label,
    )

    train_loader = build_train_loader(train_ds, cfg.train_bag_batch_size)
    val_loader = build_eval_loader(val_ds)

    model = FourBranchGatedMIL(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
    )

    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    scaler = get_grad_scaler(amp_enabled)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    print(f"[Train] Fold {fold_index}: train patients={len(train_pids)}, validation patients={len(val_pids)}")

    for epoch in range(1, int(cfg.max_epochs) + 1):
        model.train()
        train_loss_sum = 0.0

        for (go_list, tf_list, kegg_list, react_list), y_batch, _pids in train_loader:
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits_batch = []
            for go_cpu, tf_cpu, kegg_cpu, reactome_cpu in zip(go_list, tf_list, kegg_list, react_list):
                logits_i, _ = forward_bag_all_cells(
                    model,
                    go_cpu,
                    tf_cpu,
                    kegg_cpu,
                    reactome_cpu,
                    device=device,
                    chunk_size=int(cfg.chunk_size),
                    use_amp=amp_enabled,
                )
                logits_batch.append(logits_i)

            logits_batch = torch.stack(logits_batch, dim=0)
            loss = criterion(logits_batch, y_batch)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss_sum += float(loss.item())

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for (go, tf, kegg, reactome), y, _pid in val_loader:
                logits, _ = forward_bag_all_cells(
                    model,
                    go[0],
                    tf[0],
                    kegg[0],
                    reactome[0],
                    device=device,
                    chunk_size=int(cfg.chunk_size),
                    use_amp=amp_enabled,
                )
                val_loss_sum += float(criterion(logits.unsqueeze(0), y.to(device)).item())

        train_loss = train_loss_sum / max(1, len(train_loader))
        val_loss = val_loss_sum / max(1, len(val_loader))
        scheduler.step(val_loss)

        print(
            f"[Train] Fold {fold_index} | Epoch {epoch:03d}/{int(cfg.max_epochs):03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss - float(cfg.min_delta):
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.patience):
                print(f"[Train] Fold {fold_index}: early stopping at epoch {epoch}.")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_true, logits = predict_logits_loader(
        model=model,
        loader=val_loader,
        device=device,
        chunk_size=int(cfg.chunk_size),
        use_amp=amp_enabled,
    )
    prob = softmax_np(logits)
    metrics = compute_metrics(y_true, prob, [str(i) for i in range(model_cfg.num_classes)])
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_loss"] = float(best_val_loss)

    print(
        f"[Train] Fold {fold_index} completed | best_epoch={best_epoch} | "
        f"best_val_loss={best_val_loss:.4f} | acc={metrics['acc']:.4f} | macro_f1={metrics['macro_f1']:.4f}"
    )
    return metrics, y_true, logits


def train_final_model(
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    all_pids,
    patient_to_indices,
    go_scores,
    tf_scores,
    kegg_scores,
    reactome_scores,
    patient_label,
    device: torch.device,
    epochs: int,
):
    train_ds = BagDataset(
        all_pids,
        patient_to_indices,
        go_scores,
        tf_scores,
        kegg_scores,
        reactome_scores,
        patient_label,
    )
    train_loader = build_train_loader(train_ds, cfg.train_bag_batch_size)

    model = FourBranchGatedMIL(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    scaler = get_grad_scaler(amp_enabled)

    print(f"[Train] Training final TEMILA on the full dataset for {epochs} epochs.")
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        for (go_list, tf_list, kegg_list, react_list), y_batch, _pids in train_loader:
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits_batch = []
            for go_cpu, tf_cpu, kegg_cpu, reactome_cpu in zip(go_list, tf_list, kegg_list, react_list):
                logits_i, _ = forward_bag_all_cells(
                    model,
                    go_cpu,
                    tf_cpu,
                    kegg_cpu,
                    reactome_cpu,
                    device=device,
                    chunk_size=int(cfg.chunk_size),
                    use_amp=amp_enabled,
                )
                logits_batch.append(logits_i)

            logits_batch = torch.stack(logits_batch, dim=0)
            loss = criterion(logits_batch, y_batch)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(1, len(train_loader))
        print(f"[Train] Final TEMILA | Epoch {epoch:03d}/{int(epochs):03d} | train_loss={train_loss:.4f}")

    print("[Train] Final TEMILA training finished.")
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def mean_std(rows, key: str):
    vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
    mean = float(np.nanmean(vals))
    std = float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else 0.0
    return mean, std


def run_training(cfg: TrainConfig) -> None:
    print("[Train] Step 2/2: TEMILA training.")
    set_seed(int(cfg.seed))
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[Train] Loading training feature file from {cfg.data_path}")
    data = np.load(cfg.data_path, allow_pickle=True)
    require_keys(
        data,
        [
            "go_bp_scores",
            "tf_scores",
            "kegg_scores",
            "reactome_scores",
            "patients",
            "subtypes",
            "datasets",
            "score_cols_go_bp",
            "score_cols_tf",
            "score_cols_kegg",
            "score_cols_reactome",
        ],
    )

    go_scores = data["go_bp_scores"]
    tf_scores = data["tf_scores"]
    kegg_scores = data["kegg_scores"]
    reactome_scores = data["reactome_scores"]
    patients = np.array([str(x) for x in data["patients"]], dtype=object)
    subtypes = np.array([str(x) for x in data["subtypes"]], dtype=object)
    datasets = np.array([str(x) for x in data["datasets"]], dtype=object)

    class_names = sorted(np.unique(subtypes))
    subtype_to_int = {s: i for i, s in enumerate(class_names)}
    y_cell = np.asarray([subtype_to_int[s] for s in subtypes], dtype=int)

    patient_to_indices, patient_label, patient_group = build_patient_maps(
        patients=patients,
        labels=y_cell,
        groups=datasets,
    )

    all_pids = np.array(sorted(patient_to_indices.keys()), dtype=object)
    y_pat = np.array([patient_label[pid] for pid in all_pids], dtype=int)
    g_pat = np.array([patient_group[pid] for pid in all_pids], dtype=object)

    print(f"[Train] Patients: {len(all_pids)} | Classes: {len(class_names)} | Datasets: {len(np.unique(g_pat))}")
    print(f"[Train] Class names: {', '.join(class_names)}")

    splits, split_info = find_best_group_stratified_splits(
        y=y_pat,
        groups=g_pat,
        n_splits=int(cfg.folds),
        seed=int(cfg.seed),
        num_classes=len(class_names),
        min_class_per_fold=int(cfg.min_class_per_fold),
        max_tries=int(cfg.max_split_tries),
    )
    print(f"[Train] Selected {cfg.folds}-fold group-stratified split with random_state={split_info['random_state']}.")

    model_cfg = ModelConfig(
        go_dim=int(go_scores.shape[1]),
        tf_dim=int(tf_scores.shape[1]),
        kegg_dim=int(kegg_scores.shape[1]),
        reactome_dim=int(reactome_scores.shape[1]),
        hidden_dim=int(cfg.hidden_dim),
        att_dim=int(cfg.att_dim),
        num_heads=int(cfg.num_heads),
        dropout=float(cfg.dropout),
        num_classes=len(class_names),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    fold_rows = []
    oof_true = []
    oof_logits = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_pids = all_pids[tr_idx].tolist()
        va_pids = all_pids[va_idx].tolist()
        fold_metrics, y_true_fold, logits_fold = train_one_fold(
            cfg=cfg,
            model_cfg=model_cfg,
            train_pids=tr_pids,
            val_pids=va_pids,
            patient_to_indices=patient_to_indices,
            go_scores=go_scores,
            tf_scores=tf_scores,
            kegg_scores=kegg_scores,
            reactome_scores=reactome_scores,
            patient_label=patient_label,
            device=device,
            fold_index=fold_idx,
        )
        fold_rows.append({"fold": int(fold_idx), **fold_metrics})
        oof_true.append(y_true_fold)
        oof_logits.append(logits_fold)

    y_oof = np.concatenate(oof_true, axis=0)
    logits_oof = np.concatenate(oof_logits, axis=0)

    raw_prob_oof = softmax_np(logits_oof)
    temperature = fit_temperature(logits_oof, y_oof)
    calibrated_prob_oof = calibrated_probabilities(logits_oof, temperature)
    print(f"[Train] Fitted temperature for confidence calibration: {temperature:.6f}")

    summary_dir = os.path.join(cfg.output_dir, "summary")
    calibrated_metrics, report = save_summary_artifacts(
        y_true=y_oof,
        y_prob=calibrated_prob_oof,
        class_names=class_names,
        summary_dir=summary_dir,
    )

    save_json(report, os.path.join(summary_dir, "classification_report.json"))

    fold_summary = {
        "acc_mean": mean_std(fold_rows, "acc")[0],
        "acc_std": mean_std(fold_rows, "acc")[1],
        "balanced_acc_mean": mean_std(fold_rows, "balanced_acc")[0],
        "balanced_acc_std": mean_std(fold_rows, "balanced_acc")[1],
        "macro_f1_mean": mean_std(fold_rows, "macro_f1")[0],
        "macro_f1_std": mean_std(fold_rows, "macro_f1")[1],
        "roc_auc_macro_mean": mean_std(fold_rows, "roc_auc_macro")[0],
        "roc_auc_macro_std": mean_std(fold_rows, "roc_auc_macro")[1],
        "pr_auc_macro_mean": mean_std(fold_rows, "pr_auc_macro")[0],
        "pr_auc_macro_std": mean_std(fold_rows, "pr_auc_macro")[1],
        "top2_acc_mean": mean_std(fold_rows, "top2_acc")[0],
        "top2_acc_std": mean_std(fold_rows, "top2_acc")[1],
        "top3_acc_mean": mean_std(fold_rows, "top3_acc")[0],
        "top3_acc_std": mean_std(fold_rows, "top3_acc")[1],
    }

    summary_payload = {
        "class_names": class_names,
        "split_info": split_info,
        "temperature": float(temperature),
        "fold_metrics": fold_rows,
        "cv_summary": fold_summary,
        "oof_metrics_raw": compute_metrics(y_oof, raw_prob_oof, class_names),
        "oof_metrics_calibrated": calibrated_metrics,
    }
    save_json(summary_payload, os.path.join(summary_dir, "metrics.json"))
    print(f"[Train] Saved evaluation summary to {summary_dir}")

    best_epochs = [int(r.get("best_epoch", 0)) for r in fold_rows if int(r.get("best_epoch", 0)) > 0]
    if int(cfg.final_epochs) > 0:
        final_epochs = int(cfg.final_epochs)
    else:
        mean_ep = int(round(float(np.mean(best_epochs)))) if best_epochs else 50
        final_epochs = int(np.clip(mean_ep, 30, int(cfg.max_epochs)))

    final_state = train_final_model(
        cfg=cfg,
        model_cfg=model_cfg,
        all_pids=all_pids.tolist(),
        patient_to_indices=patient_to_indices,
        go_scores=go_scores,
        tf_scores=tf_scores,
        kegg_scores=kegg_scores,
        reactome_scores=reactome_scores,
        patient_label=patient_label,
        device=device,
        epochs=final_epochs,
    )

    checkpoint = {
        "model_state": final_state,
        "model_config": asdict(model_cfg),
        "class_names": class_names,
        "subtype_to_int": subtype_to_int,
        "temperature": float(temperature),
        "final_epochs": int(final_epochs),
        "feature_columns": {
            "go_bp_scores": [str(x) for x in data["score_cols_go_bp"].tolist()],
            "tf_scores": [str(x) for x in data["score_cols_tf"].tolist()],
            "kegg_scores": [str(x) for x in data["score_cols_kegg"].tolist()],
            "reactome_scores": [str(x) for x in data["score_cols_reactome"].tolist()],
        },
        "train_config": asdict(cfg),
    }
    checkpoint_path = os.path.join(cfg.output_dir, "final_model.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"[Train] Saved final checkpoint to {checkpoint_path}")
    print("[Train] Training workflow completed successfully.")
