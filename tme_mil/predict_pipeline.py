import csv
import os

import numpy as np
import torch

from .calibration import calibrated_probabilities, confidence_features
from .config import ModelConfig, PredictConfig
from .data import BagDataset, align_by_cols, build_eval_loader, build_patient_maps, require_keys
from .model import FourBranchGatedMIL, forward_bag_all_cells


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint format is invalid")
    required = ["model_state", "model_config", "class_names", "feature_columns"]
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise KeyError(f"checkpoint missing keys: {missing}")
    return checkpoint


@torch.no_grad()
def predict_logits_loader(
    model: FourBranchGatedMIL,
    loader,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
):
    model.eval()
    logits_all = []
    patient_ids = []

    for (go, tf, kegg, reactome), pid in loader:
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
        patient_ids.append(str(pid[0]))

    return np.vstack(logits_all), patient_ids


def run_prediction(cfg: PredictConfig) -> None:
    print("[Predict] Step 2/2: model inference.")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[Predict] Loading checkpoint from {cfg.checkpoint_path}")
    checkpoint = load_checkpoint(cfg.checkpoint_path)
    feature_columns = checkpoint["feature_columns"]
    class_names = [str(x) for x in checkpoint["class_names"]]
    temperature = float(checkpoint.get("temperature", 1.0))

    print(f"[Predict] Loading prediction feature file from {cfg.data_path}")
    data = np.load(cfg.data_path, allow_pickle=True)
    require_keys(
        data,
        [
            "go_bp_scores",
            "tf_scores",
            "kegg_scores",
            "reactome_scores",
            "patients",
            "score_cols_go_bp",
            "score_cols_tf",
            "score_cols_kegg",
            "score_cols_reactome",
        ],
    )

    go_scores, go_info = align_by_cols(
        data["go_bp_scores"],
        data["score_cols_go_bp"],
        feature_columns["go_bp_scores"],
        fill_value=float(cfg.align_fill_value),
    )
    tf_scores, tf_info = align_by_cols(
        data["tf_scores"],
        data["score_cols_tf"],
        feature_columns["tf_scores"],
        fill_value=float(cfg.align_fill_value),
    )
    kegg_scores, kegg_info = align_by_cols(
        data["kegg_scores"],
        data["score_cols_kegg"],
        feature_columns["kegg_scores"],
        fill_value=float(cfg.align_fill_value),
    )
    reactome_scores, reactome_info = align_by_cols(
        data["reactome_scores"],
        data["score_cols_reactome"],
        feature_columns["reactome_scores"],
        fill_value=float(cfg.align_fill_value),
    )

    print(
        f"[Predict] Feature alignment | GO {go_info['num_common']}/{go_info['target_dim']} | "
        f"TF {tf_info['num_common']}/{tf_info['target_dim']} | "
        f"KEGG {kegg_info['num_common']}/{kegg_info['target_dim']} | "
        f"Reactome {reactome_info['num_common']}/{reactome_info['target_dim']}"
    )

    patients = np.array([str(x) for x in data["patients"]], dtype=object)
    patient_to_indices, _, _ = build_patient_maps(patients=patients)
    patient_ids = sorted(patient_to_indices.keys())
    print(f"[Predict] Patients queued for inference: {len(patient_ids)}")

    ds = BagDataset(
        patient_ids=patient_ids,
        patient_to_indices=patient_to_indices,
        go_scores=go_scores,
        tf_scores=tf_scores,
        kegg_scores=kegg_scores,
        reactome_scores=reactome_scores,
        patient_label=None,
    )
    loader = build_eval_loader(ds)

    model_cfg = ModelConfig(**checkpoint["model_config"])
    model = FourBranchGatedMIL(model_cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[Predict] Using device: {device}")

    logits, ordered_patient_ids = predict_logits_loader(
        model=model,
        loader=loader,
        device=device,
        chunk_size=int(cfg.chunk_size),
        use_amp=bool(cfg.use_amp and device.type == "cuda"),
    )

    calibrated_prob = calibrated_probabilities(logits, temperature)
    confidence, margin, _ = confidence_features(calibrated_prob)

    out_csv = os.path.join(cfg.output_dir, "patient_predictions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "patient_id",
            "pred_subtype",
            "pred_confidence",
            "second_subtype",
            "confidence_margin",
        ]
        writer.writerow(header)

        for i, pid in enumerate(ordered_patient_ids):
            prob = calibrated_prob[i]
            order = np.argsort(-prob)
            top1 = int(order[0])
            top2 = int(order[1]) if len(order) > 1 else top1

            row = [
                pid,
                class_names[top1],
                f"{float(confidence[i]):.6f}",
                class_names[top2],
                f"{float(margin[i]):.6f}",
            ]
            writer.writerow(row)

    print(f"[Predict] Applied calibrated confidence with temperature={temperature:.6f}")
    print(f"[Predict] Saved patient-level predictions to {out_csv}")
    print("[Predict] Prediction workflow completed successfully.")
