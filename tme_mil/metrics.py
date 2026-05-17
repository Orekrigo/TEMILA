import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
CM_TEXT_FONTSIZE = 15
LINE_WIDTH = 2.4
CMAP = plt.cm.Blues


def safe_auc_pr(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_bin = np.zeros((len(y_true), num_classes), dtype=int)
    y_bin[np.arange(len(y_true)), y_true] = 1

    per_auc = {}
    per_ap = {}
    for c in range(num_classes):
        pos = int(y_bin[:, c].sum())
        if 0 < pos < len(y_true):
            per_auc[c] = float(roc_auc_score(y_bin[:, c], y_prob[:, c]))
            per_ap[c] = float(average_precision_score(y_bin[:, c], y_prob[:, c]))
        else:
            per_auc[c] = float("nan")
            per_ap[c] = float("nan")

    roc_auc_macro = float(np.nanmean(list(per_auc.values())))
    pr_auc_macro = float(np.nanmean(list(per_ap.values())))

    y_flat = y_bin.ravel()
    p_flat = y_prob.ravel()
    if len(np.unique(y_flat)) >= 2:
        roc_auc_micro = float(roc_auc_score(y_flat, p_flat))
        pr_auc_micro = float(average_precision_score(y_flat, p_flat))
    else:
        roc_auc_micro = float("nan")
        pr_auc_micro = float("nan")

    out = {
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_micro": roc_auc_micro,
        "pr_auc_macro": pr_auc_macro,
        "pr_auc_micro": pr_auc_micro,
    }
    for c in range(num_classes):
        out[f"roc_auc_class_{c}"] = per_auc[c]
        out[f"pr_auc_class_{c}"] = per_ap[c]
    return out


def topk_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 2) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if int(k) <= 1:
        return float(np.mean(np.argmax(y_prob, axis=1) == y_true))
    topk = np.argsort(-y_prob, axis=1)[:, : int(k)]
    hit = (topk == y_true[:, None]).any(axis=1)
    return float(hit.mean())


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "top2_acc": float(topk_accuracy(y_true, y_prob, k=2)),
        "top3_acc": float(topk_accuracy(y_true, y_prob, k=3)),
        **safe_auc_pr(y_true, y_prob, len(class_names)),
    }
    return metrics


def build_classification_report(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> dict:
    y_pred = np.argmax(y_prob, axis=1)
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )


def _build_binary_matrix(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    y_bin = np.zeros((len(y_true), num_classes), dtype=int)
    y_bin[np.arange(len(y_true)), y_true] = 1
    return y_bin


def _fmt_metric(name: str, value: float) -> str:
    if np.isfinite(value):
        return f"{name} = {value:.2f}"
    return f"{name} = NA"


def _style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.6, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)


def _save_figure(fig, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path, dpi=600, bbox_inches="tight")



def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], out_path: str) -> None:
    num_classes = len(class_names)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_bin = _build_binary_matrix(y_true, num_classes)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    for c in range(num_classes):
        pos = int(y_bin[:, c].sum())
        if 0 < pos < len(y_true):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
            auc_val = roc_auc_score(y_bin[:, c], y_prob[:, c])
            ax.plot(fpr, tpr, linewidth=LINE_WIDTH, label=f"{class_names[c]} ({_fmt_metric('AUC', auc_val)})")
    if len(np.unique(y_bin.ravel())) >= 2:
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        auc_val = roc_auc_score(y_bin.ravel(), y_prob.ravel())
        ax.plot(fpr, tpr, linewidth=LINE_WIDTH + 0.2, label=f"micro ({_fmt_metric('AUC', auc_val)})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.2, label="chance")
    ax.set_xlabel("False Positive Rate", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=LABEL_FONTSIZE)
    ax.set_title("ROC Curves", fontsize=TITLE_FONTSIZE, pad=12)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, frameon=True)
    _style_axis(ax)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)



def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], out_path: str) -> None:
    num_classes = len(class_names)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_bin = _build_binary_matrix(y_true, num_classes)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    for c in range(num_classes):
        pos = int(y_bin[:, c].sum())
        if 0 < pos < len(y_true):
            precision, recall, _ = precision_recall_curve(y_bin[:, c], y_prob[:, c])
            ap_val = average_precision_score(y_bin[:, c], y_prob[:, c])
            ax.plot(recall, precision, linewidth=LINE_WIDTH, label=f"{class_names[c]} ({_fmt_metric('AP', ap_val)})")
    if len(np.unique(y_bin.ravel())) >= 2 and int(y_bin.ravel().sum()) > 0:
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
        ap_val = average_precision_score(y_bin.ravel(), y_prob.ravel())
        ax.plot(recall, precision, linewidth=LINE_WIDTH + 0.2, label=f"micro ({_fmt_metric('AP', ap_val)})")
    ax.set_xlabel("Recall", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=LABEL_FONTSIZE)
    ax.set_title("Precision-Recall Curves", fontsize=TITLE_FONTSIZE, pad=12)
    ax.legend(loc="lower left", fontsize=LEGEND_FONTSIZE, frameon=True)
    _style_axis(ax)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)



def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    out_path: str,
    normalize_mode: str,
) -> None:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).astype(np.float64)

    if normalize_mode == "true":
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm_disp = cm / denom
        title = "Confusion Matrix (norm by true)"
    elif normalize_mode == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        cm_disp = cm / denom
        title = "Confusion Matrix (norm by pred)"
    else:
        raise ValueError("normalize_mode must be 'true' or 'pred'.")

    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=CMAP(norm(cm_disp[i, j])),
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)
    ax.set_xlabel("Predicted label", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True label", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(-0.5, len(class_names) - 0.5)
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    ax.set_yticklabels(class_names, fontsize=TICK_FONTSIZE)

    thresh = cm_disp.max() / 2.0 if cm_disp.size else 0.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_disp[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_disp[i, j] > thresh else "black",
                fontsize=CM_TEXT_FONTSIZE,
                fontweight="bold",
            )

    _style_axis(ax)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)



def save_summary_artifacts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    summary_dir: str,
) -> tuple[dict, dict]:
    os.makedirs(summary_dir, exist_ok=True)
    metrics = compute_metrics(y_true, y_prob, class_names)
    report = build_classification_report(y_true, y_prob, class_names)
    plot_roc_curves(y_true, y_prob, class_names, os.path.join(summary_dir, "roc_curves.pdf"))
    plot_pr_curves(y_true, y_prob, class_names, os.path.join(summary_dir, "pr_curves.pdf"))
    plot_confusion_matrix(y_true, y_prob, class_names, os.path.join(summary_dir, "confusion_matrix_true_norm.pdf"), "true")
    plot_confusion_matrix(y_true, y_prob, class_names, os.path.join(summary_dir, "confusion_matrix_pred_norm.pdf"), "pred")
    return metrics, report
