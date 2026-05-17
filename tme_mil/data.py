from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def require_keys(npz, keys: List[str]) -> None:
    missing = [k for k in keys if k not in npz.files]
    if missing:
        raise KeyError(f"NPZ missing keys: {missing}")


def to_str_list(arr: np.ndarray) -> List[str]:
    return [str(x) for x in np.asarray(arr, dtype=object).tolist()]


def align_by_cols(
    scores: np.ndarray,
    cols: np.ndarray,
    target_cols: List[str],
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    cols_list = to_str_list(cols)
    target_list = [str(x) for x in target_cols]
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got shape={scores.shape}")

    col2idx = {c: i for i, c in enumerate(cols_list)}
    aligned = np.full((scores.shape[0], len(target_list)), fill_value, dtype=scores.dtype)

    common_src = []
    common_tgt = []
    missing_cols = []
    for j, c in enumerate(target_list):
        i = col2idx.get(c)
        if i is None:
            missing_cols.append(c)
        else:
            common_src.append(i)
            common_tgt.append(j)

    if common_src:
        aligned[:, common_tgt] = scores[:, common_src]

    target_set = set(target_list)
    extra_cols = [c for c in cols_list if c not in target_set]

    info = {
        "src_dim": int(scores.shape[1]),
        "target_dim": int(len(target_list)),
        "num_common": int(len(common_src)),
        "num_missing": int(len(missing_cols)),
        "num_extra_src": int(len(extra_cols)),
        "missing_example": missing_cols[:20],
        "extra_src_example": extra_cols[:20],
        "fill_value": float(fill_value),
    }
    return aligned, info


def build_patient_maps(
    patients: np.ndarray,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
):
    patient_to_indices: Dict[str, List[int]] = {}
    patient_label: Dict[str, int] = {}
    patient_group: Dict[str, str] = {}

    for i, pid_raw in enumerate(patients):
        pid = str(pid_raw)
        patient_to_indices.setdefault(pid, []).append(int(i))

        if labels is not None:
            y = int(labels[i])
            if pid in patient_label and patient_label[pid] != y:
                raise ValueError(f"Patient {pid} has inconsistent labels across cells")
            patient_label[pid] = y

        if groups is not None:
            g = str(groups[i])
            if pid in patient_group and patient_group[pid] != g:
                raise ValueError(f"Patient {pid} has inconsistent groups across cells")
            patient_group[pid] = g

    return patient_to_indices, patient_label, patient_group


class BagDataset(Dataset):
    def __init__(
        self,
        patient_ids: List[str],
        patient_to_indices: Dict[str, List[int]],
        go_scores: np.ndarray,
        tf_scores: np.ndarray,
        kegg_scores: np.ndarray,
        reactome_scores: np.ndarray,
        patient_label: Dict[str, int] | None = None,
    ):
        self.patient_ids = list(patient_ids)
        self.patient_to_indices = patient_to_indices
        self.go_scores = go_scores
        self.tf_scores = tf_scores
        self.kegg_scores = kegg_scores
        self.reactome_scores = reactome_scores
        self.patient_label = patient_label

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        indices = self.patient_to_indices[pid]
        go = torch.tensor(self.go_scores[indices], dtype=torch.float32)
        tf = torch.tensor(self.tf_scores[indices], dtype=torch.float32)
        kegg = torch.tensor(self.kegg_scores[indices], dtype=torch.float32)
        reactome = torch.tensor(self.reactome_scores[indices], dtype=torch.float32)

        if self.patient_label is None:
            return (go, tf, kegg, reactome), pid

        y = torch.tensor(self.patient_label[pid], dtype=torch.long)
        return (go, tf, kegg, reactome), y, pid


def bag_collate(batch):
    xs, ys, pids = zip(*batch)
    go_list = [x[0] for x in xs]
    tf_list = [x[1] for x in xs]
    kegg_list = [x[2] for x in xs]
    react_list = [x[3] for x in xs]
    y = torch.stack(list(ys), dim=0)
    return (go_list, tf_list, kegg_list, react_list), y, list(pids)


def build_train_loader(train_ds: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=bag_collate,
        num_workers=0,
    )


def build_eval_loader(ds: Dataset) -> DataLoader:
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
