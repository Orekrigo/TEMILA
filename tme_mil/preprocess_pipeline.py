import re
from pathlib import Path

import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc

from .preprocess_config import PredictPreprocessConfig, TrainPreprocessConfig


def read_gmt(gmt_path: str | Path) -> dict[str, list[str]]:
    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gene_sets[str(parts[0])] = [str(x) for x in parts[2:]]
    return gene_sets


def filter_gene_sets_by_size(gene_sets: dict[str, list[str]], min_size: int, max_size: int) -> dict[str, list[str]]:
    return {
        name: genes
        for name, genes in gene_sets.items()
        if int(min_size) <= len(genes) <= int(max_size)
    }


def filter_gene_sets_by_overlap(
    gene_sets: dict[str, list[str]],
    genes_in_data: pd.Index,
    min_overlap_genes: int,
) -> dict[str, list[str]]:
    filtered = {}
    for name, genes in gene_sets.items():
        overlap = genes_in_data.intersection(genes)
        if len(overlap) >= int(min_overlap_genes):
            filtered[name] = [str(x) for x in overlap.tolist()]
    return filtered


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name))


def gene_sets_to_network(gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, genes in gene_sets.items():
        source = sanitize_name(name)
        for gene in genes:
            rows.append((source, str(gene)))
    return pd.DataFrame(rows, columns=["source", "target"])


def score_with_ulm(adata, net: pd.DataFrame, label: str):
    print(f"[Preprocess] Scoring {label} features with ULM.")
    dc.mt.ulm(data=adata, net=net, tmin=0, empty=True, verbose=True)
    scores = dc.pp.get_obsm(adata, key="score_ulm")
    print(f"[Preprocess] Completed {label} scoring with shape {scores.shape}.")
    return scores


def to_dense_matrix(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def save_npz(path: str | Path, **kwargs) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **kwargs)


def build_train_networks(cfg: TrainPreprocessConfig):
    print(f"[Train] Loading training AnnData from {cfg.adata_path}")
    adata = sc.read(cfg.adata_path)
    print(f"[Train] Loaded training AnnData with {adata.n_obs} cells and {adata.n_vars} genes.")
    genes_in_data = pd.Index(adata.var_names)

    print("[Train] Reading GMT files.")
    go_sets = read_gmt(cfg.go_gmt_path)
    kegg_sets = read_gmt(cfg.kegg_gmt_path)
    reactome_sets = read_gmt(cfg.reactome_gmt_path)

    print("[Train] Filtering gene sets by size and overlap.")
    go_sets = filter_gene_sets_by_size(go_sets, cfg.min_gene_set_size, cfg.max_gene_set_size)
    kegg_sets = filter_gene_sets_by_size(kegg_sets, cfg.min_gene_set_size, cfg.max_gene_set_size)
    reactome_sets = filter_gene_sets_by_size(reactome_sets, cfg.min_gene_set_size, cfg.max_gene_set_size)

    go_sets = filter_gene_sets_by_overlap(go_sets, genes_in_data, cfg.min_overlap_genes)
    kegg_sets = filter_gene_sets_by_overlap(kegg_sets, genes_in_data, cfg.min_overlap_genes)
    reactome_sets = filter_gene_sets_by_overlap(reactome_sets, genes_in_data, cfg.min_overlap_genes)

    go_net = gene_sets_to_network(go_sets)
    kegg_net = gene_sets_to_network(kegg_sets)
    reactome_net = gene_sets_to_network(reactome_sets)
    print(f"[Train] GO pathways retained: {len(go_sets)}")
    print(f"[Train] KEGG pathways retained: {len(kegg_sets)}")
    print(f"[Train] Reactome pathways retained: {len(reactome_sets)}")
    print("[Train] Downloading DoRothEA network for human.")
    dorothea_tf = dc.op.dorothea(organism="human")

    network_dir = Path(cfg.output_network_dir)
    network_dir.mkdir(parents=True, exist_ok=True)
    go_net.to_csv(network_dir / "go_bp_net_train_fixed.csv")
    kegg_net.to_csv(network_dir / "kegg_net_train_fixed.csv")
    reactome_net.to_csv(network_dir / "reactome_net_train_fixed.csv")
    dorothea_tf.to_csv(network_dir / "DoRothEA_TF_net_train_fixed.csv")
    print(f"[Train] Saved reusable networks to {network_dir}")

    return adata, go_net, dorothea_tf, kegg_net, reactome_net


def run_train_preprocess(cfg: TrainPreprocessConfig) -> None:
    print("[Train] Step 1/2: building training features.")
    adata, go_net, dorothea_tf, kegg_net, reactome_net = build_train_networks(cfg)

    scores_tf = score_with_ulm(adata, dorothea_tf, "TF")
    scores_kegg = score_with_ulm(adata, kegg_net, "KEGG")
    scores_reactome = score_with_ulm(adata, reactome_net, "Reactome")
    scores_go = score_with_ulm(adata, go_net, "GO")

    save_npz(
        cfg.output_npz_path,
        go_bp_scores=to_dense_matrix(scores_go.X),
        tf_scores=to_dense_matrix(scores_tf.X),
        kegg_scores=to_dense_matrix(scores_kegg.X),
        reactome_scores=to_dense_matrix(scores_reactome.X),
        patients=adata.obs[cfg.patient_col].to_numpy(),
        subtypes=adata.obs[cfg.subtype_col].to_numpy(),
        datasets=adata.obs[cfg.dataset_col].to_numpy(),
        score_cols_go_bp=np.array(scores_go.var_names.to_list(), dtype=object),
        score_cols_tf=np.array(scores_tf.var_names.to_list(), dtype=object),
        score_cols_kegg=np.array(scores_kegg.var_names.to_list(), dtype=object),
        score_cols_reactome=np.array(scores_reactome.var_names.to_list(), dtype=object),
    )
    print(f"[Train] Saved training feature matrix to {cfg.output_npz_path}")


def load_prediction_networks(network_dir: str | Path):
    network_dir = Path(network_dir)
    go_bp = pd.read_csv(network_dir / "go_bp_net_train_fixed.csv", index_col=0)
    dorothea_tf = pd.read_csv(network_dir / "DoRothEA_TF_net_train_fixed.csv", index_col=0)
    kegg = pd.read_csv(network_dir / "kegg_net_train_fixed.csv", index_col=0)
    reactome = pd.read_csv(network_dir / "reactome_net_train_fixed.csv", index_col=0)
    return go_bp, dorothea_tf, kegg, reactome


def run_predict_preprocess(cfg: PredictPreprocessConfig) -> None:
    print("[Predict] Step 1/2: building prediction features.")
    print(f"[Predict] Loading reusable networks from {cfg.network_dir}")
    go_bp, dorothea_tf, kegg, reactome = load_prediction_networks(cfg.network_dir)
    print(f"[Predict] Loading prediction AnnData from {cfg.adata_path}")
    adata = sc.read(cfg.adata_path)
    print(f"[Predict] Loaded prediction AnnData with {adata.n_obs} cells and {adata.n_vars} genes.")

    scores_tf = score_with_ulm(adata, dorothea_tf, "TF")
    scores_kegg = score_with_ulm(adata, kegg, "KEGG")
    scores_reactome = score_with_ulm(adata, reactome, "Reactome")
    scores_go = score_with_ulm(adata, go_bp, "GO")

    save_npz(
        cfg.output_npz_path,
        go_bp_scores=to_dense_matrix(scores_go.X),
        tf_scores=to_dense_matrix(scores_tf.X),
        kegg_scores=to_dense_matrix(scores_kegg.X),
        reactome_scores=to_dense_matrix(scores_reactome.X),
        patients=adata.obs[cfg.patient_col].to_numpy(),
        score_cols_go_bp=np.array(scores_go.var_names.to_list(), dtype=object),
        score_cols_tf=np.array(scores_tf.var_names.to_list(), dtype=object),
        score_cols_kegg=np.array(scores_kegg.var_names.to_list(), dtype=object),
        score_cols_reactome=np.array(scores_reactome.var_names.to_list(), dtype=object),
    )
    print(f"[Predict] Saved prediction feature matrix to {cfg.output_npz_path}")
