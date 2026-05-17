from pathlib import Path

from tme_mil.config import TrainConfig
from tme_mil.preprocess_config import TrainPreprocessConfig
from tme_mil.preprocess_pipeline import run_train_preprocess
from tme_mil.train_pipeline import run_training


ROOT = Path(__file__).resolve().parent

PREPROCESS_CONFIG = TrainPreprocessConfig(
    adata_path=str(ROOT / "data" / "TME_Tumor.h5ad"),
    go_gmt_path=str(ROOT / "GMT" / "c5.go.bp.v2025.1.Hs.symbols.gmt"),
    kegg_gmt_path=str(ROOT / "GMT" / "kegg_hsa.gmt"),
    reactome_gmt_path=str(ROOT / "GMT" / "ReactomePathways.gmt"),
    output_npz_path=str(ROOT / "data" / "tme_mil_go_tf_kegg_reactome.npz"),
    output_network_dir=str(ROOT / "data"),
    patient_col="Patient",
    subtype_col="Subtype",
    dataset_col="Dataset",
    min_gene_set_size=20,
    max_gene_set_size=200,
    min_overlap_genes=5,
)

TRAIN_CONFIG = TrainConfig(
    data_path=str(ROOT / "data" / "tme_mil_go_tf_kegg_reactome.npz"),
    output_dir=str(ROOT / "train_results"),
    seed=2026,
    folds=5,
    hidden_dim=64,
    num_heads=4,
    att_dim=16,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-2,
    max_epochs=200,
    patience=20,
    min_delta=1e-4,
    min_class_per_fold=1,
    max_split_tries=1000,
    train_bag_batch_size=4,
    use_amp=False,
    chunk_size=0,
    final_epochs=0,
)


def main() -> None:
    run_train_preprocess(PREPROCESS_CONFIG)
    run_training(TRAIN_CONFIG)


if __name__ == "__main__":
    main()
