from dataclasses import dataclass


@dataclass
class TrainPreprocessConfig:
    adata_path: str = "../output/TME_Tumor.h5ad"
    go_gmt_path: str = "GMT/c5.go.bp.v2025.1.Hs.symbols.gmt"
    kegg_gmt_path: str = "GMT/kegg_hsa.gmt"
    reactome_gmt_path: str = "GMT/ReactomePathways.gmt"
    output_npz_path: str = "data/tme_mil_go_tf_kegg_reactome.npz"
    output_network_dir: str = "data"
    patient_col: str = "Patient"
    subtype_col: str = "Subtype"
    dataset_col: str = "Dataset"
    min_gene_set_size: int = 20
    max_gene_set_size: int = 200
    min_overlap_genes: int = 5


@dataclass
class PredictPreprocessConfig:
    adata_path: str = "../Validation/output/sc_after_annotation.h5ad"
    network_dir: str = "data"
    output_npz_path: str = "data/tme_mil_go_tf_kegg_reactome_validation.npz"
    patient_col: str = "Patient"
