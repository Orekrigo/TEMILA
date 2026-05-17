from dataclasses import dataclass


@dataclass
class ModelConfig:
    go_dim: int
    tf_dim: int
    kegg_dim: int
    reactome_dim: int
    hidden_dim: int = 64
    att_dim: int = 16
    num_heads: int = 4
    dropout: float = 0.1
    num_classes: int = 4


@dataclass
class TrainConfig:
    data_path: str = "data/tme_mil_go_tf_kegg_reactome.npz"
    output_dir: str = "train_results"
    seed: int = 2026
    folds: int = 5
    hidden_dim: int = 64
    num_heads: int = 4
    att_dim: int = 16
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-4
    min_class_per_fold: int = 1
    max_split_tries: int = 1000
    train_bag_batch_size: int = 4
    use_amp: bool = False
    chunk_size: int = 0
    final_epochs: int = 0


@dataclass
class PredictConfig:
    data_path: str = "data/tme_mil_go_tf_kegg_reactome_validation.npz"
    checkpoint_path: str = "train_results/final_model.pt"
    output_dir: str = "predict_results"
    use_amp: bool = False
    chunk_size: int = 0
    align_fill_value: float = 0.0
