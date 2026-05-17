import argparse
from pathlib import Path

from tme_mil.config import PredictConfig
from tme_mil.preprocess_config import PredictPreprocessConfig
from tme_mil.preprocess_pipeline import run_predict_preprocess
from tme_mil.predict_pipeline import run_prediction


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "train_results" / "final_model.pt"
DEFAULT_OUTPUT_DIR = ROOT / "predict_results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run preprocessing and subtype prediction for a user-provided .h5ad file.",
    )
    parser.add_argument(
        "--adata",
        required=True,
        help="Path to the input .h5ad file.",
    )
    parser.add_argument(
        "--patient-col",
        default="Patient",
        help="Column name in adata.obs that stores patient identifiers.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for prediction outputs.",
    )
    parser.add_argument(
        "--feature-npz",
        default=None,
        help="Optional path for the intermediate feature NPZ file.",
    )
    return parser


def resolve_feature_npz_path(adata_path: Path, output_dir: Path, feature_npz: str | None) -> Path:
    if feature_npz:
        return Path(feature_npz).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{adata_path.stem}_features.npz"


def main() -> None:
    args = build_parser().parse_args()

    adata_path = Path(args.adata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    feature_npz_path = resolve_feature_npz_path(adata_path, output_dir, args.feature_npz)

    preprocess_config = PredictPreprocessConfig(
        adata_path=str(adata_path),
        network_dir=str(ROOT / "data"),
        output_npz_path=str(feature_npz_path),
        patient_col=str(args.patient_col),
    )

    predict_config = PredictConfig(
        data_path=str(feature_npz_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        use_amp=False,
        chunk_size=0,
        align_fill_value=0.0,
    )

    print("[Predict] Input AnnData:", adata_path)
    print("[Predict] Patient column:", args.patient_col)
    print("[Predict] Intermediate feature file:", feature_npz_path)
    print("[Predict] Checkpoint:", checkpoint_path)
    print("[Predict] Output directory:", output_dir)

    run_predict_preprocess(preprocess_config)
    run_prediction(predict_config)


if __name__ == "__main__":
    main()
