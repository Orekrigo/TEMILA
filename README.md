# TEMILA

**TEMILA** (**T**umor micro**E**nvironment **M**ultiple-**I**nstance **L**earning with **A**ttention) is a pretrained framework for patient-level clear cell renal cell carcinoma (ccRCC) subtype prediction from non-epithelial tumor microenvironment (TME) single-cell RNA-seq data.

This repository is intended mainly for applying the pretrained TEMILA model to user-provided `.h5ad` files. A training pipeline is also included for users who want to retrain the model on a labeled cohort.

## Overview

TEMILA converts single-cell expression data into GO, transcription factor (TF), KEGG, and Reactome activity scores, groups cells by patient, and uses an attention-based multiple-instance learning model to generate patient-level subtype predictions.

The prediction workflow is:

1. Load a user-provided `.h5ad` file.
2. Compute pathway and TF activity features using fixed networks from the training workflow.
3. Group cells by patient identifier.
4. Aggregate cell-level representations with attention-based multiple-instance learning.
5. Export patient-level subtype predictions and confidence scores.

## Included Prediction Files

The pretrained prediction workflow requires the model checkpoint and the fixed feature-scoring networks generated during training. This repository includes:

```text
train_results/final_model.pt
data/go_bp_net_train_fixed.csv
data/DoRothEA_TF_net_train_fixed.csv
data/kegg_net_train_fixed.csv
data/reactome_net_train_fixed.csv
```

The checkpoint stores the trained TEMILA model, class names, feature columns, and confidence-calibration temperature. The four CSV files store the fixed feature-scoring networks used during training and are required for preprocessing new prediction data.

The `GMT/` files are used only by the training workflow to build feature-scoring networks. For pretrained prediction, use the fixed CSV files in `data/`; do not regenerate feature networks from `GMT/` unless retraining the model.

## Installation

Create a Python 3.12 environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run prediction on a `.h5ad` file:

```bash
python predict_run.py --adata "/path/to/sample.h5ad"
```

By default, TEMILA expects patient identifiers in `adata.obs["Patient"]`. If your file uses a different column name, pass it explicitly:

```bash
python predict_run.py --adata "/path/to/sample.h5ad" --patient-col "patient_id"
```

Optional arguments:

```bash
python predict_run.py \
  --adata "/path/to/sample.h5ad" \
  --patient-col "patient_id" \
  --checkpoint "train_results/final_model.pt" \
  --output-dir "predict_results"
```

## Input AnnData Requirements

The input `.h5ad` file must satisfy the following requirements:

- Rows in `adata` represent cells and columns represent genes.
- The input should contain non-epithelial TME cells from ccRCC samples. TEMILA does not automatically remove epithelial cells or other unwanted cell populations.
- `adata.obs` must contain a patient identifier column. The default column name is `Patient`.
- `adata.var_names` should be human gene symbols, using the same gene-naming convention as the training data.

Prediction does not require `Subtype`, `Dataset`, or `cell_type` columns.

### Expression Matrix in `adata.X`

TEMILA uses `adata.X` directly for GO, TF, KEGG, and Reactome activity scoring. The prediction code does not perform normalization, log transformation, scaling, batch correction, or layer selection before scoring.

The released model expects `adata.X` to contain normalized and log-transformed expression values. Before saving the `.h5ad` file for prediction, preprocess the AnnData object with:

```python
scanpy.pp.normalize_total(adata)
scanpy.pp.log1p(adata)
```

Raw count matrices should not be passed directly to TEMILA. The training script also uses the training AnnData object's `adata.X` directly, so training data should be normalized and log-transformed in the same way before running `train_run.py`.

If the desired matrix is stored in `adata.layers`, copy that layer to `adata.X` before running prediction.

## Output

Prediction writes patient-level results to:

```text
predict_results/patient_predictions.csv
```

The output table contains:

- `patient_id`: patient identifier from the input AnnData object.
- `pred_subtype`: predicted patient subtype.
- `pred_confidence`: calibrated confidence for the predicted subtype.
- `second_subtype`: subtype with the second-highest calibrated confidence.
- `confidence_margin`: difference between the top and second confidence values.

`pred_confidence` is a calibrated model confidence score, not an absolute clinical probability. Interpret it together with `confidence_margin`, especially when applying the model to data from a different platform, cohort, or preprocessing workflow.

The workflow also writes an intermediate feature file unless a custom `--feature-npz` path is provided.

## Retraining

Routine use of TEMILA does not require retraining. To train a new model on a labeled cohort, update the paths and settings in `train_run.py`, then run:

```bash
python train_run.py
```

Training expects a labeled `.h5ad` file with these columns in `adata.obs`:

- `Patient`: patient identifier.
- `Subtype`: patient-level subtype label.
- `Dataset`: cohort or dataset identifier used for group-stratified cross-validation.

The training pipeline builds fixed activity-scoring networks, generates pathway and TF features, trains TEMILA, evaluates cross-validation performance, and exports `train_results/final_model.pt`.
