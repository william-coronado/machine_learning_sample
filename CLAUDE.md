# Machine Learning Sample — Claude Guide

## Project Overview

This project demonstrates machine learning techniques across three domains: binary fraud classification, deep-learning autoencoders, and advanced tabular feature engineering. It includes Jupyter notebooks, a reusable `ml_pipeline` framework, and a `feature_engineering` package.

### Notebooks

| Notebook | Technique | Dataset |
|---|---|---|
| `online_payment_fraud_detection.ipynb` | Binary classification (LR, XGBoost, RF), ROC-AUC, confusion matrix | Telco fraud CSV (~6 M rows, Google Drive) |
| `autoencoder_hands_on.ipynb` | Fully-connected + denoising autoencoders in PyTorch | MNIST (torchvision) |
| `ieee_cis_feature_engineering.ipynb` | Missingness, frequency/target encoding, aggregations, temporal features, PCA, feature selection, model lift | Synthetic IEEE-CIS schema replica |

### Packages

- **`ml_pipeline/`** — Modular pipeline framework used by `online_payment_fraud_detection.ipynb` and `run_fraud_pipeline.py`:
  - `core/` — Domain-agnostic building blocks: `base_pipeline.py`, `evaluation.py`, `model_utils.py`, `preprocessing.py`
  - `use_cases/` — Telco fraud implementation: `fraud_detection.py` (config, feature engineering, model selection), `fraud_pipeline.py` (concrete `BasePipeline` subclass)
- **`feature_engineering/`** — IEEE-CIS feature engineering utilities: `aggregations.py`, `encoders.py`, `preprocessing.py`, `selection.py`, `temporal.py`, `utils.py`

### Entry Points

- **`run_fraud_pipeline.py`** — CLI for the fraud detection pipeline:

  ```bash
  # Run with defaults
  python3 run_fraud_pipeline.py

  # Override data path, split, and best model index
  python3 run_fraud_pipeline.py --data-path data/online_payment_fraud_data.csv \
                                --test-size 0.2 \
                                --best-model-idx 1
  ```

  CLI flags: `--data-path`, `--test-size`, `--random-state`, `--threshold`, `--best-model-idx`

### Dataset Sources

| Dataset | Source | Local cache |
|---|---|---|
| Telco fraud CSV | Google Drive (auto-downloaded via `gdown`) | `data/online_payment_fraud_data.csv` |
| MNIST | `torchvision.datasets` (auto-downloaded) | torchvision default cache |
| IEEE-CIS | Synthetic replica generated in-notebook | In-memory only |

## Dependencies

All dependencies are Python packages installed via pip. The session-start hook installs them automatically in remote sessions.

Key packages:
- `numpy`, `pandas`, `matplotlib`, `seaborn` — Data analysis and visualization
- `scikit-learn` — Machine learning models and metrics
- `xgboost` — Gradient boosted trees
- `gdown` — Google Drive dataset download
- `torch`, `torchvision` — PyTorch deep learning
- `jupyter`, `nbformat`, `nbconvert`, `ipykernel` — Notebook tooling
- `flake8`, `nbqa` — Linting
- `nbmake` — Notebook test execution

## Linting

Run flake8 on notebooks via nbqa:

```bash
python3 -m nbqa flake8 autoencoder_hands_on.ipynb --max-line-length=120 --ignore=E501,W503,E402
python3 -m nbqa flake8 ieee_cis_feature_engineering.ipynb --max-line-length=120 --ignore=E501,W503,E402
```

Run flake8 on Python packages directly:

```bash
python3 -m flake8 feature_engineering/ --max-line-length=120 --ignore=E501,W503,E402
python3 -m flake8 ml_pipeline/ --max-line-length=120 --ignore=E501,W503,E402
python3 -m flake8 run_fraud_pipeline.py --max-line-length=120 --ignore=E501,W503,E402
```

## Testing

Run notebooks as tests using nbmake:

```bash
python3 -m pytest --nbmake autoencoder_hands_on.ipynb -v
python3 -m pytest --nbmake online_payment_fraud_detection.ipynb -v
python3 -m pytest --nbmake ieee_cis_feature_engineering.ipynb -v
```
