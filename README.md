# Machine Learning Sample Project

## Project Overview

This project demonstrates machine learning techniques across three domains: binary fraud classification, deep-learning autoencoders, and advanced tabular feature engineering. It includes Jupyter notebooks, a reusable `ml_pipeline` framework, and a `feature_engineering` package.

## Project Structure

```
machine_learning_sample/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── run_fraud_pipeline.py              # CLI entry point for fraud detection
├── online_payment_fraud_detection.ipynb
├── autoencoder_hands_on.ipynb
├── ieee_cis_feature_engineering.ipynb
├── ml_pipeline/                       # Reusable ML pipeline framework
│   ├── core/
│   │   ├── base_pipeline.py           # Abstract BasePipeline template
│   │   ├── evaluation.py              # ROC-AUC scoring, confusion matrix
│   │   ├── model_utils.py             # Model training helpers
│   │   └── preprocessing.py           # Encoding, splitting utilities
│   └── use_cases/
│       ├── fraud_detection.py         # Telco-specific config, feature engineering, models
│       └── fraud_pipeline.py          # Concrete FraudDetectionPipeline
├── feature_engineering/               # IEEE-CIS feature engineering package
│   ├── aggregations.py                # Entity-level group statistics
│   ├── encoders.py                    # Frequency & target encoding
│   ├── preprocessing.py               # Imputation, sentinel encoding
│   ├── selection.py                   # Variance/correlation pruning, permutation importance
│   ├── temporal.py                    # Temporal features + sin/cos encoding
│   └── utils.py                       # Shared helpers, synthetic data loader
└── data/
    └── online_payment_fraud_data.csv  # Auto-downloaded on first run
```

## Featured Notebooks

### 1. Online Payment Fraud Detection (`online_payment_fraud_detection.ipynb`)

A complete binary classification pipeline for detecting fraudulent online payment transactions, using the telco fraud dataset (~6 M rows). Covers data loading, feature engineering, model training (Logistic Regression, XGBoost, Random Forest), ROC-AUC evaluation, and confusion matrix visualisation.

The same pipeline is also runnable from the command line via `run_fraud_pipeline.py` (see [Running the Fraud Pipeline](#running-the-fraud-pipeline) below).

### 2. Autoencoders using PyTorch (`autoencoder_hands_on.ipynb`)

A hands-on guide to autoencoders in PyTorch on the MNIST dataset:

1. Loading and preprocessing MNIST.
2. Designing and training a fully-connected autoencoder for image reconstruction.
3. Visualising reconstructions and exploring the latent space.
4. Extending to a denoising autoencoder that reconstructs clean images from noisy inputs.
5. Suggestions for further experimentation (convolutional layers, sparsity penalties, VAEs).

### 3. IEEE-CIS Feature Engineering (`ieee_cis_feature_engineering.ipynb`)

A senior data-science reference guide to feature engineering for tabular fraud detection, using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset schema. Because the dataset requires Kaggle authentication, the notebook uses a synthetic replica that mirrors the real schema, missing-value patterns, cardinalities, and class imbalance (~3.5% fraud). The same code runs unchanged on the real CSV files.

| Section | Technique |
|---------|-----------|
| §3 | Missingness analysis & heuristic imputation |
| §4 | Frequency encoding & smoothed target (mean) encoding |
| §5 | Hand-crafted interaction features |
| §6 | Entity-level aggregation (group statistics) |
| §7 | Temporal feature extraction + cyclical sin/cos encoding |
| §8 | PCA compression of correlated V-feature blocks |
| §9 | Feature selection: variance, correlation pruning, permutation importance |
| §10 | Baseline vs. engineered-features model lift |

Backed by the reusable `feature_engineering/` package:

| Module | Responsibility |
|--------|----------------|
| `preprocessing.py` | Missing-value imputation, sentinel encoding, indicator flags |
| `encoders.py` | Frequency encoding, smoothed leave-one-out target encoding |
| `aggregations.py` | Entity-level group statistics (mean, std, min, max, count) |
| `temporal.py` | Temporal feature extraction and cyclical sin/cos encoding |
| `selection.py` | Variance filtering, correlation pruning, permutation importance |
| `utils.py` | Shared helpers (synthetic data loader, plotting utilities) |

## Running the Fraud Pipeline

Run with defaults (downloads data automatically if absent):

```bash
python3 run_fraud_pipeline.py
```

Override any parameter:

```bash
python3 run_fraud_pipeline.py \
    --data-path data/online_payment_fraud_data.csv \
    --test-size 0.2 \
    --random-state 42 \
    --threshold 0.5 \
    --best-model-idx 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-path` | `data/online_payment_fraud_data.csv` | Path to the fraud CSV |
| `--test-size` | `0.30` | Fraction of data reserved for testing |
| `--random-state` | `42` | RNG seed for reproducibility |
| `--threshold` | `0.5` | Decision threshold for fraud classification |
| `--best-model-idx` | `1` | Index of model whose confusion matrix is displayed (0=LR, 1=XGB, 2=RF) |

## Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Linting

```bash
# Notebooks
python3 -m nbqa flake8 autoencoder_hands_on.ipynb --max-line-length=120 --ignore=E501,W503,E402
python3 -m nbqa flake8 ieee_cis_feature_engineering.ipynb --max-line-length=120 --ignore=E501,W503,E402

# Python packages and scripts
python3 -m flake8 feature_engineering/ ml_pipeline/ run_fraud_pipeline.py \
    --max-line-length=120 --ignore=E501,W503,E402
```

## Testing

Run notebooks as tests using nbmake:

```bash
python3 -m pytest --nbmake autoencoder_hands_on.ipynb -v
python3 -m pytest --nbmake online_payment_fraud_detection.ipynb -v
python3 -m pytest --nbmake ieee_cis_feature_engineering.ipynb -v
```
