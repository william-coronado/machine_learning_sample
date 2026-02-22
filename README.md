# Machine Learning Sample Project

## Project Overview

This project demonstrates various machine learning techniques and algorithms through practical implementations. It serves as a comprehensive sample repository showcasing different aspects of the machine learning workflow, from feature engineering to model evaluation.

## Project Structure

```
machine_learning_sample/
├── README.md
├── online_payment_fraud_detection.ipynb
├── autoencoder_hands_on.ipynb
├── ieee_cis_feature_engineering.ipynb
├── feature_engineering/
│   ├── __init__.py
│   ├── aggregations.py
│   ├── encoders.py
│   ├── preprocessing.py
│   ├── selection.py
│   ├── temporal.py
│   └── utils.py
└── data/
    └── online_payment_fraud_data.csv
```

## Featured Notebooks

### 1. Online Payment Fraud Detection (`online_payment_fraud_detection.ipynb`)

This notebook demonstrates a complete machine learning pipeline for detecting fraudulent online payment transactions. It's an excellent example of binary classification applied to a real-world problem in financial technology.


### 2. autoencoders using PyTorch (`autoencoder_hands_on.ipynb`)

This notebook provides a hands-on guide to understanding and implementing autoencoders using PyTorch. It includes the following sections:

1. Loading and preprocessing the MNIST dataset.

2. Designing and training a simple fully-connected autoencoder to reconstruct images.

3. Visualizing image reconstructions and exploring the latent space representations.

4. Extending the model to a denoising autoencoder, aimed at reconstructing clean images from noisy inputs.

5. Suggestions for further experimentation, such as using convolutional layers, adding sparsity penalties, or exploring variational autoencoders (VAEs).

### 3. IEEE-CIS Feature Engineering (`ieee_cis_feature_engineering.ipynb`)

A senior data-science reference guide to feature engineering for tabular fraud detection, using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset schema. Because the dataset requires Kaggle authentication, the notebook uses a synthetic replica that mirrors the real dataset's schema, missing-value patterns, cardinalities, and class imbalance (~3.5% fraud). The same code runs unchanged on the real CSV files.

Techniques covered:

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

#### Supporting module: `feature_engineering/`

The notebook is backed by a reusable Python package under `feature_engineering/`:

| Module | Responsibility |
|--------|---------------|
| `preprocessing.py` | Missing-value imputation, sentinel encoding, indicator flags |
| `encoders.py` | Frequency encoding, smoothed leave-one-out target encoding |
| `aggregations.py` | Entity-level group statistics (mean, std, min, max, count) |
| `temporal.py` | Temporal feature extraction and cyclical sin/cos encoding |
| `selection.py` | Variance filtering, correlation pruning, permutation importance |
| `utils.py` | Shared helpers (synthetic data loader, plotting utilities) |
