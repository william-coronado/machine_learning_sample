# Machine Learning Sample — Claude Guide

## Project Overview

This project contains Jupyter notebooks demonstrating machine learning techniques:

- `online_payment_fraud_detection.ipynb` — Binary classification for fraud detection using scikit-learn and XGBoost
- `autoencoder_hands_on.ipynb` — Fully-connected and denoising autoencoders in PyTorch on MNIST

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

Run flake8 on a notebook via nbqa:

```bash
python3 -m nbqa flake8 autoencoder_hands_on.ipynb --max-line-length=120 --ignore=E501,W503,E402
```

## Testing

Run notebooks as tests using nbmake:

```bash
python3 -m pytest --nbmake autoencoder_hands_on.ipynb -v
python3 -m pytest --nbmake online_payment_fraud_detection.ipynb -v
```
