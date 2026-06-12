"""
fraud_detection.py
------------------
Telco online-payment fraud detection — use-case-specific configuration,
feature engineering, and model selection.

This module isolates every decision that is specific to the dataset described
in ``online_payment_fraud_detection.ipynb``:

  - Which columns identify the sender / receiver (high-cardinality IDs to drop).
  - How to encode the ``type`` column (one-hot via get_dummies).
  - Which models and hyperparameters work well for this class-imbalanced task.
  - The decision threshold tuned for the business cost of missed fraud.

Public API
----------
TELCO_COLS_TO_DROP   : list[str]  — columns removed before training.
TARGET_COL           : str        — prediction target ("isFraud").
FRAUD_THRESHOLD      : float      — decision threshold for fraud classification.
DATA_URL             : str        — Google Drive download URL for the dataset.

engineer_telco_features(df)  → pd.DataFrame
get_fraud_models()           → list of sklearn-compatible estimators
load_telco_fraud_data(filepath, url, quiet) → pd.DataFrame

suggest_lr_params(trial)  → dict  — Optuna search space for LogisticRegression.
suggest_xgb_params(trial) → dict  — Optuna search space for XGBClassifier.
suggest_rf_params(trial)  → dict  — Optuna search space for RandomForestClassifier.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd
import gdown
from ml_pipeline.core.preprocessing import encode_categoricals, drop_columns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

if TYPE_CHECKING:
    import optuna

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Target column name in the raw dataset.
TARGET_COL: str = "isFraud"

#: High-cardinality ID columns and the raw categorical column that gets
#: replaced by one-hot dummies; all removed before model training.
TELCO_COLS_TO_DROP: list[str] = ["type", "nameOrig", "nameDest", "isFlaggedFraud"]

#: Categorical columns to one-hot encode during feature engineering.
CATEGORICAL_COLS: list[str] = ["type"]

#: Decision threshold for converting predicted probabilities into fraud labels.
#: Lowering this value increases recall at the cost of more false positives.
FRAUD_THRESHOLD: float = 0.5

#: Google Drive download URL for the telco fraud CSV.
DATA_URL: str = "https://drive.google.com/uc?id=127JqP3WGjBVihR-ZcUR86T3wwy3_g63v"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_telco_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply telco-fraud-specific feature engineering to the raw DataFrame.

    Steps
    -----
    1. One-hot encode the ``type`` column (PAYMENT, TRANSFER, CASH_OUT, …).
       ``drop_first=True`` avoids perfect multicollinearity.
    2. Drop high-cardinality account-ID columns (``nameOrig``, ``nameDest``)
       and the leaky ``isFlaggedFraud`` flag.

    The original ``type`` column is dropped as part of step 2 so callers
    receive a fully numeric DataFrame ready for scikit-learn estimators.

    Parameters
    ----------
    df : Raw transaction DataFrame as loaded from the CSV.

    Returns
    -------
    pd.DataFrame with engineered features; no string columns remain.
    """
    # 1. One-hot encode transaction type
    df = encode_categoricals(df, CATEGORICAL_COLS, drop_first=True)

    # 2. Remove columns that are either identifiers, leakage, or now redundant
    df = drop_columns(df, TELCO_COLS_TO_DROP)

    return df


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def get_fraud_models() -> list:
    """
    Return a list of estimators pre-configured for telco fraud detection.

    Model choices and hyperparameter rationale
    ------------------------------------------
    LogisticRegression
        Provides a fast linear baseline; ``max_iter=1000`` ensures convergence
        on the large dataset (~6 M rows).

    XGBClassifier
        Best performer in the notebook.  Hyperparameters are tuned for fraud:
        - ``n_estimators=200`` : more boosting rounds for complex patterns.
        - ``max_depth=6``      : captures non-linear interactions.
        - ``learning_rate=0.1``: stable convergence.
        - ``subsample=0.8`` / ``colsample_bytree=0.8``: prevent overfitting.
        - ``eval_metric='logloss'``: appropriate for binary classification.

    RandomForestClassifier
        Ensemble baseline; kept lightweight (``n_estimators=7``) for speed
        on the large dataset while still providing a diversity reference point.

    Returns
    -------
    list of un-fitted sklearn-compatible estimators.
    """
    return [
        LogisticRegression(max_iter=1000),
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        ),
        RandomForestClassifier(
            n_estimators=7,
            criterion="entropy",
            random_state=7,
        ),
    ]


# ---------------------------------------------------------------------------
# Hyperparameter search spaces (Optuna)
# ---------------------------------------------------------------------------

def suggest_lr_params(trial: "optuna.Trial") -> dict:
    """
    Suggest hyperparameters for ``LogisticRegression``.

    Parameters
    ----------
    trial : An Optuna trial object.

    Returns
    -------
    dict of constructor kwargs suitable for ``LogisticRegression(**params)``.
    """
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "max_iter": 1000,
    }


def suggest_xgb_params(trial: "optuna.Trial") -> dict:
    """
    Suggest hyperparameters for ``XGBClassifier``.

    Parameters
    ----------
    trial : An Optuna trial object.

    Returns
    -------
    dict of constructor kwargs suitable for ``XGBClassifier(**params)``.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "eval_metric": "logloss",
        "random_state": 42,
    }


def suggest_rf_params(trial: "optuna.Trial") -> dict:
    """
    Suggest hyperparameters for ``RandomForestClassifier``.

    Parameters
    ----------
    trial : An Optuna trial object.

    Returns
    -------
    dict of constructor kwargs suitable for ``RandomForestClassifier(**params)``.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": 42,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_telco_fraud_data(
    filepath: str = "data/online_payment_fraud_data.csv",
    url: str = DATA_URL,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Load the telco online-payment fraud CSV, downloading it if necessary.

    The file is downloaded from Google Drive only when it does not already
    exist at ``filepath``.  The parent directory is created automatically.

    Parameters
    ----------
    filepath : Local path where the CSV is stored / should be saved.
    url      : Google Drive download URL.
    quiet    : If True, suppress gdown progress output.

    Returns
    -------
    pd.DataFrame with the raw transaction data.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if not os.path.exists(filepath):
        print(f"Downloading dataset to {filepath} …")
        downloaded_path = gdown.download(url, filepath, quiet=quiet)
        # Validate that the download succeeded and produced a usable file.
        if not downloaded_path or not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            raise RuntimeError(
                f"Failed to download telco fraud dataset from {url!r} to {filepath!r}. "
                "Please check your network connection and the URL, then try again."
            )
    else:
        print(f"{filepath} already exists. Skipping download.")
        if os.path.getsize(filepath) == 0:
            raise RuntimeError(
                f"Existing file at {filepath!r} is empty (possibly from a failed download). "
                "Delete it and try again."
            )

    return pd.read_csv(filepath)
