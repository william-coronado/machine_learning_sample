"""
evaluation.py
-------------
Generic model evaluation utilities for binary classification pipelines.

Both functions are agnostic to the specific use case — they accept any
fitted sklearn-compatible classifiers and standard X/y inputs.

Public API
----------
evaluate_models(models, X_train, y_train, X_test, y_test) → list[dict]
plot_confusion_matrix(model, X_test, y_test)               → None
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay


def evaluate_models(
    models: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    """
    Compute and print ROC-AUC on the training and test sets for each model.

    Parameters
    ----------
    models  : List of fitted sklearn-compatible classifiers.
    X_train, y_train : Training data (for train AUC).
    X_test, y_test   : Test data (for validation AUC).

    Returns
    -------
    List of dicts with keys: ``model``, ``name``, ``train_auc``, ``test_auc``.
    """
    results = []
    for model in models:
        name = type(model).__name__
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)

        print(f"{name}:")
        print(f"  Train AUC : {train_auc:.4f}")
        print(f"  Test  AUC : {test_auc:.4f}")
        print()

        results.append({
            "model": model,
            "name": name,
            "train_auc": train_auc,
            "test_auc": test_auc,
        })
    return results


def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cmap: str = "Blues",
) -> None:
    """
    Display a confusion matrix for a single fitted classifier.

    Parameters
    ----------
    model  : Fitted sklearn-compatible classifier.
    X_test : Test feature matrix.
    y_test : True test labels.
    cmap   : Matplotlib colormap name.
    """
    cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    cm.plot(cmap=cmap)
    plt.title(f"Confusion Matrix — {type(model).__name__}")
    plt.tight_layout()
    plt.show()
