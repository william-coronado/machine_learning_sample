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


def _get_scores(model, X):
    """
    Return ``(scores, uses_proba)`` for ``X``, preferring ``predict_proba``.

    ``uses_proba`` is True when scores are calibrated probabilities in [0, 1]
    (threshold 0.5 applies) and False when they are raw decision-function
    outputs (threshold 0.0 applies).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], True
    if hasattr(model, "decision_function"):
        return model.decision_function(X), False
    raise AttributeError(
        f"{type(model).__name__!r} implements neither predict_proba nor "
        "decision_function; cannot compute ROC-AUC scores."
    )


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
    models  : List of fitted sklearn-compatible classifiers that expose
              ``predict_proba`` or ``decision_function``.
    X_train, y_train : Training data (for train AUC).
    X_test, y_test   : Test data (for validation AUC).

    Returns
    -------
    List of dicts with keys: ``model``, ``name``, ``train_auc``, ``test_auc``.
    """
    results = []
    for model in models:
        name = type(model).__name__
        train_proba, _ = _get_scores(model, X_train)
        test_proba, _ = _get_scores(model, X_test)

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
    threshold: float = 0.5,
) -> None:
    """
    Display a confusion matrix for a single fitted classifier.

    Parameters
    ----------
    model     : Fitted sklearn-compatible classifier.
    X_test    : Test feature matrix.
    y_test    : True test labels.
    cmap      : Matplotlib colormap name.
    threshold : Decision threshold applied to predicted probabilities or
                decision scores when deriving class labels. Ignored if the
                estimator does not expose probabilistic outputs or a
                decision function.
    """
    # Derive predicted labels at the specified threshold, if possible.
    try:
        scores, uses_proba = _get_scores(model, X_test)
        # probability scores use the caller-supplied threshold (default 0.5);
        # raw decision_function outputs use 0.0 as the natural boundary.
        effective_threshold = threshold if uses_proba else 0.0
        y_pred = (scores >= effective_threshold).astype(int)
    except AttributeError:
        # Fallback: use the model's own class predictions (threshold assumed
        # to be handled internally by the estimator).
        y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=cmap)
    plt.title(f"Confusion Matrix — {type(model).__name__}")
    plt.tight_layout()
    plt.show()
