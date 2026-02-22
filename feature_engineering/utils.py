"""
utils.py
--------
Shared visualisation and evaluation helpers used throughout the notebook.

Functions
---------
plot_missing(df, figsize, max_cols)
    Horizontal bar chart of missingness rates.

plot_feature_importance(importance, top_n, title, figsize)
    Horizontal bar chart of feature importances.

roc_comparison(models, X_test, y_test, labels, figsize)
    Overlaid ROC curves for multiple fitted classifiers.

plot_pca_explained_variance(pca, figsize)
    Scree plot of cumulative explained variance.

evaluate_model(model, X_train, y_train, X_test, y_test, name)
    Print and return a dict of AUC, AP, and classification report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    classification_report,
    RocCurveDisplay,
)
from sklearn.decomposition import PCA


sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Missing-value chart
# ---------------------------------------------------------------------------

def plot_missing(
    df: pd.DataFrame,
    figsize: tuple = (10, 6),
    max_cols: int = 40,
    title: str = "Missing values per column",
) -> None:
    """Bar chart of missing-value percentages (top ``max_cols`` only)."""
    miss = df.isnull().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0].head(max_cols)

    if miss.empty:
        print("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    miss.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Missing (%)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    for bar in ax.patches:
        width = bar.get_width()
        if width > 0:
            ax.text(
                width + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%", va="center", fontsize=8,
            )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Feature importance chart
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 30,
    title: str = "Feature Importances",
    figsize: tuple = (10, 8),
) -> None:
    """
    Horizontal bar chart of a feature importance Series.

    Parameters
    ----------
    importance : pd.Series  index = feature names, values = importances.
    top_n      : Number of top features to display.
    """
    top = importance.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    top.plot.barh(ax=ax, color="teal", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ROC comparison
# ---------------------------------------------------------------------------

def roc_comparison(
    models: list,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str] | None = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot overlaid ROC curves for a list of fitted classifiers.

    Parameters
    ----------
    models : list of fitted sklearn-compatible classifiers.
    labels : Display names aligned with ``models``.
    """
    labels = labels or [type(m).__name__ for m in models]
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for m, label, color in zip(models, labels, colors):
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# PCA scree plot
# ---------------------------------------------------------------------------

def plot_pca_explained_variance(
    pca: PCA,
    figsize: tuple = (9, 4),
    title: str = "PCA Explained Variance",
) -> None:
    """Scree + cumulative explained variance plot."""
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].bar(range(1, len(ratios) + 1), ratios * 100, color="steelblue")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Per-component")

    axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100,
                 marker="o", ms=4, color="darkorange")
    axes[1].axhline(95, ls="--", color="grey", label="95 %")
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative")
    axes[1].legend()

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Model evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Model",
    threshold: float = 0.5,
) -> dict:
    """
    Return and print AUC-ROC, Average Precision, and classification report.

    Parameters
    ----------
    model     : Fitted sklearn-compatible classifier.
    threshold : Decision threshold for the classification report.

    Returns
    -------
    dict with keys: name, train_auc, test_auc, test_ap.
    """
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    train_auc = auc(*roc_curve(y_train, train_proba)[:2])
    test_auc = auc(*roc_curve(y_test, test_proba)[:2])
    test_ap = average_precision_score(y_test, test_proba)

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Train AUC-ROC : {train_auc:.4f}")
    print(f"  Test  AUC-ROC : {test_auc:.4f}")
    print(f"  Test  Avg Prec: {test_ap:.4f}")
    print(f"\n{classification_report(y_test, test_pred, digits=4)}")

    return {"name": name, "train_auc": train_auc, "test_auc": test_auc, "test_ap": test_ap}
