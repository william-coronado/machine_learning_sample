"""
selection.py
------------
Post-engineering feature selection utilities.

Functions
---------
drop_high_correlation(df, threshold, target, verbose)
    Remove one of each pair of features with |ρ| > threshold.

variance_threshold_select(df, threshold, exclude)
    Drop near-zero-variance features.

importance_select(X, y, model, top_n, plot)
    Select top-n features by permutation importance.

pca_compress(df, cols, n_components, prefix)
    Replace a set of correlated columns with their principal components
    (used for V-feature blocks).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Correlation-based pruning
# ---------------------------------------------------------------------------

def drop_high_correlation(
    df: pd.DataFrame,
    threshold: float = 0.95,
    target: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Greedily remove one column from each highly-correlated pair.

    When ``target`` is provided, of the two correlated features the one with
    *lower* absolute correlation to the target is dropped (greedy strategy).
    Otherwise the second column in lexicographic order is dropped.

    Parameters
    ----------
    df        : DataFrame of numeric features (+ optional target column).
    threshold : Pearson |ρ| above which one feature is dropped.
    target    : Name of the target column (kept, never dropped).
    verbose   : Print each dropped feature and the correlated pair.

    Returns
    -------
    DataFrame with correlated columns removed.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in num_cols:
        num_cols = [c for c in num_cols if c != target]

    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    if target and target in df.columns:
        target_corr = df[num_cols].corrwith(df[target]).abs()

    for col in upper.columns:
        if col in to_drop:
            continue
        correlated = upper.index[upper[col] > threshold].tolist()
        for partner in correlated:
            if partner in to_drop:
                continue
            if target and target in df.columns:
                # Drop the one less correlated with target
                victim = col if target_corr[col] < target_corr[partner] else partner
            else:
                victim = col  # drop the first of the pair
            to_drop.add(victim)
            if verbose:
                ρ = corr.loc[col, partner]
                print(f"  Dropping '{victim}' (|ρ| = {ρ:.3f} with '{partner}')")

    kept = [c for c in df.columns if c not in to_drop]
    if verbose:
        print(f"\nDropped {len(to_drop)} features. {len(kept)} remain.")
    return df[kept]


# ---------------------------------------------------------------------------
# Variance threshold
# ---------------------------------------------------------------------------

def variance_threshold_select(
    df: pd.DataFrame,
    threshold: float = 0.0,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """
    Drop numeric columns whose variance is ≤ ``threshold``.

    Parameters
    ----------
    df        : DataFrame (may contain non-numeric columns; they are kept).
    threshold : Features with variance ≤ this are removed.
    exclude   : Column names to skip (e.g., target).

    Returns
    -------
    DataFrame with low-variance columns removed.
    """
    exclude = set(exclude or [])
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    variances = df[num_cols].var()
    to_drop = variances[variances <= threshold].index.tolist()

    non_num = [c for c in df.columns if c not in num_cols]
    kept = non_num + [c for c in num_cols if c not in to_drop]

    print(f"Variance threshold: dropped {len(to_drop)} features.")
    return df[kept]


# ---------------------------------------------------------------------------
# Importance-based selection
# ---------------------------------------------------------------------------

def importance_select(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    top_n: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select ``top_n`` features by sklearn permutation importance.

    A lightweight RandomForest is used by default if no model is supplied.

    Parameters
    ----------
    X            : Feature DataFrame.
    y            : Target series.
    model        : Pre-fitted or unfitted sklearn-compatible estimator.
                   If unfitted, it will be fit on X, y.
    top_n        : Number of top features to keep.
    random_state : RNG seed for the permutation importance.

    Returns
    -------
    (X_selected, importance_series)
        X_selected        : DataFrame with top-n columns.
        importance_series : Feature importances sorted descending.
    """
    if model is None:
        model = RandomForestClassifier(
            n_estimators=100, max_depth=6,
            n_jobs=-1, random_state=random_state
        )

    # Fit if the model has not been trained yet
    try:
        model.predict(X.iloc[:1])
    except Exception:
        model.fit(X, y)

    result = permutation_importance(
        model, X, y, n_repeats=5,
        random_state=random_state, n_jobs=-1,
    )
    imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    top_features = imp.head(top_n).index.tolist()

    return X[top_features], imp


# ---------------------------------------------------------------------------
# PCA compression for V-feature blocks
# ---------------------------------------------------------------------------

def pca_compress(
    df: pd.DataFrame,
    cols: list[str],
    n_components: int | float = 0.95,
    prefix: str = "pca",
    fit_df: pd.DataFrame | None = None,
    pca: PCA | None = None,
) -> tuple[pd.DataFrame, PCA]:

    """
    Replace ``cols`` with their principal components, retaining
    ``n_components`` (int = exact, float = variance explained).

    V1-V339 in the IEEE-CIS dataset are grouped into correlated blocks.
    PCA per block is a powerful way to:
    - reduce dimensionality while retaining variance,
    - remove the block-wise missing-value structure,
    - produce decorrelated inputs for linear models.

    Parameters
    ----------
    df           : DataFrame to transform.
    cols         : Columns to compress (should be numeric, same missingness pattern).
    n_components : Components to keep (int) or variance to retain (float 0–1).
    prefix       : Prefix for output PCA columns.
    fit_df       : If provided, PCA is fitted on this DataFrame and applied
                   to ``df`` (useful for test-set transforms).

    Returns
    -------
    (df_out, pca_fitted)
        df_out     : Original DataFrame with ``cols`` replaced by PCA columns.
        pca_fitted : Fitted PCA object (for later transform on test data).
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        warnings.warn(f"None of {cols} found in DataFrame. Skipping PCA.")
        return df, PCA()

    fit_source = fit_df if fit_df is not None else df
    data_fit = fit_source[present].fillna(0)
    data_tr = df[present].fillna(0)

    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(data_fit)

    transformed = pca.transform(data_tr)
    n_out = transformed.shape[1]
    pca_cols = [f"{prefix}_c{i}" for i in range(n_out)]

    df = df.drop(columns=present)
    pca_df = pd.DataFrame(transformed, columns=pca_cols,
                          index=df.index, dtype=np.float32)
    df = pd.concat([df, pca_df], axis=1)

    var_explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"PCA: {len(present)} cols → {n_out} components "
          f"({var_explained * 100:.1f} % variance retained).")
    return df, pca
