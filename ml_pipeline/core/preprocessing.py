"""
preprocessing.py
----------------
Generic, domain-agnostic data preprocessing utilities.

These helpers are intentionally free of any dataset-specific knowledge so they
can be reused across different ML pipelines.

Public API
----------
encode_categoricals(df, cols, drop_first)   → pd.DataFrame
drop_columns(df, cols)                       → pd.DataFrame
split_features_target(df, target_col, extra_drop_cols) → (pd.DataFrame, pd.Series)
train_test_split_data(X, y, test_size, random_state)   → (X_train, X_test, y_train, y_test)
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def encode_categoricals(
    df: pd.DataFrame,
    cols: list[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode the specified categorical columns and append the dummies.

    The original columns are **not** dropped automatically — callers should
    include them in ``drop_columns`` if they are no longer needed.

    Parameters
    ----------
    df         : Input DataFrame.
    cols       : Columns to encode.
    drop_first : If True, drop the first dummy to avoid multicollinearity.

    Returns
    -------
    pd.DataFrame with dummy columns appended.
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    dummies = pd.get_dummies(df[present], drop_first=drop_first)
    return pd.concat([df, dummies], axis=1)


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Drop a list of columns from a DataFrame, ignoring any that are absent.

    Parameters
    ----------
    df   : Input DataFrame.
    cols : Column names to remove.

    Returns
    -------
    pd.DataFrame without the specified columns.
    """
    to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=to_drop)


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    extra_drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate a DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df              : DataFrame containing both features and target.
    target_col      : Name of the target column.
    extra_drop_cols : Additional columns to exclude from X (e.g., ID columns).

    Returns
    -------
    (X, y) : Feature DataFrame and target Series.
    """
    drop = [target_col] + (extra_drop_cols or [])
    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df[target_col]
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split with optional stratification.

    Parameters
    ----------
    X            : Feature matrix.
    y            : Target vector.
    test_size    : Fraction of data reserved for the test set.
    random_state : RNG seed for reproducibility.
    stratify     : If True, preserve the class distribution of ``y`` in both
                   splits. Recommended for imbalanced classification tasks;
                   must be False for regression targets.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
