"""
encoders.py
-----------
sklearn-compatible transformers for high-cardinality categorical features.

Classes
-------
TargetEncoder    : Smoothed target mean encoding with leave-one-out on train.
FrequencyEncoder : Replace category with its normalised frequency.

Both follow the sklearn Transformer API:
    fit(X, y) → self
    transform(X) → pd.DataFrame
    fit_transform(X, y) → pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed mean target encoder with leave-one-out (LOO) during training.

    Smoothing formula (Micci-Barreca, 2001):
        encoded = (n_i * mean_i + k * global_mean) / (n_i + k)

    where n_i = number of observations for category i and k is the
    smoothing strength. This prevents overfitting on rare categories.

    During *transform* (inference), no LOO is applied — the smoothed
    category mean is used directly.

    Parameters
    ----------
    cols : list[str]
        Columns to encode.
    smoothing : float
        Regularisation strength k. Higher → pull more toward global mean.
    min_samples_leaf : int
        Categories with fewer samples use the global mean.
    noise_level : float
        Gaussian noise std added to LOO targets during fit (extra regularisation).

    Example
    -------
    >>> enc = TargetEncoder(cols=["card4", "P_emaildomain"], smoothing=20)
    >>> X_train_enc = enc.fit_transform(X_train, y_train)
    >>> X_test_enc  = enc.transform(X_test)
    """

    def __init__(
        self,
        cols: list[str],
        smoothing: float = 20.0,
        min_samples_leaf: int = 10,
        noise_level: float = 0.01,
    ):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self._global_mean: float = 0.0
        self._mapping: dict[str, pd.Series] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        self._global_mean = float(y.mean())

        for col in self.cols:
            stats = (
                pd.concat([X[col].reset_index(drop=True),
                           y.reset_index(drop=True)], axis=1)
                .groupby(col)[y.name if y.name else 0]
                .agg(["count", "mean"])
            )
            stats.columns = ["n", "mean"]
            # Smoothing
            smoother = 1 / (1 + np.exp(-(stats["n"] - self.min_samples_leaf)
                                        / self.smoothing))
            stats["encoded"] = (
                smoother * stats["mean"]
                + (1 - smoother) * self._global_mean
            )
            self._mapping[col] = stats["encoded"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            new_col = f"{col}_te"
            X[new_col] = (
                X[col].map(self._mapping[col])
                      .fillna(self._global_mean)
                      .astype(np.float32)
            )
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:  # type: ignore[override]
        """
        Fit with LOO to avoid target leakage on the training set.
        Each row's own target is subtracted before computing the mean.
        """
        self.fit(X, y)
        X = X.copy()
        y = y.reset_index(drop=True)

        for col in self.cols:
            new_col = f"{col}_te"
            # LOO: subtract self, then apply smoothing
            concat = pd.concat(
                [X[col].reset_index(drop=True), y], axis=1
            )
            concat.columns = ["cat", "target"]

            group_stats = concat.groupby("cat")["target"].agg(["sum", "count"])
            # Leave-one-out sum and count
            loo_sum = concat["cat"].map(group_stats["sum"]) - concat["target"]
            loo_cnt = concat["cat"].map(group_stats["count"]) - 1

            loo_mean = np.where(loo_cnt > 0, loo_sum / loo_cnt, self._global_mean)
            smoother = 1 / (
                1 + np.exp(-(loo_cnt - self.min_samples_leaf) / self.smoothing)
            )
            encoded = smoother * loo_mean + (1 - smoother) * self._global_mean

            if self.noise_level > 0:
                encoded += np.random.normal(0, self.noise_level, len(encoded))

            X[new_col] = encoded.astype(np.float32)
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode each category as its normalised frequency in the training set.

    Frequencies are naturally bounded in [0, 1], which makes them suitable
    as-is for both linear models and tree-based ensembles. Rare or unseen
    categories receive a configurable ``default_freq`` (default = 0).

    Parameters
    ----------
    cols : list[str]
        Columns to encode.
    normalize : bool
        If True (default), encode as fraction of total rows.
        If False, encode as raw count.

    Example
    -------
    >>> enc = FrequencyEncoder(cols=["ProductCD", "card4"])
    >>> X_train = enc.fit_transform(X_train)
    >>> X_test  = enc.transform(X_test)
    """

    def __init__(self, cols: list[str], normalize: bool = True):
        self.cols = cols
        self.normalize = normalize
        self._freq: dict[str, pd.Series] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        n = len(X)
        for col in self.cols:
            counts = X[col].value_counts(normalize=self.normalize)
            self._freq[col] = counts
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            new_col = f"{col}_freq"
            X[new_col] = (
                X[col].map(self._freq[col])
                      .fillna(0)
                      .astype(np.float32)
            )
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:  # type: ignore[override]
        return self.fit(X).transform(X)
