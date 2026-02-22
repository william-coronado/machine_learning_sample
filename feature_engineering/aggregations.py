"""
aggregations.py
---------------
Generate group-level statistical features (a.k.a. aggregation or entity features).

These features encode the *behaviour* of an entity (card, email domain, device)
across transactions — a powerful signal for fraud detection because fraudsters
often reuse the same card on many transactions in a short window.

Class
-----
GroupAggregator : Configurable multi-group, multi-agg feature factory.

Example
-------
>>> agg = GroupAggregator(
...     group_cols=["card1", "card4"],
...     agg_cols=["TransactionAmt"],
...     agg_funcs=["mean", "std", "max", "count"],
... )
>>> X_train = agg.fit_transform(X_train)
>>> X_test  = agg.transform(X_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable


_AGG_NAMES = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "median": np.nanmedian,
    "max": np.nanmax,
    "min": np.nanmin,
    "count": len,
    "nunique": lambda x: len(set(x)),
}


def _agg_to_str(func) -> str:
    if callable(func) and hasattr(func, "__name__"):
        return func.__name__
    return str(func)


class GroupAggregator(BaseEstimator, TransformerMixin):
    """
    For each (group_cols, agg_col, agg_func) combination, compute a
    group-level statistic and merge it back as a new feature column.

    Parameters
    ----------
    groups : list[dict]
        Each dict specifies one aggregation recipe::

            {
                "group_cols": ["card1"],                # groupby keys
                "agg_cols":   ["TransactionAmt"],       # columns to aggregate
                "agg_funcs":  ["mean", "std", "max"],   # statistics
            }

        ``agg_funcs`` accepts strings ("mean", "std", "median", "max",
        "min", "count", "nunique") or arbitrary callables.

    suffix_sep : str
        Separator used to build the new column name, e.g.:
        ``card1_TransactionAmt_mean``.

    Example
    -------
    >>> groups = [
    ...     {"group_cols": ["card1"],
    ...      "agg_cols": ["TransactionAmt"], "agg_funcs": ["mean", "std"]},
    ...     {"group_cols": ["card1", "card4"],
    ...      "agg_cols": ["TransactionAmt"], "agg_funcs": ["count", "mean"]},
    ...     {"group_cols": ["P_emaildomain"],
    ...      "agg_cols": ["TransactionAmt", "C1"], "agg_funcs": ["mean", "max"]},
    ... ]
    >>> agg = GroupAggregator(groups=groups)
    >>> X_train = agg.fit_transform(X_train)
    """

    def __init__(
        self,
        groups: list[dict] | None = None,
        suffix_sep: str = "_",
    ):
        # Default recipe mirrors common IEEE-CIS baseline approaches
        self.groups = groups or [
            {
                "group_cols": ["card1"],
                "agg_cols": ["TransactionAmt"],
                "agg_funcs": ["mean", "std", "max", "count"],
            },
            {
                "group_cols": ["card1", "card4"],
                "agg_cols": ["TransactionAmt"],
                "agg_funcs": ["mean", "count"],
            },
            {
                "group_cols": ["addr1", "card1"],
                "agg_cols": ["TransactionAmt"],
                "agg_funcs": ["mean", "std"],
            },
            {
                "group_cols": ["P_emaildomain"],
                "agg_cols": ["TransactionAmt"],
                "agg_funcs": ["mean", "std", "max"],
            },
            {
                "group_cols": ["card1"],
                "agg_cols": ["D1", "D2"],
                "agg_funcs": ["mean", "std"],
            },
        ]
        self.suffix_sep = suffix_sep
        self._lookup_tables: list[tuple[list[str], str, str, pd.Series]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_func(func) -> tuple[Callable, str]:
        if isinstance(func, str):
            if func not in _AGG_NAMES:
                raise ValueError(
                    f"Unknown agg function '{func}'. "
                    f"Choose from {list(_AGG_NAMES)} or pass a callable."
                )
            return _AGG_NAMES[func], func
        return func, _agg_to_str(func)

    def _col_name(self, group_cols: list[str], agg_col: str, func_name: str) -> str:
        return self.suffix_sep.join(group_cols + [agg_col, func_name])

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "GroupAggregator":
        """Compute and store all group statistics from X."""
        self._lookup_tables = []

        for recipe in self.groups:
            group_cols: list[str] = recipe["group_cols"]
            agg_cols: list[str] = recipe["agg_cols"]
            agg_funcs = recipe["agg_funcs"]

            # Skip if any column missing
            required = group_cols + agg_cols
            missing = [c for c in required if c not in X.columns]
            if missing:
                continue

            grouped = X.groupby(group_cols, observed=True)

            for agg_col in agg_cols:
                for raw_func in agg_funcs:
                    func, func_name = self._resolve_func(raw_func)
                    new_col = self._col_name(group_cols, agg_col, func_name)

                    series = grouped[agg_col].agg(func).rename(new_col)
                    self._lookup_tables.append(
                        (group_cols, agg_col, new_col, series)
                    )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Merge stored group statistics back into X."""
        X = X.copy()

        for group_cols, agg_col, new_col, series in self._lookup_tables:
            lookup_df = series.reset_index()
            X = X.merge(lookup_df, on=group_cols, how="left")

            # Cast to float32 to save memory
            if new_col in X.columns:
                X[new_col] = X[new_col].astype(np.float32)

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:  # type: ignore[override]
        return self.fit(X).transform(X)

    @property
    def generated_feature_names(self) -> list[str]:
        """Names of all features that will be (or have been) added."""
        names = []
        for recipe in self.groups:
            group_cols = recipe["group_cols"]
            for agg_col in recipe["agg_cols"]:
                for raw_func in recipe["agg_funcs"]:
                    _, func_name = self._resolve_func(raw_func)
                    names.append(self._col_name(group_cols, agg_col, func_name))
        return names
