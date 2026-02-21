"""
preprocessing.py
----------------
Utilities for loading, memory-optimising, and imputing IEEE-CIS-style data.

Public API
----------
reduce_mem_usage(df)          -> pd.DataFrame
summarise_missing(df)         -> pd.DataFrame
impute_by_type(df, cat_fill, num_strategy) -> pd.DataFrame
load_synthetic_ieee(n_rows, seed) -> (pd.DataFrame, pd.DataFrame)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


# ---------------------------------------------------------------------------
# Memory optimisation
# ---------------------------------------------------------------------------

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest dtype that preserves all values.

    Typical savings: 50-70 % on IEEE-CIS-sized frames.

    Parameters
    ----------
    df : pd.DataFrame
    verbose : bool
        Print before/after memory report.

    Returns
    -------
    pd.DataFrame
        Same data, smaller footprint (in-place modification, copy returned).
    """
    df = df.copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        # Skip object, category, and pandas extension types (StringDtype, etc.)
        if col_type == object or str(col_type) == "category":
            continue
        try:
            # Extension dtypes (e.g. StringDtype) are not numpy dtypes
            np.dtype(col_type)
        except TypeError:
            continue

        c_min, c_max = df[col].min(), df[col].max()

        if np.issubdtype(col_type, np.integer):
            for dtype in (np.int8, np.int16, np.int32, np.int64):
                if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        saved = 100 * (1 - end_mem / start_mem)
        print(f"Memory: {start_mem:.1f} MB → {end_mem:.1f} MB  ({saved:.1f} % reduction)")
    return df


# ---------------------------------------------------------------------------
# Missing-value diagnostics
# ---------------------------------------------------------------------------

def summarise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a sorted DataFrame describing missingness per column.

    Columns in the output
    ---------------------
    n_missing, pct_missing, dtype, n_unique, example_values
    """
    n_missing = df.isnull().sum()
    pct_missing = n_missing / len(df) * 100

    summary = pd.DataFrame({
        "n_missing": n_missing,
        "pct_missing": pct_missing.round(2),
        "dtype": df.dtypes,
        "n_unique": df.nunique(),
    })

    # Add a non-null example per column for quick inspection
    examples = {
        col: df[col].dropna().iloc[:3].tolist() if df[col].notna().any() else []
        for col in df.columns
    }
    summary["example_values"] = pd.Series(examples)

    return summary.sort_values("pct_missing", ascending=False)


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_by_type(
    df: pd.DataFrame,
    cat_fill: str = "Unknown",
    num_strategy: Literal["median", "mean", "minus999"] = "median",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply heuristic imputation rules consistent with tree-based model needs:

    - Categorical / object columns → constant ``cat_fill`` (default "Unknown").
    - Boolean / M-flag columns     → constant "Unknown" (preserves 3-way info).
    - Numeric columns              → median, mean, or -999 sentinel.

    The -999 sentinel is useful for gradient-boosted trees: it lets the model
    learn that missingness itself is a signal.

    Parameters
    ----------
    df           : Input DataFrame.
    cat_fill     : Fill value for categorical columns.
    num_strategy : One of {"median", "mean", "minus999"}.
    inplace      : Modify in-place; otherwise return a copy.
    """
    if not inplace:
        df = df.copy()

    m_cols = [c for c in df.columns if c.startswith("M") and df[c].dtype == object]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Categorical / M-flags
    for col in cat_cols:
        df[col] = df[col].fillna(cat_fill)

    # Numeric
    for col in num_cols:
        if df[col].isnull().any():
            if num_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif num_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            else:  # minus999
                df[col] = df[col].fillna(-999)

    return df


# ---------------------------------------------------------------------------
# Synthetic IEEE-CIS data generator
# ---------------------------------------------------------------------------

def _make_v_blocks(rng: np.random.Generator, n: int) -> dict:
    """
    Generate V1–V339 Vesta-style features with realistic block-missingness.
    Columns within a block share the same missing-value pattern.
    """
    V_BLOCKS = [
        (range(1, 12), 0.05),    # block 1  – mostly observed
        (range(12, 35), 0.45),   # block 2  – moderate missingness
        (range(35, 53), 0.15),
        (range(53, 75), 0.55),
        (range(75, 95), 0.10),
        (range(95, 138), 0.50),
        (range(138, 167), 0.20),
        (range(167, 217), 0.60),
        (range(217, 279), 0.30),
        (range(279, 340), 0.70),  # block 10 – mostly missing
    ]
    result = {}
    for block_range, miss_rate in V_BLOCKS:
        # One shared missingness mask per block
        mask = rng.random(n) < miss_rate
        for v in block_range:
            values = rng.standard_normal(n).astype(np.float32)
            values[mask] = np.nan
            result[f"V{v}"] = values
    return result


def load_synthetic_ieee(
    n_rows: int = 100_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic transaction and identity tables that mirror the
    schema, missing-value patterns, and class imbalance of the real
    IEEE-CIS Fraud Detection dataset.

    Returns
    -------
    train_transaction : pd.DataFrame  (n_rows rows)
    train_identity    : pd.DataFrame  (~72 % of transactions have identity data)

    Usage
    -----
    >>> from feature_engineering.preprocessing import load_synthetic_ieee
    >>> trans, identity = load_synthetic_ieee(n_rows=50_000)
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # ------------------------------------------------------------------
    # Transaction table
    # ------------------------------------------------------------------
    fraud_mask = rng.random(n) < 0.035          # ~3.5 % fraud rate

    trans = pd.DataFrame()
    trans["TransactionID"] = np.arange(2_987_000, 2_987_000 + n)
    trans["isFraud"] = fraud_mask.astype(np.int8)

    # Time: spans ~6 months of seconds
    trans["TransactionDT"] = rng.integers(86_400, 15_811_200, n).astype(np.int32)

    # Amount: log-normal with higher amounts for fraud
    base_amt = rng.lognormal(4.0, 1.5, n)
    fraud_boost = np.where(fraud_mask, rng.lognormal(1.5, 0.8, n), 1.0)
    trans["TransactionAmt"] = (base_amt * fraud_boost).round(2).astype(np.float32)

    # Product code
    trans["ProductCD"] = rng.choice(["W", "H", "C", "S", "R"], n,
                                    p=[0.60, 0.17, 0.10, 0.08, 0.05])

    # Card features
    trans["card1"] = rng.integers(1000, 18_396, n).astype(np.int16)
    trans["card2"] = rng.choice(
        [np.nan, 111.0, 222.0, 321.0, 400.0, 512.0],
        n, p=[0.08, 0.25, 0.20, 0.20, 0.18, 0.09]
    ).astype("float32")
    trans["card3"] = rng.choice([150.0, 185.0, 144.0, np.nan], n,
                                p=[0.60, 0.25, 0.10, 0.05]).astype("float32")
    trans["card4"] = rng.choice(
        ["visa", "mastercard", "american express", "discover", None],
        n, p=[0.50, 0.35, 0.08, 0.05, 0.02]
    )
    trans["card5"] = rng.choice(
        [226.0, 102.0, 166.0, 117.0, np.nan], n,
        p=[0.35, 0.25, 0.20, 0.15, 0.05]
    ).astype("float32")
    trans["card6"] = rng.choice(
        ["debit", "credit", "debit or credit", "charge card", None],
        n, p=[0.60, 0.30, 0.05, 0.03, 0.02]
    )

    # Address & distance
    trans["addr1"] = rng.choice(
        list(range(100, 540)) + [np.nan], n,
    )
    trans["addr1"] = trans["addr1"].astype("float32")
    trans["addr2"] = rng.choice(
        [87.0, 96.0, 65.0, np.nan], n, p=[0.60, 0.20, 0.15, 0.05]
    ).astype("float32")
    trans["dist1"] = np.where(rng.random(n) < 0.60, np.nan,
                              rng.lognormal(3, 1.5, n)).astype("float32")
    trans["dist2"] = np.where(rng.random(n) < 0.93, np.nan,
                              rng.lognormal(3, 1.5, n)).astype("float32")

    # Email domains
    p_domains = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com",
                 "outlook.com", "icloud.com", None]
    r_domains = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com",
                 "outlook.com", None, None]
    trans["P_emaildomain"] = rng.choice(
        p_domains, n, p=[0.35, 0.20, 0.10, 0.15, 0.08, 0.07, 0.05]
    )
    trans["R_emaildomain"] = rng.choice(
        r_domains, n, p=[0.25, 0.20, 0.10, 0.12, 0.08, 0.13, 0.12]
    )

    # C features (count-type, always observed, skewed)
    for i in range(1, 15):
        trans[f"C{i}"] = rng.integers(0, 3000, n).astype(np.int16)

    # D features (timedelta, partially missing)
    d_miss = [0.45, 0.05, 0.95, 0.60, 0.55, 0.95, 0.70, 0.20,
              0.90, 0.85, 0.80, 0.55, 0.40, 0.35, 0.30]
    for i, miss in enumerate(d_miss, start=1):
        vals = np.where(rng.random(n) < miss, np.nan,
                        rng.integers(0, 640, n).astype(float))
        trans[f"D{i}"] = vals.astype("float32")

    # M features (match flags T/F, partially missing)
    for i in range(1, 10):
        miss = 0.10 if i < 4 else 0.45
        vals = rng.choice(["T", "F", None], n,
                          p=[(1 - miss) * 0.65, (1 - miss) * 0.35, miss])
        trans[f"M{i}"] = vals

    # V features
    v_data = _make_v_blocks(rng, n)
    for col, arr in v_data.items():
        trans[col] = arr

    # ------------------------------------------------------------------
    # Identity table (~72 % of transactions)
    # ------------------------------------------------------------------
    id_mask = rng.random(n) < 0.72
    id_ids = trans.loc[id_mask, "TransactionID"].values
    n_id = len(id_ids)

    identity = pd.DataFrame()
    identity["TransactionID"] = id_ids
    identity["DeviceType"] = rng.choice(
        ["desktop", "mobile", None], n_id, p=[0.55, 0.40, 0.05]
    )
    identity["DeviceInfo"] = rng.choice(
        ["Windows", "iOS Device", "MacOS", "Android", "rv:11.0", None],
        n_id, p=[0.35, 0.22, 0.15, 0.18, 0.05, 0.05]
    )

    # id_01 to id_11: numeric
    for i in range(1, 12):
        miss = 0.05 if i <= 6 else 0.40
        vals = np.where(rng.random(n_id) < miss, np.nan,
                        rng.standard_normal(n_id) * 10)
        identity[f"id_{i:02d}"] = vals.astype("float32")

    # id_12 to id_38: categorical
    cat_options = {
        "id_12": (["Found", "NotFound", None], [0.85, 0.10, 0.05]),
        "id_15": (["New", "Found", "Unknown", None], [0.45, 0.40, 0.10, 0.05]),
        "id_16": (["Found", "NotFound", None], [0.75, 0.20, 0.05]),
        "id_23": (["IP_PROXY:TRANSPARENT", "IP_PROXY:ANONYMOUS", None],
                  [0.05, 0.03, 0.92]),
        "id_27": (["Found", "NotFound", None], [0.80, 0.15, 0.05]),
        "id_28": (["New", "Found", None], [0.50, 0.45, 0.05]),
        "id_29": (["Found", "NotFound", None], [0.75, 0.20, 0.05]),
        "id_30": (["Android 7.0", "iOS 11.1.2", "Windows 10", None],
                  [0.18, 0.22, 0.35, 0.25]),
        "id_31": (["chrome 62.0", "mobile safari 11.0", "ie 11.0", None],
                  [0.25, 0.22, 0.18, 0.35]),
        "id_33": (["1920x1080", "1366x768", "375x667", None],
                  [0.30, 0.25, 0.20, 0.25]),
        "id_34": (["match_status:2", "match_status:1", None],
                  [0.60, 0.30, 0.10]),
        "id_35": (["T", "F", None], [0.65, 0.30, 0.05]),
        "id_36": (["T", "F", None], [0.10, 0.85, 0.05]),
        "id_37": (["T", "F", None], [0.70, 0.25, 0.05]),
        "id_38": (["T", "F", None], [0.80, 0.15, 0.05]),
    }
    for col, (choices, probs) in cat_options.items():
        identity[col] = rng.choice(choices, n_id, p=probs)

    # Remaining numeric id columns
    for i in range(12, 39):
        col = f"id_{i:02d}"
        if col not in identity.columns:
            miss = 0.50
            vals = np.where(rng.random(n_id) < miss, np.nan,
                            rng.integers(0, 100, n_id).astype(float))
            identity[col] = vals.astype("float32")

    return trans.reset_index(drop=True), identity.reset_index(drop=True)
