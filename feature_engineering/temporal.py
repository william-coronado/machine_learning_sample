"""
temporal.py
-----------
Extract time-based features from IEEE-CIS ``TransactionDT``.

``TransactionDT`` is a *timedelta* in seconds from a reference point
(not an absolute timestamp), so we can recover day-of-week, hour-of-day,
etc., by mapping it to a known reference date.

Public API
----------
extract_temporal_features(df, ref_date, col, cyclical) -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime


# Reference date inferred from competition data analysis
_DEFAULT_REF = datetime(2017, 11, 30)


def extract_temporal_features(
    df: pd.DataFrame,
    ref_date: datetime = _DEFAULT_REF,
    col: str = "TransactionDT",
    cyclical: bool = True,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Derive rich temporal features from a seconds-since-epoch column.

    Linear features extracted
    -------------------------
    - ``hour``        : Hour of day  [0, 23]
    - ``day_of_week`` : Day of week  [0 = Mon, 6 = Sun]
    - ``day_of_month``: Day of month [1, 31]
    - ``month``       : Month        [1, 12]
    - ``is_weekend``  : 1 if Sat/Sun, else 0
    - ``is_night``    : 1 if 22:00–05:59, else 0  (fraud signal)

    Cyclical features (sin/cos pairs, when ``cyclical=True``)
    ---------------------------------------------------------
    Cyclical encoding maps periodic features onto a unit circle so that
    23:00 and 00:00 are recognised as *close* by distance-based models:

        sin_hour = sin(2π * hour / 24)
        cos_hour = cos(2π * hour / 24)

    Same for day_of_week (period=7) and month (period=12).

    Parameters
    ----------
    df          : DataFrame containing ``col``.
    ref_date    : Reference datetime from which TransactionDT offsets are
                  measured.
    col         : Name of the timedelta-in-seconds column.
    cyclical    : If True, add sin/cos pairs for periodic features.
    drop_original : If True, drop ``col`` after extraction.

    Returns
    -------
    pd.DataFrame with new temporal columns appended.
    """
    df = df.copy()
    ref_ts = pd.Timestamp(ref_date)

    # Absolute timestamps
    dt_series = ref_ts + pd.to_timedelta(df[col], unit="s")

    # Linear features
    df["hour"] = dt_series.dt.hour.astype(np.int8)
    df["day_of_week"] = dt_series.dt.dayofweek.astype(np.int8)
    df["day_of_month"] = dt_series.dt.day.astype(np.int8)
    df["month"] = dt_series.dt.month.astype(np.int8)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_night"] = (
        (df["hour"] >= 22) | (df["hour"] < 6)
    ).astype(np.int8)

    # Transaction age in days from start of dataset (useful as a raw trend feature)
    df["tx_age_days"] = (df[col] / 86_400).astype(np.float32)

    if cyclical:
        df = _add_cyclical(df, "hour", period=24)
        df = _add_cyclical(df, "day_of_week", period=7)
        df = _add_cyclical(df, "month", period=12)

    if drop_original:
        df = df.drop(columns=[col])

    return df


def _add_cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Add sin/cos encoding for a periodic column."""
    angle = 2 * np.pi * df[col] / period
    df[f"sin_{col}"] = np.sin(angle).astype(np.float32)
    df[f"cos_{col}"] = np.cos(angle).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Email-domain features
# ---------------------------------------------------------------------------

def extract_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse P_emaildomain and R_emaildomain into structured sub-features.

    New columns
    -----------
    - ``P_email_tld``         : Top-level domain of purchaser email.
    - ``R_email_tld``         : Top-level domain of recipient email.
    - ``email_domain_match``  : 1 if purchaser == recipient domain, else 0.
    - ``P_is_protonmail``     : 1 if privacy-oriented provider.
    - ``R_is_protonmail``     : same for recipient.
    """
    df = df.copy()

    _privacy = {"protonmail.com", "tutanota.com", "guerrillamail.com",
                "anonymous.com", "yopmail.com"}

    for prefix in ("P", "R"):
        col = f"{prefix}_emaildomain"
        if col not in df.columns:
            continue
        domain = df[col].fillna("unknown")
        df[f"{prefix}_email_tld"] = domain.str.split(".").str[-1]
        df[f"{prefix}_is_privacy_email"] = domain.isin(_privacy).astype(np.int8)

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_domain_match"] = (
            df["P_emaildomain"] == df["R_emaildomain"]
        ).astype(np.int8)

    return df
