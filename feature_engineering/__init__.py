"""
Feature Engineering Toolkit for IEEE-CIS Fraud Detection.

Modules
-------
preprocessing  : Data loading, memory reduction, missing-value analysis & imputation.
encoders       : Target encoding, frequency encoding, leave-one-out encoding.
aggregations   : Group-level statistical feature generation.
temporal       : Time-delta decomposition and cyclical features.
selection      : Correlation pruning, variance thresholding, and importance-based selection.
utils          : Shared plotting helpers and evaluation utilities.
"""

from .preprocessing import (  # noqa: F401
    reduce_mem_usage,
    summarise_missing,
    impute_by_type,
    load_synthetic_ieee,
)
from .encoders import TargetEncoder, FrequencyEncoder  # noqa: F401
from .aggregations import GroupAggregator  # noqa: F401
from .temporal import extract_temporal_features  # noqa: F401
from .selection import (  # noqa: F401
    drop_high_correlation,
    variance_threshold_select,
    importance_select,
)
from .utils import plot_missing, plot_feature_importance, roc_comparison  # noqa: F401
