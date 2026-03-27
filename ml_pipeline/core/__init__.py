"""
ml_pipeline.core
================
Generic, reusable building blocks for ML pipelines.

Modules
-------
base_pipeline  : Abstract base class that every concrete pipeline must implement.
preprocessing  : Data encoding, column management, and train/test splitting.
model_utils    : Model training helpers.
evaluation     : Metrics computation and visualisation.
"""

from .base_pipeline import BasePipeline  # noqa: F401
from .preprocessing import (  # noqa: F401
    encode_categoricals,
    drop_columns,
    split_features_target,
    train_test_split_data,
)
from .model_utils import train_models  # noqa: F401
from .evaluation import evaluate_models, plot_confusion_matrix  # noqa: F401
