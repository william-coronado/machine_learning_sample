"""
fraud_pipeline.py
-----------------
Concrete ML pipeline for telco online-payment fraud detection.

``FraudDetectionPipeline`` wires together:
  - Generic core utilities  (ml_pipeline.core)
  - Telco-specific logic    (ml_pipeline.use_cases.fraud_detection)

It inherits the ``run()`` template from ``BasePipeline``, which executes the
five lifecycle hooks in order:
    load_data → preprocess → build_models → train → evaluate

Example
-------
>>> from ml_pipeline.use_cases.fraud_pipeline import FraudDetectionPipeline
>>> pipeline = FraudDetectionPipeline(data_path="data/online_payment_fraud_data.csv")
>>> pipeline.run()
"""

from __future__ import annotations

import pandas as pd

from ml_pipeline.core.base_pipeline import BasePipeline
from ml_pipeline.core.preprocessing import split_features_target, train_test_split_data
from ml_pipeline.core.model_utils import train_models
from ml_pipeline.core.evaluation import evaluate_models, plot_confusion_matrix

from ml_pipeline.use_cases.fraud_detection import (
    TARGET_COL,
    FRAUD_THRESHOLD,
    engineer_telco_features,
    get_fraud_models,
    load_telco_fraud_data,
    DATA_URL,
)


class FraudDetectionPipeline(BasePipeline):
    """
    End-to-end fraud detection pipeline for the telco online-payment dataset.

    Parameters
    ----------
    data_path    : Local path to the CSV file.
    data_url     : Google Drive URL (used only if the file does not exist).
    test_size    : Fraction of data reserved for evaluation (default 0.30).
    random_state : RNG seed for the train/test split.
    threshold    : Decision threshold for the confusion matrix.
                   Defaults to ``FRAUD_THRESHOLD`` (0.5).
    best_model_idx : Index of the model in ``build_models()`` whose confusion
                     matrix is displayed in ``evaluate()``.  Default is 1
                     (XGBClassifier, the best-performing model in the notebook).
    """

    def __init__(
        self,
        data_path: str = "data/online_payment_fraud_data.csv",
        data_url: str = DATA_URL,
        test_size: float = 0.30,
        random_state: int = 42,
        threshold: float = FRAUD_THRESHOLD,
        best_model_idx: int = 1,
    ) -> None:
        self.data_path = data_path
        self.data_url = data_url
        self.test_size = test_size
        self.random_state = random_state
        self.threshold = threshold
        self.best_model_idx = best_model_idx

        # Set by lifecycle hooks
        self._raw: pd.DataFrame | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.models: list = []
        self.eval_results: list[dict] = []

    # ------------------------------------------------------------------
    # BasePipeline hooks
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Download (if needed) and read the telco fraud CSV."""
        self._raw = load_telco_fraud_data(
            filepath=self.data_path,
            url=self.data_url,
        )
        print(f"  Loaded {len(self._raw):,} rows × {self._raw.shape[1]} columns.")

    def preprocess(self) -> None:
        """
        Apply telco-specific feature engineering then split into train/test.

        Steps
        -----
        1. ``engineer_telco_features`` — one-hot encode ``type``, drop ID cols.
        2. ``split_features_target``   — separate X and y.
        3. ``train_test_split_data``   — 70 / 30 stratified split.
        """
        if self._raw is None:
            raise ValueError("Raw data not loaded. Call load_data() first.")
        df = engineer_telco_features(self._raw)

        X, y = split_features_target(df, target_col=TARGET_COL)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_data(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=True,
        )
        print(
            f"  Train: {len(self.X_train):,} rows | "
            f"Test: {len(self.X_test):,} rows | "
            f"Features: {self.X_train.shape[1]}"
        )

    def build_models(self) -> list:
        """Return telco-fraud-tuned estimators (LR, XGBClassifier, RF)."""
        return get_fraud_models()

    def train(self) -> None:
        """Fit all models on the training set."""
        self.models = train_models(self.models, self.X_train, self.y_train)

    def evaluate(self) -> None:
        """
        Report AUC-ROC for every model; display confusion matrix for the best.

        The ``best_model_idx`` parameter selects which model's confusion matrix
        to display (default: XGBClassifier at index 1).
        """
        if not self.models:
            raise ValueError(
                "No trained models are available. Make sure to call `train()` before `evaluate()`."
            )

        if not isinstance(self.best_model_idx, int):
            raise ValueError(
                f"best_model_idx must be an integer, got {type(self.best_model_idx).__name__!r}."
            )

        if self.best_model_idx < 0 or self.best_model_idx >= len(self.models):
            model_names = [type(m).__name__ for m in self.models]
            valid_indices = list(range(len(self.models)))
            raise ValueError(
                "Invalid best_model_idx {idx}. Valid indices are {indices} "
                "corresponding to models {names}.".format(
                    idx=self.best_model_idx,
                    indices=valid_indices,
                    names=model_names,
                )
            )

        self.eval_results = evaluate_models(
            self.models,
            self.X_train, self.y_train,
            self.X_test, self.y_test,
        )

        best = self.models[self.best_model_idx]
        print(f"Confusion matrix for: {type(best).__name__}")
        plot_confusion_matrix(best, self.X_test, self.y_test, threshold=self.threshold)
