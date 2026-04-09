"""
base_pipeline.py
----------------
Abstract base class for all ML pipelines in this framework.

Concrete subclasses must implement:
    load_data()    → store raw data internally.
    preprocess()   → transform raw data into model-ready X_train/X_test/y_train/y_test.
    build_models() → return the list of sklearn-compatible estimators to train.
    train()        → fit models and store them.
    evaluate()     → compute and report metrics.

The ``run()`` method orchestrates all steps in order.

Example
-------
>>> pipeline = FraudDetectionPipeline(data_path="data/fraud.csv")
>>> pipeline.run()
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """
    Template for a complete ML pipeline.

    Subclasses implement the five abstract hooks; ``run()`` calls them
    in a fixed order so the high-level flow stays consistent across
    different use cases.
    """

    # ------------------------------------------------------------------
    # Abstract hooks — must be overridden
    # ------------------------------------------------------------------

    @abstractmethod
    def load_data(self) -> None:
        """Load raw data from disk, a URL, or any other source."""

    @abstractmethod
    def preprocess(self) -> None:
        """
        Transform raw data into train/test splits stored on ``self``.

        After this method, the instance must expose at minimum:
            self.X_train, self.X_test, self.y_train, self.y_test
        """

    @abstractmethod
    def build_models(self) -> list:
        """
        Return a list of un-fitted sklearn-compatible estimators.

        The list is passed to ``train()`` and ``evaluate()``.
        """

    @abstractmethod
    def train(self) -> None:
        """Fit all models returned by ``build_models()``."""

    @abstractmethod
    def evaluate(self) -> None:
        """Compute and report performance metrics for all fitted models."""

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full pipeline end-to-end.

        Steps (in order):
            1. load_data
            2. preprocess
            3. build_models  (stored internally for use by train/evaluate)
            4. train
            5. evaluate
        """
        print("=== [1/5] Loading data ===")
        self.load_data()

        print("\n=== [2/5] Preprocessing ===")
        self.preprocess()

        print("\n=== [3/5] Building models ===")
        self.models = self.build_models()
        print(f"  {len(self.models)} model(s) configured.")

        print("\n=== [4/5] Training ===")
        self.train()

        print("\n=== [5/5] Evaluation ===")
        self.evaluate()

        print("\n=== Pipeline complete ===")
