"""
run_fraud_pipeline.py
---------------
Entry point for the telco online-payment fraud detection pipeline.

Usage
-----
Run with defaults:
    python3 run_fraud_pipeline.py

Override data path and split:
    python3 run_fraud_pipeline.py --data-path data/online_payment_fraud_data.csv \
                            --test-size 0.2 \
                            --best-model-idx 1
"""

import argparse
import os
import warnings

from ml_pipeline.use_cases.fraud_detection import FRAUD_THRESHOLD, get_fraud_models
from ml_pipeline.use_cases.fraud_pipeline import FraudDetectionPipeline


def _valid_test_size(value: str) -> float:
    test_size = float(value)
    if not 0.0 < test_size < 1.0:
        raise argparse.ArgumentTypeError("--test-size must be between 0 and 1 (exclusive).")
    return test_size


def _valid_threshold(value: str) -> float:
    threshold = float(value)
    if not 0.0 < threshold < 1.0:
        raise argparse.ArgumentTypeError("--threshold must be between 0 and 1 (exclusive).")
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the telco fraud detection ML pipeline."
    )
    parser.add_argument(
        "--data-path",
        default="data/online_payment_fraud_data.csv",
        help="Path to the fraud CSV (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--test-size",
        type=_valid_test_size,
        default=0.30,
        help="Fraction of data reserved for testing (default: 0.30).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--threshold",
        type=_valid_threshold,
        default=FRAUD_THRESHOLD,
        help=f"Decision threshold for fraud classification (default: {FRAUD_THRESHOLD}).",
    )
    parser.add_argument(
        "--best-model-idx",
        type=int,
        default=1,
        help="Index of the model whose confusion matrix is displayed (default: 1 = XGBClassifier).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.data_path):
        warnings.warn(
            f"Data file not found at {args.data_path!r}; the pipeline will attempt to download it "
            "from the configured dataset URL.",
            stacklevel=2,
        )

    model_count = len(get_fraud_models())
    if args.best_model_idx < 0 or args.best_model_idx >= model_count:
        raise ValueError(
            f"--best-model-idx must be between 0 and {model_count - 1}. "
            f"Got {args.best_model_idx}."
        )

    pipeline = FraudDetectionPipeline(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        threshold=args.threshold,
        best_model_idx=args.best_model_idx,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
