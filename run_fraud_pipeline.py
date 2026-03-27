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

from ml_pipeline.use_cases.fraud_pipeline import FraudDetectionPipeline
from ml_pipeline.use_cases.fraud_detection import FRAUD_THRESHOLD


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
        type=float,
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
        type=float,
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
