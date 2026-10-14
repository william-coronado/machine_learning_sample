import pandas as pd
import pytest

from ml_pipeline.core.preprocessing import (
    drop_columns,
    encode_categoricals,
    split_features_target,
    train_test_split_data,
)
from ml_pipeline.use_cases.fraud_detection import load_telco_fraud_data
from ml_pipeline.use_cases.fraud_pipeline import FraudDetectionPipeline


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
            "amount": [10.0, 25.0, 30.0],
            "isFraud": [0, 1, 0],
            "nameOrig": ["a", "b", "c"],
        }
    )


def test_encode_categoricals_adds_dummy_columns(sample_df):
    encoded = encode_categoricals(sample_df, ["type"], drop_first=True)

    assert "type" in encoded.columns
    assert "type_PAYMENT" in encoded.columns
    assert "type_TRANSFER" in encoded.columns
    assert "type_CASH_OUT" not in encoded.columns


def test_split_features_target_and_split_invariants(sample_df):
    X, y = split_features_target(sample_df, target_col="isFraud")

    assert list(X.columns) == ["type", "amount", "nameOrig"]
    assert y.name == "isFraud"
    assert y.tolist() == [0, 1, 0]

    y_balanced = pd.Series([0, 0, 1, 1], name="isFraud")
    X_balanced = pd.DataFrame(
        {
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
            "amount": [10.0, 25.0, 30.0, 35.0],
            "nameOrig": ["a", "b", "c", "d"],
        }
    )

    X_train, X_test, y_train, y_test = train_test_split_data(
        X_balanced, y_balanced, test_size=0.5, random_state=42, stratify=True
    )

    assert len(X_train) == len(y_train) == 2
    assert len(X_test) == len(y_test) == 2
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_drop_columns_ignores_missing_names(sample_df):
    cleaned = drop_columns(sample_df, ["type", "missing_col"])

    assert "type" not in cleaned.columns
    assert "amount" in cleaned.columns


def test_invalid_best_model_idx_raises_value_error():
    pipeline = FraudDetectionPipeline(best_model_idx=99)
    pipeline.models = [object(), object()]

    with pytest.raises(ValueError, match="Invalid best_model_idx"):
        pipeline.evaluate()


def test_load_telco_fraud_data_raises_on_missing_file(tmp_path):
    missing = tmp_path / "missing.csv"

    with pytest.raises(RuntimeError, match="Failed to download"):
        load_telco_fraud_data(filepath=str(missing), url="https://example.com/does-not-matter.csv")


def test_load_telco_fraud_data_raises_on_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")

    with pytest.raises(RuntimeError, match="empty"):
        load_telco_fraud_data(filepath=str(empty_file), url="https://example.com/ignored.csv")


def test_load_telco_fraud_data_raises_on_invalid_csv(tmp_path):
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("not,a,valid,csv\n1,2\n")

    with pytest.raises(RuntimeError, match="missing required columns"):
        load_telco_fraud_data(filepath=str(invalid), url="https://example.com/ignored.csv")


def test_fraud_pipeline_run_smoke(monkeypatch):
    synthetic = pd.DataFrame(
        {
            "step": list(range(8)),
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
            "amount": [10.0, 15.0, 20.0, 25.0, 12.0, 18.0, 22.0, 28.0],
            "oldbalanceOrg": [100.0, 200.0, 300.0, 400.0, 110.0, 210.0, 310.0, 410.0],
            "newbalanceOrig": [90.0, 185.0, 280.0, 375.0, 98.0, 192.0, 288.0, 382.0],
            "oldbalanceDest": [0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 110.0, 110.0],
            "newbalanceDest": [0.0, 0.0, 90.0, 90.0, 0.0, 0.0, 98.0, 98.0],
            "nameOrig": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "nameDest": ["x", "y", "z", "q", "r", "s", "t", "u"],
            "isFlaggedFraud": [0, 0, 0, 0, 0, 0, 0, 0],
            "isFraud": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    monkeypatch.setattr(
        "ml_pipeline.use_cases.fraud_pipeline.load_telco_fraud_data",
        lambda filepath, url: synthetic,
    )

    pipeline = FraudDetectionPipeline(data_path="fake.csv", test_size=0.5, random_state=42, best_model_idx=1, show_plots=False)
    pipeline.load_data()
    pipeline.preprocess()
    pipeline.models = pipeline.build_models()
    pipeline.train()
    pipeline.evaluate()

    assert len(pipeline.eval_results) == 3
    assert pipeline.eval_results[1]["name"] == "XGBClassifier"
    assert pipeline.X_train is not None and pipeline.X_test is not None
