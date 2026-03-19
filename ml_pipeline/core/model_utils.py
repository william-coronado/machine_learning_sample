"""
model_utils.py
--------------
Generic model-training helpers that work with any sklearn-compatible estimator.

Public API
----------
train_model(model, X_train, y_train)          → fitted model
train_models(models, X_train, y_train)        → list of fitted models
"""

from __future__ import annotations

import pandas as pd


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit a single sklearn-compatible estimator.

    Parameters
    ----------
    model   : Un-fitted estimator (must implement ``.fit()``).
    X_train : Training feature matrix.
    y_train : Training target vector.

    Returns
    -------
    The same model instance after fitting.
    """
    model.fit(X_train, y_train)
    return model


def train_models(
    models: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list:
    """
    Fit a list of sklearn-compatible estimators sequentially.

    Parameters
    ----------
    models  : List of un-fitted estimators.
    X_train : Training feature matrix.
    y_train : Training target vector.

    Returns
    -------
    The same list with every model fitted in place.
    """
    for model in models:
        print(f"  Fitting {type(model).__name__} ...")
        train_model(model, X_train, y_train)
    return models
