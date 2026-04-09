"""
tuning.py
---------
Reusable hyperparameter tuning utilities built on Optuna.

This module provides a thin, framework-agnostic layer on top of Optuna so
that any sklearn-compatible estimator can be tuned with minimal boilerplate.
It also offers a comparison helper that benchmarks Optuna against
``GridSearchCV`` and ``RandomizedSearchCV`` on the same model/data.

Public API
----------
run_optuna_study(objective_fn, n_trials, direction, study_name, use_pruner)
    → optuna.Study

sklearn_cv_objective(model_fn, X, y, cv, scoring)
    → Callable[[optuna.Trial], float]

compare_search_strategies(model_fn, param_grid, X, y, n_iter, cv)
    → pd.DataFrame

plot_optimization_history(study)    → None
plot_param_importances(study)       → None
"""

from __future__ import annotations

import time
from typing import Callable

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV


# ---------------------------------------------------------------------------
# Suppress verbose Optuna logging by default — callers can override.
# ---------------------------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Core Optuna helpers
# ---------------------------------------------------------------------------

def run_optuna_study(
    objective_fn: Callable,
    n_trials: int = 50,
    direction: str = "maximize",
    study_name: str | None = None,
    use_pruner: bool = True,
) -> "optuna.Study":
    """
    Create and run an Optuna study.

    Parameters
    ----------
    objective_fn : Callable[[optuna.Trial], float]
        Function that proposes hyperparameters via ``trial.suggest_*`` methods
        and returns a scalar metric to optimise.
    n_trials     : Number of trials to run.
    direction    : ``"maximize"`` (e.g. ROC-AUC) or ``"minimize"`` (e.g. loss).
    study_name   : Optional name for the study (useful when persisting to SQLite).
    use_pruner   : If True, attach a ``MedianPruner`` to cut unpromising trials
                   early (requires the objective to report intermediate values).

    Returns
    -------
    optuna.Study with all completed trials.
    """
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0) if use_pruner else None
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        pruner=pruner,
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study


def sklearn_cv_objective(
    model_fn: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> Callable:
    """
    Factory: wrap a ``model_fn(trial) -> estimator`` into an Optuna objective.

    The returned callable:
    1. Calls ``model_fn(trial)`` to build an sklearn estimator whose
       hyperparameters are sampled from the trial.
    2. Runs ``cross_val_score`` and returns the mean CV score.

    Parameters
    ----------
    model_fn : ``(optuna.Trial) -> sklearn estimator`` — proposes params and
               returns a freshly constructed (unfitted) estimator.
    X, y     : Full training dataset used for cross-validation.
    cv       : Number of CV folds.
    scoring  : sklearn scoring string (default: ``"roc_auc"``).

    Returns
    -------
    Callable suitable for ``run_optuna_study(objective_fn=...)``.
    """
    def objective(trial: "optuna.Trial") -> float:
        model = model_fn(trial)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return float(scores.mean())

    return objective


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

def compare_search_strategies(
    model_fn: Callable,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    optuna_objective_fn: Callable | None = None,
    n_iter: int = 50,
    n_optuna_trials: int = 50,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Benchmark GridSearchCV, RandomizedSearchCV, and Optuna on the same task.

    Parameters
    ----------
    model_fn             : Callable returning an unfitted sklearn estimator
                           (no arguments).  Used as the base for grid/random search.
    param_grid           : Parameter grid / distributions for grid and random search.
    X, y                 : Training data.
    optuna_objective_fn  : ``(optuna.Trial) -> float`` objective for Optuna.
                           If None, Optuna is skipped.
    n_iter               : Number of iterations for ``RandomizedSearchCV``.
    n_optuna_trials      : Number of trials for the Optuna study.
    cv                   : Number of CV folds.
    scoring              : sklearn scoring metric.
    random_state         : RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        strategy, best_score, search_time_s, n_evaluations
    """
    rows = []

    # --- GridSearchCV ---
    t0 = time.perf_counter()
    grid = GridSearchCV(
        model_fn(), param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=False,
    )
    grid.fit(X, y)
    grid_time = time.perf_counter() - t0
    rows.append({
        "strategy": "GridSearchCV",
        "best_score": grid.best_score_,
        "search_time_s": round(grid_time, 2),
        "n_evaluations": len(grid.cv_results_["mean_test_score"]),
    })

    # --- RandomizedSearchCV ---
    t0 = time.perf_counter()
    rand = RandomizedSearchCV(
        model_fn(), param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
        n_jobs=-1, refit=False, random_state=random_state,
    )
    rand.fit(X, y)
    rand_time = time.perf_counter() - t0
    rows.append({
        "strategy": "RandomizedSearchCV",
        "best_score": rand.best_score_,
        "search_time_s": round(rand_time, 2),
        "n_evaluations": n_iter,
    })

    # --- Optuna ---
    if optuna_objective_fn is not None:
        t0 = time.perf_counter()
        study = run_optuna_study(optuna_objective_fn, n_trials=n_optuna_trials, use_pruner=False)
        optuna_time = time.perf_counter() - t0
        rows.append({
            "strategy": "Optuna (Bayesian)",
            "best_score": study.best_value,
            "search_time_s": round(optuna_time, 2),
            "n_evaluations": len(study.trials),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_optimization_history(study: "optuna.Study") -> None:
    """
    Plot Optuna's trial-by-trial objective values.

    Requires ``optuna[visualization]`` (plotly) or falls back to matplotlib.

    Parameters
    ----------
    study : A completed ``optuna.Study`` instance.
    """
    try:
        import plotly  # noqa: F401
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
    except ImportError:
        import matplotlib.pyplot as plt
        values = [t.value for t in study.trials if t.value is not None]
        best_so_far = [max(values[: i + 1]) for i in range(len(values))]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(range(len(values)), values, alpha=0.5, label="Trial value", s=20)
        ax.plot(range(len(best_so_far)), best_so_far, color="red", label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Objective value")
        ax.set_title(f"Optimization history — {study.study_name or 'study'}")
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_param_importances(study: "optuna.Study") -> None:
    """
    Plot hyperparameter importances for a completed Optuna study.

    Requires ``optuna[visualization]`` (plotly) or falls back to a bar chart
    computed via Optuna's ``get_param_importances`` utility.

    Parameters
    ----------
    study : A completed ``optuna.Study`` instance with at least a few trials.
    """
    try:
        import plotly  # noqa: F401
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
    except ImportError:
        import matplotlib.pyplot as plt
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())
        vals = list(importances.values())
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.5)))
        ax.barh(names, vals)
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter importances")
        plt.tight_layout()
        plt.show()
