"""Temporal split, median baseline, and XGBoost training.

Cost is right-skewed, so models train on log1p(cost) and predict back with
expm1 + clip. Every model (baseline and challengers) uses the identical
chronological split and hyperparameters — only the feature set changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TARGET = "NEXT_90D_COST"


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def temporal_split(df: pd.DataFrame) -> Split:
    """Chronological 60/20/20 split on distinct INDEX_DATE values."""
    dates = np.sort(df["INDEX_DATE"].unique())
    n = len(dates)
    if n < 3:
        raise ValueError("need at least 3 distinct index dates for a temporal split")
    t_end = dates[int(n * 0.6) - 1]
    v_end = dates[int(n * 0.8) - 1]
    train = df[df["INDEX_DATE"] <= t_end]
    val = df[(df["INDEX_DATE"] > t_end) & (df["INDEX_DATE"] <= v_end)]
    test = df[df["INDEX_DATE"] > v_end]
    return Split(train, val, test, pd.Timestamp(t_end), pd.Timestamp(v_end), pd.Timestamp(dates[-1]))


def make_model(model_cfg: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        subsample=model_cfg["subsample"],
        colsample_bytree=model_cfg["colsample_bytree"],
        random_state=model_cfg["random_seed"],
        objective="reg:squarederror",
        n_jobs=0,
    )


def predict_cost(model, X) -> np.ndarray:
    """Invert the log1p training target: expm1 + clip to non-negative dollars."""
    return np.clip(np.expm1(model.predict(X)), 0, None)


def train_predict(split: Split, feature_cols: list[str], model_cfg: dict):
    """Train on log1p(target); return (fitted_model, test_pred_cost in dollars)."""
    model = make_model(model_cfg)
    model.fit(split.train[feature_cols], np.log1p(split.train[TARGET].to_numpy()))
    return model, predict_cost(model, split.test[feature_cols])


def median_baseline(split: Split) -> np.ndarray:
    """Naïve constant prediction = training-set median next-90-day cost."""
    med = float(split.train[TARGET].median())
    return np.full(len(split.test), med)


def high_cost_threshold(split: Split, pct: float = 90.0) -> float:
    """Fixed high-cost cutoff from the TRAINING target (reused across all runs)."""
    return float(np.percentile(split.train[TARGET].to_numpy(), pct))
