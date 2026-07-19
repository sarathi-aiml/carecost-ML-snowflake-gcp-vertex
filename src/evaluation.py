"""Metrics and the deterministic acceptance gate.

The gate — not Gemini's confidence or prose — decides ACCEPT / REJECT / REVIEW.
"""
from __future__ import annotations

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    high_cost_threshold: float,
) -> dict:
    """MAE, RMSE, RMSLE, and high-cost recall against a fixed cost cutoff.

    High-cost recall: of members whose *actual* cost is high, how many land in
    the top predicted-risk group (same-size top-k as the actual high-cost set).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))

    actual_high = y_true >= high_cost_threshold
    k = int(actual_high.sum())
    if k == 0 or np.ptp(y_pred) == 0:
        # No high-cost members, or constant predictions (e.g. the median baseline) —
        # ranking is undefined, so recall is not meaningful.
        recall = float("nan")
    else:
        top_pred = np.argsort(-y_pred)[:k]
        flagged = np.zeros_like(actual_high)
        flagged[top_pred] = True
        recall = float((actual_high & flagged).sum() / k)
    return {"mae": mae, "rmse": rmse, "rmsle": rmsle, "high_cost_recall": recall}


def decide_challenger(
    baseline: dict,
    challenger: dict,
    min_mae_improvement_pct: float,
    max_recall_drop: float,
) -> tuple[str, str]:
    """Return (decision, reason) in {ACCEPT, REJECT, REVIEW}."""
    mae_improvement_pct = (baseline["mae"] - challenger["mae"]) / baseline["mae"] * 100.0
    recall_drop = baseline["high_cost_recall"] - challenger["high_cost_recall"]

    if mae_improvement_pct < min_mae_improvement_pct:
        return "REJECT", "MAE improvement did not meet threshold"
    if recall_drop > max_recall_drop:
        return "REVIEW", "Overall error improved but high-cost recall declined"
    return "ACCEPT", "Holdout metrics passed configured gates"


def mae_improvement_pct(baseline_mae: float, challenger_mae: float) -> float:
    return (baseline_mae - challenger_mae) / baseline_mae * 100.0


if __name__ == "__main__":
    base = {"mae": 3000.0, "high_cost_recall": 0.70}
    assert decide_challenger(base, {"mae": 3100.0, "high_cost_recall": 0.70}, 1.0, 0.02)[0] == "REJECT"
    assert decide_challenger(base, {"mae": 2800.0, "high_cost_recall": 0.71}, 1.0, 0.02)[0] == "ACCEPT"
    assert decide_challenger(base, {"mae": 2800.0, "high_cost_recall": 0.60}, 1.0, 0.02)[0] == "REVIEW"
    print("evaluation gate ok")
