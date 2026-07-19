"""Residual computation and interpretable error-segment discovery.

A shallow decision tree finds the leaf with the worst systematic
underprediction. Only an *aggregate* summary of that segment (counts and means,
plus human-readable conditions) is ever sent to Gemini — never member rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree

from feature_catalog import ALLOWED_FEATURE_NAMES


def build_predictions_frame(test_df: pd.DataFrame, pred_cost: np.ndarray, run_id: str, model_type: str) -> pd.DataFrame:
    """MODEL_PREDICTIONS-shaped frame with residual and absolute error."""
    actual = test_df["NEXT_90D_COST"].to_numpy()
    residual = actual - pred_cost
    return pd.DataFrame({
        "RUN_ID": run_id,
        "MODEL_TYPE": model_type,
        "MEMBER_ID": test_df["MEMBER_ID"].to_numpy(),
        "INDEX_DATE": pd.to_datetime(test_df["INDEX_DATE"]).dt.date,
        "ACTUAL_COST": actual,
        "PREDICTED_COST": pred_cost,
        "RESIDUAL": residual,
        "ABSOLUTE_ERROR": np.abs(residual),
    })


def discover_segment(test_df: pd.DataFrame, residual: np.ndarray, feature_cols: list[str]) -> dict:
    """Find the highest-underprediction leaf and return an aggregate summary."""
    positive = residual[residual > 0]
    cut = float(np.percentile(positive, 90)) if len(positive) else float("inf")
    y = (residual >= cut).astype(int)
    X = test_df[feature_cols]

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=0).fit(X, y)
    leaf_id = np.argmax(tree.tree_.value.ravel() * (tree.tree_.children_left == _tree.TREE_LEAF))
    mask = tree.apply(X) == leaf_id
    conditions = _leaf_conditions(tree, feature_cols, leaf_id)

    seg = test_df[mask]
    seg_resid = residual[mask]
    return {
        "segment_id": "segment_001",
        "segment_description": _describe(conditions),
        "member_count": int(mask.sum()),
        "mean_actual_cost": round(float(seg["NEXT_90D_COST"].mean()), 2),
        "mean_predicted_cost": round(float((seg["NEXT_90D_COST"].to_numpy() - seg_resid).mean()), 2),
        "mean_residual": round(float(seg_resid.mean()), 2),
        "conditions": conditions,
        "existing_features": feature_cols,
        "allowed_feature_families": list(ALLOWED_FEATURE_NAMES),
    }


def _leaf_conditions(tree, feature_cols, leaf_id) -> list[str]:
    t = tree.tree_
    path, node = [], 0
    # Walk root→leaf, recording the split direction taken.
    while node != leaf_id and t.children_left[node] != _tree.TREE_LEAF:
        feat = feature_cols[t.feature[node]]
        thr = t.threshold[node]
        left = _reaches(t, t.children_left[node], leaf_id)
        path.append((feat, "<=" if left else ">", thr))
        node = t.children_left[node] if left else t.children_right[node]
    # Collapse repeated splits on the same (feature, direction) to the tightest bound.
    tightest: dict[tuple[str, str], float] = {}
    for feat, op, thr in path:
        key = (feat, op)
        keep = min if op == "<=" else max
        tightest[key] = thr if key not in tightest else keep(tightest[key], thr)
    conds = [f"{feat} {op} {thr:.2f}" for (feat, op), thr in tightest.items()]
    return conds or ["(root leaf)"]


def _reaches(t, node, target) -> bool:
    if node == target:
        return True
    if t.children_left[node] == _tree.TREE_LEAF:
        return False
    return _reaches(t, t.children_left[node], target) or _reaches(t, t.children_right[node], target)


def _describe(conditions: list[str]) -> str:
    return "Segment where: " + "; ".join(conditions)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "MEMBER_ID": [f"M{i}" for i in range(400)],
        "INDEX_DATE": pd.Timestamp("2026-01-01"),
        "NEXT_90D_COST": rng.lognormal(7, 1, 400),
        "ED_COUNT_30D": rng.integers(0, 5, 400),
        "COST_90D": rng.lognormal(7, 1, 400),
    })
    resid = df["NEXT_90D_COST"].to_numpy() - rng.lognormal(6.5, 1, 400)
    s = discover_segment(df, resid, ["ED_COUNT_30D", "COST_90D"])
    assert s["member_count"] >= 0 and "conditions" in s
    print(s["segment_description"], "| n =", s["member_count"])
