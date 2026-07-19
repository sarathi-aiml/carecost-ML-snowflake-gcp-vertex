"""Shared orchestration for the CareCost Fusion flow.

Thin functions that stitch the modules together so the Streamlit UI, the notebook,
and the Vertex Pipeline all drive the same logic. Every step works offline (pandas
features, Gemini stub, local experiment recorder) and lights up when creds exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from generate_claims import generate_claims
from features import build_member_features, monthly_index_dates, BASELINE_MODEL_FEATURES
from modeling import temporal_split, train_predict, median_baseline, high_cost_threshold, TARGET
from evaluation import compute_metrics, decide_challenger, mae_improvement_pct
from residuals import discover_segment, build_predictions_frame
from feature_catalog import validate_candidates, materialize, CATALOG
from gemini_hypothesis import propose_features
from vertex_geneval import evaluate_hypotheses

DEFAULT_MODEL_CFG = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.85, colsample_bytree=0.85, random_seed=42)


@dataclass
class Baseline:
    split: object
    model: object
    pred: np.ndarray
    metrics: dict
    median_metrics: dict
    high_cost: float
    residual: np.ndarray


def features_from_snowflake(conn, member_count=2000, history_months=15,
                            index_date="2026-01-01", seed=42) -> pd.DataFrame:
    """Load claims into Snowflake and build MEMBER_FEATURES_BASE **in-warehouse** (SQL).

    The live path for the app. Requires an open Snowflake connection.
    """
    from pathlib import Path
    from snowflake_io import run_sql_script, write_df, read_table
    sql = Path(__file__).resolve().parents[1] / "sql"
    claims = generate_claims(member_count, history_months, index_date, seed)
    run_sql_script(conn, sql / "00_setup.sql")
    load = claims.copy(); load["SERVICE_DATE"] = load["SERVICE_DATE"].dt.date
    write_df(conn, load, "RAW_CLAIMS")
    run_sql_script(conn, sql / "01_base_features.sql")
    df = read_table(conn, "SELECT * FROM MEMBER_FEATURES_BASE ORDER BY INDEX_DATE")
    df["INDEX_DATE"] = pd.to_datetime(df["INDEX_DATE"])
    return df


def features_synthetic(member_count=2000, history_months=15, index_date="2026-01-01",
                       seed=42) -> pd.DataFrame:
    """Build features from deterministic synthetic claims (pandas mirror of the SQL).

    Used by the Vertex Pipeline, whose Google-compute IPs aren't in the Snowflake
    network allowlist — a reproducible, self-contained data source, not a fallback.
    """
    return build_member_features(generate_claims(member_count, history_months, index_date, seed),
                                 monthly_index_dates(index_date, 10))


def train_baseline(df: pd.DataFrame, model_cfg=DEFAULT_MODEL_CFG) -> Baseline:
    split = temporal_split(df)
    thr = high_cost_threshold(split)
    model, pred = train_predict(split, BASELINE_MODEL_FEATURES, model_cfg)
    y = split.test[TARGET].to_numpy()
    return Baseline(split, model, pred, compute_metrics(y, pred, thr),
                    compute_metrics(y, median_baseline(split), thr), thr, y - pred)


def residual_segment(base: Baseline) -> dict:
    return discover_segment(base.split.test.reset_index(drop=True), base.residual, BASELINE_MODEL_FEATURES)


def gemini_step(segment: dict, project: str, location="us-central1",
                model="gemini-2.5-flash", available_columns=None) -> dict:
    """Propose (function-calling) → validate → score (Gen AI Eval). Requires live Vertex."""
    response = propose_features(segment, model, project, location)
    names = [c.feature_name for c in response.candidates]
    cols = set(available_columns) if available_columns is not None else set()
    accepted, rejected = validate_candidates(names, cols)
    eval_scores = evaluate_hypotheses(response.candidates, segment, project, location)
    return {"response": response, "accepted": accepted, "rejected": rejected, "eval_scores": eval_scores}


def challenger_step(df: pd.DataFrame, base: Baseline, accepted: list[str],
                    model_cfg=DEFAULT_MODEL_CFG, min_impr=1.0, max_recall_drop=0.02,
                    hyp_by_name=None) -> list[dict]:
    """Train one challenger per accepted feature; apply the deterministic gate."""
    results = []
    for name in accepted:
        fdf = df.copy()
        fdf[name] = materialize(fdf, name)
        csplit = temporal_split(fdf)
        _, cpred = train_predict(csplit, BASELINE_MODEL_FEATURES + [name], model_cfg)
        cm = compute_metrics(csplit.test[TARGET].to_numpy(), cpred, base.high_cost)
        decision, reason = decide_challenger(base.metrics, cm, min_impr, max_recall_drop)
        results.append({
            "run_id": f"challenger-{name.lower().replace('_', '-')}", "feature_name": name,
            "metrics": cm, "mae_improvement_pct": mae_improvement_pct(base.metrics["mae"], cm["mae"]),
            "decision": decision, "decision_reason": reason, "sql_expr": CATALOG[name].sql_expr,
            "hypothesis": (hyp_by_name or {}).get(name, ""),
        })
    return results
