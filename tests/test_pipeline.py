"""Locks the demo's headline accept/reject at seed 42 (pure ML, no cloud).

Uses the synthetic feature path and drives the gate directly with the whitelist
features — Gemini itself requires live Vertex and is exercised end-to-end elsewhere.
"""
import pipeline as P


def test_demo_accept_reject_holds():
    df = P.features_synthetic(member_count=2000, seed=42)
    base = P.train_baseline(df)
    assert base.metrics["mae"] < base.median_metrics["mae"]  # XGBoost beats median

    candidates = ["COST_ACCELERATION", "PROVIDER_FRAGMENTATION", "INPATIENT_COST_SHARE"]
    results = {r["feature_name"]: r["decision"]
               for r in P.challenger_step(df, base, candidates)}
    assert results["COST_ACCELERATION"] == "ACCEPT"
    assert results["PROVIDER_FRAGMENTATION"] == "REJECT"
    assert results["INPATIENT_COST_SHARE"] == "REJECT"
