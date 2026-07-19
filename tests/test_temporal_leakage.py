"""Temporal-leakage guardrails on the pandas feature builder (mirror of the SQL)."""
from datetime import timedelta

import pandas as pd

from generate_claims import generate_claims
from features import build_member_features, monthly_index_dates, _windows


def test_history_excludes_index_date_and_future():
    """A claim on/after INDEX_DATE must not affect any historical window feature."""
    claims = pd.DataFrame({
        "MEMBER_ID": ["M1", "M1"],
        "SERVICE_DATE": pd.to_datetime(["2025-12-01", "2026-01-15"]),  # one before, one after
        "PAID_AMOUNT": [100.0, 999999.0],
        "INPATIENT_FLAG": [0, 1],
        "ED_FLAG": [0, 0],
        "PROVIDER_ID": ["P1", "P2"],
        "DIAGNOSIS_GROUP": ["OTHER", "OTHER"],
    })
    idx = pd.Timestamp("2026-01-01")
    hist = claims[claims["SERVICE_DATE"] < idx]
    feats = _windows(hist, idx)
    # The 2026-01-15 claim (>= INDEX_DATE) must be entirely absent from history.
    assert feats.loc["M1", "COST_90D"] == 100.0
    assert feats.loc["M1", "INPATIENT_COST_90D"] == 0.0


def test_target_only_in_next_90_days():
    claims = generate_claims(member_count=120, random_seed=7)
    index_dates = monthly_index_dates("2026-01-01", 6)
    feats = build_member_features(claims, index_dates)
    # Reconstruct target for one row and confirm it only counts the 90d window.
    row = feats.iloc[0]
    idx = pd.Timestamp(row["INDEX_DATE"])
    window = claims[(claims["MEMBER_ID"] == row["MEMBER_ID"])
                    & (claims["SERVICE_DATE"] >= idx)
                    & (claims["SERVICE_DATE"] < idx + timedelta(days=90))]
    assert abs(window["PAID_AMOUNT"].sum() - row["NEXT_90D_COST"]) < 1e-6


def test_no_omitted_ratio_features_in_base():
    claims = generate_claims(member_count=120, random_seed=7)
    feats = build_member_features(claims, monthly_index_dates("2026-01-01", 6))
    assert "COST_ACCELERATION" not in feats.columns
    assert "PROVIDER_FRAGMENTATION" not in feats.columns
