"""Pandas mirror of the Snowflake baseline-feature SQL.

The Snowflake SQL (``sql/01_base_features.sql``) is the system of record. This
module reproduces the exact same window logic in pandas so the notebook can run
locally without a warehouse and so the temporal-leakage test has something to
assert against. Same boundary rules as the SQL:

    historical features:  SERVICE_DATE < INDEX_DATE
    target (NEXT_90D):    INDEX_DATE <= SERVICE_DATE < INDEX_DATE + 90d
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = [
    "COST_30D", "COST_90D", "COST_180D",
    "CLAIM_COUNT_30D", "CLAIM_COUNT_90D",
    "INPATIENT_COUNT_90D", "ED_COUNT_30D", "ED_COUNT_90D",
    "DISTINCT_PROVIDER_COUNT_90D", "DISTINCT_DIAGNOSIS_COUNT_90D",
    # Extra base column required only to derive INPATIENT_COST_SHARE.
    "INPATIENT_COST_90D",
]
# Columns fed to the baseline XGBoost model (spec 13.2 — excludes INPATIENT_COST_90D).
BASELINE_MODEL_FEATURES = [c for c in BASE_FEATURE_COLUMNS if c != "INPATIENT_COST_90D"]


def monthly_index_dates(index_date: str, count: int = 10) -> list[pd.Timestamp]:
    """Return ``count`` monthly index dates ending at ``index_date``."""
    idx = date.fromisoformat(index_date)
    out = []
    for k in range(count - 1, -1, -1):
        y = idx.year + (idx.month - 1 - k) // 12
        m = (idx.month - 1 - k) % 12 + 1
        out.append(pd.Timestamp(date(y, m, 1)))
    return out


def build_member_features(claims: pd.DataFrame, index_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Build MEMBER_FEATURES_BASE: one row per (member, index_date) with history."""
    claims = claims.sort_values("SERVICE_DATE")
    members = claims["MEMBER_ID"].unique()
    frames = []
    for idx in index_dates:
        idx = pd.Timestamp(idx)
        target_end = idx + timedelta(days=90)
        hist = claims[claims["SERVICE_DATE"] < idx]
        fut = claims[(claims["SERVICE_DATE"] >= idx) & (claims["SERVICE_DATE"] < target_end)]
        f = _windows(hist, idx)
        f = f.reindex(members).fillna(0.0)
        f["NEXT_90D_COST"] = (
            fut.groupby("MEMBER_ID")["PAID_AMOUNT"].sum().reindex(members).fillna(0.0)
        )
        f.insert(0, "INDEX_DATE", idx)
        f.insert(0, "MEMBER_ID", members)
        # Keep only members with any history before the index date.
        f = f[f["CLAIM_COUNT_180D"] > 0]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    return out.drop(columns=["CLAIM_COUNT_180D"])


def _windows(hist: pd.DataFrame, idx: pd.Timestamp) -> pd.DataFrame:
    d30, d90, d180 = idx - timedelta(days=30), idx - timedelta(days=90), idx - timedelta(days=180)
    w30 = hist[hist["SERVICE_DATE"] >= d30]
    w90 = hist[hist["SERVICE_DATE"] >= d90]
    w180 = hist[hist["SERVICE_DATE"] >= d180]

    def s(g, col):
        return g.groupby("MEMBER_ID")[col].sum()

    def c(g):
        return g.groupby("MEMBER_ID").size()

    out = pd.DataFrame({
        "COST_30D": s(w30, "PAID_AMOUNT"),
        "COST_90D": s(w90, "PAID_AMOUNT"),
        "COST_180D": s(w180, "PAID_AMOUNT"),
        "CLAIM_COUNT_30D": c(w30),
        "CLAIM_COUNT_90D": c(w90),
        "CLAIM_COUNT_180D": c(w180),
        "INPATIENT_COUNT_90D": s(w90, "INPATIENT_FLAG"),
        "ED_COUNT_30D": s(w30, "ED_FLAG"),
        "ED_COUNT_90D": s(w90, "ED_FLAG"),
        "DISTINCT_PROVIDER_COUNT_90D": w90.groupby("MEMBER_ID")["PROVIDER_ID"].nunique(),
        "DISTINCT_DIAGNOSIS_COUNT_90D": w90.groupby("MEMBER_ID")["DIAGNOSIS_GROUP"].nunique(),
        "INPATIENT_COST_90D": s(w90[w90["INPATIENT_FLAG"] == 1], "PAID_AMOUNT"),
    })
    return out.fillna(0.0)  # members absent from a sub-window aggregate to 0


if __name__ == "__main__":
    from generate_claims import generate_claims
    claims = generate_claims(member_count=200, random_seed=1)
    feats = build_member_features(claims, monthly_index_dates("2026-01-01", 6))
    assert set(BASE_FEATURE_COLUMNS).issubset(feats.columns)
    assert (feats["NEXT_90D_COST"] >= 0).all()
    print(f"{len(feats):,} feature rows, mean next-90d cost = {feats['NEXT_90D_COST'].mean():.0f}")
