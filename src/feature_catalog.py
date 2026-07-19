"""Whitelist feature catalog and validators.

Gemini may only *name* features from this catalog. It never writes formulas or
SQL — the formula and the safe division live here, in code the model cannot
touch. Each candidate is validated (whitelisted, base columns present, produces
finite values) before any challenger trains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

MAX_CANDIDATES = 3


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    required_base_columns: tuple[str, ...]
    compute: Callable[[pd.DataFrame], pd.Series]
    sql_expr: str
    description: str


def _acc(df, num, denom, months):
    return df[num] / np.maximum(df[denom] / months, 1.0)


CATALOG: dict[str, CatalogEntry] = {
    "COST_ACCELERATION": CatalogEntry(
        "COST_ACCELERATION", ("COST_30D", "COST_90D"),
        lambda d: _acc(d, "COST_30D", "COST_90D", 3.0),
        "COST_30D / GREATEST(COST_90D / 3, 1)",
        "Recent 30-day cost vs prior 90-day monthly average.",
    ),
    "PROVIDER_FRAGMENTATION": CatalogEntry(
        "PROVIDER_FRAGMENTATION", ("DISTINCT_PROVIDER_COUNT_90D", "CLAIM_COUNT_90D"),
        lambda d: d["DISTINCT_PROVIDER_COUNT_90D"] / np.maximum(d["CLAIM_COUNT_90D"], 1.0),
        "DISTINCT_PROVIDER_COUNT_90D / GREATEST(CLAIM_COUNT_90D, 1)",
        "How distributed recent care is across providers.",
    ),
    "ED_ACCELERATION": CatalogEntry(
        "ED_ACCELERATION", ("ED_COUNT_30D", "ED_COUNT_90D"),
        lambda d: _acc(d, "ED_COUNT_30D", "ED_COUNT_90D", 3.0),
        "ED_COUNT_30D / GREATEST(ED_COUNT_90D / 3, 1)",
        "Recent emergency utilization vs longer-term monthly rate.",
    ),
    "INPATIENT_COST_SHARE": CatalogEntry(
        "INPATIENT_COST_SHARE", ("INPATIENT_COST_90D", "COST_90D"),
        lambda d: d["INPATIENT_COST_90D"] / np.maximum(d["COST_90D"], 1.0),
        "INPATIENT_COST_90D / GREATEST(COST_90D, 1)",
        "Share of recent cost attributable to inpatient services.",
    ),
}
ALLOWED_FEATURE_NAMES = tuple(CATALOG)


def validate_candidates(names: list[str], available_columns: set[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Split requested feature names into (accepted, [(name, reject_reason), ...])."""
    accepted: list[str] = []
    rejected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in names:
        if len(accepted) >= MAX_CANDIDATES:
            rejected.append((name, "exceeds max of three candidates"))
            continue
        if name not in CATALOG:
            rejected.append((name, "not in whitelist"))
            continue
        if name in seen or name in available_columns:
            rejected.append((name, "already present / duplicate"))
            continue
        missing = [c for c in CATALOG[name].required_base_columns if c not in available_columns]
        if missing:
            rejected.append((name, f"missing base column(s): {', '.join(missing)}"))
            continue
        accepted.append(name)
        seen.add(name)
    return accepted, rejected


def materialize(df: pd.DataFrame, name: str) -> pd.Series:
    """Compute a validated candidate feature; raise if values are unusable."""
    series = CATALOG[name].compute(df).replace([np.inf, -np.inf], np.nan)
    if series.isna().all():
        raise ValueError(f"{name} produced only missing/infinite values")
    return series.fillna(0.0)


if __name__ == "__main__":
    cols = set(BASE := ["COST_30D", "COST_90D", "CLAIM_COUNT_90D", "ED_COUNT_30D",
                        "ED_COUNT_90D", "DISTINCT_PROVIDER_COUNT_90D", "INPATIENT_COST_90D"])
    acc, rej = validate_candidates(
        ["COST_ACCELERATION", "PROVIDER_FRAGMENTATION", "FAKE", "ED_ACCELERATION", "INPATIENT_COST_SHARE"], cols
    )
    assert acc == ["COST_ACCELERATION", "PROVIDER_FRAGMENTATION", "ED_ACCELERATION"], acc
    assert any(r[0] == "FAKE" for r in rej)
    assert any(r[0] == "INPATIENT_COST_SHARE" for r in rej)  # dropped: over the 3-cap
    print("catalog ok:", acc)
