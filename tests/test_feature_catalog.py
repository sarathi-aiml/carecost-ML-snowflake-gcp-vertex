import numpy as np
import pandas as pd
import pytest

from feature_catalog import validate_candidates, materialize, CATALOG

BASE_COLS = {
    "COST_30D", "COST_90D", "CLAIM_COUNT_90D", "ED_COUNT_30D", "ED_COUNT_90D",
    "DISTINCT_PROVIDER_COUNT_90D", "INPATIENT_COST_90D",
}


def test_only_whitelist_accepted():
    acc, rej = validate_candidates(["COST_ACCELERATION", "NOT_A_FEATURE"], BASE_COLS)
    assert acc == ["COST_ACCELERATION"]
    assert ("NOT_A_FEATURE", "not in whitelist") in rej


def test_limited_to_three():
    acc, rej = validate_candidates(list(CATALOG) + ["COST_ACCELERATION"], BASE_COLS)
    assert len(acc) == 3
    assert any("three" in r[1] for r in rej)


def test_missing_base_column_rejected():
    acc, rej = validate_candidates(["INPATIENT_COST_SHARE"], BASE_COLS - {"INPATIENT_COST_90D"})
    assert acc == []
    assert any("missing base column" in r[1] for r in rej)


def test_division_by_zero_protected():
    df = pd.DataFrame({"COST_30D": [500.0], "COST_90D": [0.0]})
    val = materialize(df, "COST_ACCELERATION")
    assert np.isfinite(val).all()


def test_all_missing_values_raises():
    df = pd.DataFrame({"COST_30D": [np.nan], "COST_90D": [np.nan]})
    with pytest.raises(ValueError):
        materialize(df, "COST_ACCELERATION")
