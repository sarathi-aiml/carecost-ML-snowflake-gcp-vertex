import numpy as np

from evaluation import decide_challenger, compute_metrics

BASE = {"mae": 3000.0, "high_cost_recall": 0.70}


def test_worse_challenger_rejected():
    d, _ = decide_challenger(BASE, {"mae": 3100.0, "high_cost_recall": 0.70}, 1.0, 0.02)
    assert d == "REJECT"


def test_better_challenger_accepted():
    d, _ = decide_challenger(BASE, {"mae": 2800.0, "high_cost_recall": 0.71}, 1.0, 0.02)
    assert d == "ACCEPT"


def test_recall_degradation_review():
    d, _ = decide_challenger(BASE, {"mae": 2800.0, "high_cost_recall": 0.60}, 1.0, 0.02)
    assert d == "REVIEW"


def test_metrics_perfect_prediction():
    y = np.array([100.0, 5000.0, 200.0, 9000.0])
    m = compute_metrics(y, y, high_cost_threshold=5000.0)
    assert m["mae"] == 0.0 and m["rmsle"] == 0.0
    assert m["high_cost_recall"] == 1.0
