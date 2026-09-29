import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_score, recall_score

# v1 golden-metrics safety net (AUDIT.md §8 session 0, Check F logic).
# Must keep passing until the session-6 retrain re-baselines it (with Jaya's approval).
ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "artifacts" / "test_predictions_v1.csv"
METRICS = ROOT / "results" / "metrics_v1.json"
PARQUET = ROOT / "data" / "processed" / "hurdat2_processed_observations.parquet"
TOL = 1e-9


@pytest.fixture(scope="module")
def golden():
    missing = [str(p) for p in (PREDICTIONS, METRICS, PARQUET) if not p.exists()]
    if missing:
        pytest.skip(f"v1 artifacts missing: {missing}")
    preds = pd.read_csv(PREDICTIONS)
    with open(METRICS) as f:
        metrics = json.load(f)
    return preds, metrics


def test_headline_metrics_match(golden):
    preds, metrics = golden
    assert len(preds) == metrics["n_test"]
    assert abs(preds["RI"].mean() - metrics["ri_rate_test"]) < TOL
    assert abs(average_precision_score(preds["RI"], preds["y_prob"]) - metrics["pr_auc_test"]) < TOL


def test_threshold_metrics_match(golden):
    preds, metrics = golden
    for threshold, expected in metrics["thresholds"].items():
        y_pred = (preds["y_prob"] >= float(threshold)).astype(int)
        assert abs(precision_score(preds["RI"], y_pred, zero_division=0) - expected["precision"]) < TOL
        assert abs(recall_score(preds["RI"], y_pred, zero_division=0) - expected["recall"]) < TOL
        assert int(y_pred.sum()) == expected["n_predicted_positive"]


def test_persistence_baseline_matches(golden):
    _, metrics = golden
    df = pd.read_parquet(PARQUET)
    storm_year = df.groupby("storm_id")["datetime"].transform("min").dt.year
    test_rows = df[storm_year >= 2020]
    assert len(test_rows) == metrics["n_test"]
    persist = (test_rows["delta_vmax_12h"] > 0).astype(float)
    assert abs(average_precision_score(test_rows["RI"], persist) - metrics["pr_auc_persistence"]) < TOL
