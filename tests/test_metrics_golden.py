import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score

from config import settings
from src.models import xgboost_model as xm

# v2 golden-metrics test (AUDIT.md §8 session 6; replaces the v1 test, Jaya approved).
# Recomputes the saved metrics from the saved calibrated predictions and the parquet
# (Check F logic), including the baseline comparison of the session 6 evaluation plan.
PREDICTIONS = settings.ARTIFACTS_DIR / settings.TEST_PREDICTIONS_FILE
METRICS = settings.RESULTS_DIR / settings.METRICS_FILE
PARQUET = settings.PROCESSED_DATA_PATH
TOL = 1e-9


@pytest.fixture(scope="module")
def golden():
    missing = [str(p) for p in (PREDICTIONS, METRICS, PARQUET) if not p.exists()]
    if missing:
        pytest.skip(f"{settings.MODEL_VERSION} artifacts missing: {missing}")
    preds = pd.read_csv(PREDICTIONS, parse_dates=["datetime"], float_precision="round_trip")
    with open(METRICS) as f:
        metrics = json.load(f)
    return preds, metrics


@pytest.fixture(scope="module")
def parquet(golden):
    df = pd.read_parquet(PARQUET)
    return df.assign(storm_year=df.groupby("storm_id")["datetime"].transform("min").dt.year)


@pytest.fixture(scope="module")
def persistence(golden, parquet):
    preds, _ = golden
    rows = preds.merge(parquet, on=["storm_id", "datetime"], how="left", validate="one_to_one")
    assert rows["delta_vmax_12h"].notna().all(), "prediction rows missing from the parquet"
    return (rows["delta_vmax_12h"] > 0).astype(float).to_numpy()


def test_model_version(golden):
    _, metrics = golden
    assert metrics["model"] == settings.MODEL_NAME and settings.MODEL_VERSION == "v2"


def test_test_rows_match_parquet(golden, parquet):
    preds, metrics = golden
    test_rows = parquet[parquet["storm_year"] >= settings.TEST_MIN_YEAR]
    assert len(preds) == len(test_rows) == metrics["n_test"]
    assert set(preds["storm_id"]) == set(test_rows["storm_id"])
    assert abs(preds["RI"].mean() - metrics["ri_rate_test"]) < TOL


def test_headline_metrics_match(golden):
    preds, metrics = golden
    y = preds["RI"]
    assert abs(average_precision_score(y, preds["y_prob"]) - metrics["pr_auc_test_raw"]) < TOL
    assert abs(average_precision_score(y, preds["y_prob_cal"]) - metrics["pr_auc_test_cal"]) < TOL
    assert abs(brier_score_loss(y, preds["y_prob"]) - metrics["brier_test_raw"]) < TOL
    assert abs(brier_score_loss(y, preds["y_prob_cal"]) - metrics["brier_test_cal"]) < TOL
    assert preds["y_prob_cal"].between(0, 1).all()


def test_threshold_and_band_metrics_match(golden):
    preds, metrics = golden
    edges = metrics["risk_bands"]["edges"]
    assert set(metrics["thresholds_cal"]) == {f"{e:.6f}" for e in edges}
    for edge in edges:
        expected = metrics["thresholds_cal"][f"{edge:.6f}"]
        y_pred = (preds["y_prob_cal"] >= edge).astype(int)
        assert abs(precision_score(preds["RI"], y_pred, zero_division=0) - expected["precision"]) < TOL
        assert abs(recall_score(preds["RI"], y_pred, zero_division=0) - expected["recall"]) < TOL
        assert int(y_pred.sum()) == expected["n_predicted_positive"]

    band_index = np.searchsorted(edges, preds["y_prob_cal"], side="right")
    counts = {label: int((band_index == i).sum()) for i, label in enumerate(metrics["risk_bands"]["labels"])}
    assert counts == metrics["band_counts_test"]


def test_bands_from_calibration_years(golden, parquet):
    _, metrics = golden
    cal_rows = parquet[parquet["storm_year"].between(settings.VAL_MIN_YEAR, settings.VAL_MAX_YEAR)]
    assert len(cal_rows) == metrics["n_val"]
    assert abs(cal_rows["RI"].mean() - metrics["risk_bands"]["observed_rate"]) < TOL


@pytest.mark.slow
def test_pr_auc_ci_match(golden):
    preds, metrics = golden
    saved = metrics["pr_auc_test_ci"]
    ci = xm.bootstrap_pr_auc_ci(preds["storm_id"], preds["RI"], preds["y_prob_cal"],
                                n_boot=saved["n_boot"], seed=saved["seed"], level=saved["level"])
    for key in ("point", "lower", "upper"):
        assert abs(ci[key] - saved[key]) < TOL, key
    assert ci["n_valid"] == saved["n_valid"]


@pytest.mark.slow
def test_baselines_match(golden, persistence):
    preds, metrics = golden
    saved = metrics["baselines"]
    y = preds["RI"].to_numpy()
    assert saved["n_rows"] == len(y) and saved["n_pos"] == int(y.sum())
    assert saved["climatology_rate"] == metrics["risk_bands"]["observed_rate"]
    assert abs(average_precision_score(y, persistence) - metrics["pr_auc_persistence"]) < TOL

    boot = saved["bootstrap"]
    again = xm.baseline_comparison(preds["storm_id"], y, preds["y_prob_cal"].to_numpy(), persistence,
                                   saved["climatology_rate"],
                                   n_boot=boot["n_boot"], seed=boot["seed"], level=boot["level"])
    assert again["bootstrap"] == boot
    for name in ("model", "persistence", "climatology", "model_minus_persistence"):
        for key in ("point", "lower", "upper"):
            assert abs(again["pr_auc"][name][key] - saved["pr_auc"][name][key]) < TOL, (name, key)
    for key in ("point", "lower", "upper"):
        assert abs(again["brier_skill_score"][key] - saved["brier_skill_score"][key]) < TOL, key
    for name in ("model", "climatology"):
        assert abs(again["brier"][name] - saved["brier"][name]) < TOL, name

    # Same rows and same resamples as the headline CI
    assert saved["pr_auc"]["model"]["lower"] == metrics["pr_auc_test_ci"]["lower"]
    assert saved["pr_auc"]["model"]["upper"] == metrics["pr_auc_test_ci"]["upper"]
