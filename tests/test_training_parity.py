import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from config import settings
from src.models.xgboost_model import (
    build_metrics,
    calibrate,
    evaluate,
    load_data,
    load_model,
    predict,
    save_artifacts,
    split_data,
    train_model,
)

# Parity for the session-2 refactor (AUDIT.md §8 session 2): the importable functions,
# run with the v1 settings, must reproduce v1 exactly. v1 used n_jobs=-1, and xgboost
# hist results depend on the thread count, so this parity is only guaranteed on the
# machine that produced v1 (CLAUDE.md "Thread-count reproducibility").
ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics_v1.json"
ARTIFACTS = ROOT / "artifacts"
PARQUET = ROOT / "data" / "processed" / "hurdat2_processed_observations.parquet"
REQUIRED = [
    METRICS,
    PARQUET,
    ARTIFACTS / "test_predictions_v1.csv",
    ARTIFACTS / "xgb_ri_v1.joblib",
    ARTIFACTS / "feature_cols_v1.joblib",
    ARTIFACTS / "platt_calibrator_v1.joblib",
]
TOL = 1e-9
CSV_TOL = 1e-7


def assert_close(actual, expected, path="metrics"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert list(actual) == expected, path
    elif isinstance(expected, float):
        assert abs(actual - expected) < TOL, f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, path


@pytest.fixture(scope="module")
def v1():
    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        pytest.skip(f"v1 artifacts missing: {missing}")
    with open(METRICS) as f:
        metrics = json.load(f)
    df = load_data()
    train, val, test = split_data(df)
    model, scale_pos_weight = train_model(train, val)
    calibrator = calibrate(model, val, settings.FEATURE_COLS)
    result = evaluate(model, test, settings.FEATURE_COLS, calibrator=calibrator)
    return dict(
        metrics=metrics, train=train, val=val, test=test, model=model,
        scale_pos_weight=scale_pos_weight, calibrator=calibrator, result=result,
    )


def test_config_holds_v1_settings(v1):
    assert settings.FEATURE_COLS == v1["metrics"]["features"]
    assert settings.FEATURE_COLS == joblib.load(ARTIFACTS / "feature_cols_v1.joblib")
    assert settings.N_JOBS == -1
    assert (settings.TRAIN_MAX_YEAR, settings.VAL_MIN_YEAR, settings.VAL_MAX_YEAR, settings.TEST_MIN_YEAR) == (
        2015, 2016, 2019, 2020,
    )


def test_split_sizes(v1):
    assert (len(v1["train"]), len(v1["val"]), len(v1["test"])) == (11459, 1283, 1786)


def test_model_matches_v1(v1):
    assert v1["model"].best_iteration == 94
    assert abs(v1["scale_pos_weight"] - v1["metrics"]["scale_pos_weight"]) < TOL


def test_test_pr_auc_and_predictions(v1):
    assert abs(v1["result"]["pr_auc"] - 0.385453) < 1e-6
    assert abs(v1["result"]["pr_auc"] - v1["metrics"]["pr_auc_test"]) < TOL
    saved = pd.read_csv(ARTIFACTS / "test_predictions_v1.csv")
    assert np.abs(v1["result"]["y_prob"] - saved["y_prob"].to_numpy()).max() < CSV_TOL


def test_calibrator_matches_v1(v1):
    saved = joblib.load(ARTIFACTS / "platt_calibrator_v1.joblib")
    assert np.abs(v1["calibrator"].coef_ - saved.coef_).max() < TOL
    assert np.abs(v1["calibrator"].intercept_ - saved.intercept_).max() < TOL


def test_metrics_dict_matches_v1(v1):
    metrics = build_metrics(
        v1["train"], v1["val"], v1["test"], v1["scale_pos_weight"], v1["result"], settings.FEATURE_COLS
    )
    assert_close(metrics, v1["metrics"])


def test_save_artifacts_writes_v1_files(v1, tmp_path):
    metrics = build_metrics(
        v1["train"], v1["val"], v1["test"], v1["scale_pos_weight"], v1["result"], settings.FEATURE_COLS
    )
    art, res = tmp_path / "a", tmp_path / "r"
    save_artifacts(
        v1["model"], v1["calibrator"], settings.FEATURE_COLS, v1["test"], v1["result"], metrics,
        artifacts_dir=art, results_dir=res,
    )
    expected = {
        "test_predictions_v1.csv", "xgb_ri_v1.joblib", "feature_cols_v1.joblib",
        "platt_calibrator_v1.joblib", "calibration_v1.png", "feature_importance_v1.csv",
    }
    assert expected <= {p.name for p in art.iterdir()}
    with open(res / "metrics_v1.json") as f:
        assert_close(json.load(f), v1["metrics"])
    written = pd.read_csv(art / "test_predictions_v1.csv")
    saved = pd.read_csv(ARTIFACTS / "test_predictions_v1.csv")
    assert list(written.columns) == list(saved.columns)
    assert (written[["storm_id", "datetime", "RI"]] == saved[["storm_id", "datetime", "RI"]]).all().all()
    assert np.abs(written["y_prob"] - saved["y_prob"]).max() < CSV_TOL


def test_load_model_and_predict(v1):
    model, calibrator, feature_cols = load_model()
    assert feature_cols == settings.FEATURE_COLS
    X = v1["test"][feature_cols]
    expected = calibrator.predict_proba(model.predict_proba(X)[:, 1].reshape(-1, 1))[:, 1]
    np.testing.assert_allclose(predict(X), expected, rtol=0, atol=TOL)
    np.testing.assert_allclose(predict(X, model, calibrator), expected, rtol=0, atol=TOL)
    assert np.abs(predict(X) - v1["result"]["y_prob_cal"]).max() < CSV_TOL
