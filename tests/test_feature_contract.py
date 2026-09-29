import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from config import settings

# Feature contract (AUDIT.md §6 decision 2, §8 session 3): one feature list, in
# config/settings.py, equal to the list saved with the model; predict() only accepts a
# DataFrame with exactly those columns, in that order.
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
PARQUET = ROOT / "data" / "processed" / "hurdat2_processed_observations.parquet"
MODEL_FILES = [ARTIFACTS / f for f in (settings.MODEL_FILE, settings.CALIBRATOR_FILE, settings.FEATURE_COLS_FILE)]

# Calibrated P(RI) from the v2 model and calibrator for fixed test rows (session 6, 2026-09-29,
# Jaya approved). AL092025 2025-10-02 00Z (a v1 row) is EX at t+24, so out of v2 scope;
# IMELDA's first v2 row replaces it.
PARITY_ROWS = [
    ("AL022024", "2024-06-29 00:00:00", 0.07400466935879456),
    ("AL122024", "2024-10-03 00:00:00", 0.1513774809374762),
    ("AL012020", "2020-05-17 06:00:00", 0.049790325257491365),
    ("AL092025", "2025-09-28 00:00:00", 0.15077937120522422),
]


@pytest.fixture(scope="module")
def v2_model():
    missing = [str(p) for p in MODEL_FILES + [PARQUET] if not p.exists()]
    if missing:
        pytest.skip(f"v2 artifacts missing: {missing}")
    from src.models.xgboost_model import load_model
    return load_model()


@pytest.fixture(scope="module")
def X(v2_model):
    df = pd.read_parquet(PARQUET)
    keys = pd.DataFrame(PARITY_ROWS, columns=["storm_id", "datetime", "expected"])
    keys["datetime"] = pd.to_datetime(keys["datetime"])
    rows = keys.merge(df, on=["storm_id", "datetime"], how="left", validate="one_to_one")
    assert rows["vmax"].notna().all(), "parity rows missing from the parquet"
    return rows[settings.FEATURE_COLS], rows["expected"].to_numpy()


def test_predict_parity_on_fixed_rows(v2_model, X):
    from src.models.xgboost_model import predict, validate_features
    X, expected = X
    validate_features(X)
    np.testing.assert_allclose(predict(X, *v2_model[:2]), expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(predict(X), expected, rtol=0, atol=1e-9)


def test_wrong_order_raises_naming_expected_order(v2_model, X):
    from src.models.xgboost_model import predict, validate_features
    X, _ = X
    reordered = X[settings.FEATURE_COLS[::-1]]
    for call in (lambda: validate_features(reordered), lambda: predict(reordered, *v2_model[:2])):
        with pytest.raises(ValueError, match="order") as err:
            call()
        assert str(settings.FEATURE_COLS) in str(err.value)


def test_missing_column_raises_naming_it(v2_model, X):
    from src.models.xgboost_model import predict, validate_features
    X, _ = X
    dropped = X.drop(columns=["mslp", "month"])
    for call in (lambda: validate_features(dropped), lambda: predict(dropped, *v2_model[:2])):
        with pytest.raises(ValueError, match="missing") as err:
            call()
        assert "mslp" in str(err.value) and "month" in str(err.value)


def test_extra_column_raises_naming_it(v2_model, X):
    from src.models.xgboost_model import predict, validate_features
    X, _ = X
    extra = X.assign(day_of_year=1)
    for call in (lambda: validate_features(extra), lambda: predict(extra, *v2_model[:2])):
        with pytest.raises(ValueError, match="unexpected") as err:
            call()
        assert "day_of_year" in str(err.value)


@pytest.mark.parametrize("make", [
    lambda X: X.to_numpy(),
    lambda X: X.to_numpy().tolist(),
    lambda X: X.iloc[0],
], ids=["numpy", "list", "series"])
def test_non_dataframe_raises_type_error(v2_model, X, make):
    from src.models.xgboost_model import predict, validate_features
    bad = make(X[0])
    with pytest.raises(TypeError, match="DataFrame"):
        validate_features(bad)
    with pytest.raises(TypeError, match="DataFrame"):
        predict(bad, *v2_model[:2])


def test_non_numeric_column_raises_naming_it(v2_model, X):
    from src.models.xgboost_model import predict, validate_features
    X, _ = X
    bad = X.assign(month=X["month"].astype(str))
    for call in (lambda: validate_features(bad), lambda: predict(bad, *v2_model[:2])):
        with pytest.raises(TypeError, match="numeric") as err:
            call()
        assert "month" in str(err.value)


def test_boolean_column_raises_naming_it(v2_model, X):
    from src.models.xgboost_model import predict, validate_features
    X, _ = X
    bad = X.assign(month=X["month"] > 6)
    for call in (lambda: validate_features(bad), lambda: predict(bad, *v2_model[:2])):
        with pytest.raises(TypeError, match="numeric") as err:
            call()
        assert "month" in str(err.value)


def test_nan_raises_naming_columns_and_row_count(v2_model, X):
    # Rows without enough history (e.g. no 12 h lag) must not be scored; the app shows
    # "no forecast available" for them instead of calling predict().
    from src.models.xgboost_model import predict, validate_features
    X, _ = X
    bad = X.copy()
    bad.loc[bad.index[[0, 2]], "delta_vmax_12h"] = np.nan
    bad.loc[bad.index[2], "mslp"] = np.nan
    for call in (lambda: validate_features(bad), lambda: predict(bad, *v2_model[:2])):
        with pytest.raises(ValueError, match="NaN") as err:
            call()
        msg = str(err.value)
        assert "['mslp', 'delta_vmax_12h']" in msg
        assert "2 rows" in msg


def test_config_equals_saved_list_and_booster(v2_model):
    model, _, saved = v2_model
    assert saved == settings.FEATURE_COLS
    assert model.get_booster().feature_names == settings.FEATURE_COLS
    assert list(model.feature_names_in_) == settings.FEATURE_COLS


def test_load_model_rejects_saved_list_that_differs_from_config(v2_model, tmp_path):
    from src.models.xgboost_model import load_model
    for f in MODEL_FILES:
        shutil.copy(f, tmp_path / f.name)
    joblib.dump(settings.FEATURE_COLS[::-1], tmp_path / settings.FEATURE_COLS_FILE)
    with pytest.raises(ValueError, match="feature"):
        load_model(tmp_path)


def test_modelling_filter_uses_the_single_list():
    from data.parse_hurdat2 import select_modelling_rows
    base = {c: 1.0 for c in settings.FEATURE_COLS}
    df = pd.DataFrame([
        {**base, "RI": 1.0, "day_of_year": np.nan},   # kept: day_of_year is not a model feature
        {**base, "RI": 0.0, "mslp": np.nan},          # dropped: missing model feature
        {**base, "RI": np.nan},                       # dropped: no label
    ])
    out = select_modelling_rows(df)
    assert len(out) == 1 and out["RI"].tolist() == [1]
    assert out["RI"].dtype.kind == "i"
