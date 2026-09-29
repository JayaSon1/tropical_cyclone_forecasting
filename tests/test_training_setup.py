import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from config import settings
from src.models import xgboost_model as xm

# Training setup (AUDIT.md §6 decisions 3, 6, 12, 12b; §8 session 5): train on storms
# with genesis 1980-2015; early stopping by season-grouped CV inside train, median
# rounds, refit; Platt, thresholds and bands fitted on 2016-19 only; test (2020+)
# untouched until evaluation; n_jobs pinned in config. Synthetic data only, no retrain.
FEATURES = settings.FEATURE_COLS
SMALL = dict(settings.XGB_PARAMS, n_estimators=40, early_stopping_rounds=5, learning_rate=0.3)


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    starts = [(f"AL{k + 1:02d}{y}", pd.Timestamp(f"{y}-08-01") + pd.Timedelta(days=10 * k))
              for y in range(1978, 2024) for k in range(3)]
    starts.append(("AL042015", pd.Timestamp("2015-12-31")))   # genesis 2015, last rows on 2016-01-01
    rows = []
    for storm_id, start in starts:
        for i in range(8):
            x = rng.normal(size=len(FEATURES))
            rows.append({"storm_id": storm_id, "name": "TEST",
                         "datetime": start + pd.Timedelta(hours=6 * i),
                         **dict(zip(FEATURES, x)),
                         "RI": int(x[0] + 0.5 * rng.normal() > 1.2)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def splits():
    return xm.split_data(synthetic())


@pytest.fixture(scope="module")
def fitted(splits):
    train, cal, test = splits
    model, spw, cv_info = xm.train_model(train, params=SMALL)
    calibrator = xm.calibrate(model, cal)
    bands = xm.risk_bands(cal)
    result = xm.evaluate(model, test, calibrator, bands)
    return model, spw, cv_info, calibrator, bands, result


def years_of(splits, index):
    lookup = pd.concat(splits)["storm_year"]
    return set(lookup.loc[index])


@pytest.fixture
def spies(monkeypatch):
    calls = {"fit": [], "eval": [], "predict": [], "platt": []}
    fit, predict_proba, platt_fit = xgb.XGBClassifier.fit, xgb.XGBClassifier.predict_proba, LogisticRegression.fit

    def spy_fit(self, X, y, *args, **kwargs):
        calls["fit"].append(X.index)
        calls["eval"] += [e[0].index for e in kwargs.get("eval_set") or []]
        return fit(self, X, y, *args, **kwargs)

    def spy_predict(self, X, *args, **kwargs):
        calls["predict"].append(X.index)
        return predict_proba(self, X, *args, **kwargs)

    def spy_platt(self, X, y, *args, **kwargs):
        calls["platt"].append(len(X))
        return platt_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", spy_fit)
    monkeypatch.setattr(xgb.XGBClassifier, "predict_proba", spy_predict)
    monkeypatch.setattr(LogisticRegression, "fit", spy_platt)
    return calls


# ---------- split ----------

def test_split_years_and_disjoint(splits):
    train, cal, test = splits
    assert (settings.TRAIN_MIN_YEAR, settings.TRAIN_MAX_YEAR) == (1980, 2015)
    assert train["storm_year"].between(1980, 2015).all()
    assert cal["storm_year"].between(2016, 2019).all()
    assert (test["storm_year"] >= 2020).all()
    assert not pd.concat(splits)["storm_year"].isin([1978, 1979]).any()
    crosser = train[train["storm_id"] == "AL042015"]
    assert len(crosser) == 8 and crosser["datetime"].max().year == 2016
    ids = [set(s["storm_id"]) for s in splits]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_split_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        xm.split_data(synthetic(), train_max_year=2017)


# ---------- early stopping ----------

def test_season_folds(splits):
    train = splits[0]
    folds = xm.season_folds(train, settings.CV_FOLDS)
    assert settings.CV_FOLDS == 5 and len(folds) == 5
    held_seasons, held_storms = [], []
    for fit_idx, hold_idx in folds:
        fit_rows, hold_rows = train.loc[fit_idx], train.loc[hold_idx]
        assert not set(fit_rows["storm_year"]) & set(hold_rows["storm_year"])
        assert not set(fit_rows["storm_id"]) & set(hold_rows["storm_id"])
        assert pd.concat([fit_rows, hold_rows])["storm_year"].between(1980, 2015).all()
        held_seasons += sorted(set(hold_rows["storm_year"]))
        held_storms += sorted(set(hold_rows["storm_id"]))
    assert sorted(held_seasons) == sorted(set(train["storm_year"]))
    assert sorted(held_storms) == sorted(set(train["storm_id"]))


def test_early_stopping_uses_only_train_years(splits, spies):
    train, cal, test = splits
    xm.train_model(train, params=SMALL)
    assert len(spies["fit"]) == settings.CV_FOLDS + 1 and len(spies["eval"]) == settings.CV_FOLDS
    for index in spies["fit"] + spies["eval"]:
        assert years_of(splits, index) <= set(range(1980, 2016))
    with pytest.raises(ValueError, match="1980-2015"):
        xm.train_model(pd.concat([train, cal]), params=SMALL)


def test_rounds_are_median_and_refit_without_es(splits, fitted):
    train = splits[0]
    model, _, cv_info = fitted[:3]
    expected = []
    for fit_idx, hold_idx in xm.season_folds(train, settings.CV_FOLDS):
        fit_rows, hold_rows = train.loc[fit_idx], train.loc[hold_idx]
        n_pos = fit_rows["RI"].sum()
        fold = xgb.XGBClassifier(**SMALL, scale_pos_weight=(len(fit_rows) - n_pos) / n_pos, n_jobs=settings.N_JOBS)
        fold.fit(fit_rows[FEATURES], fit_rows["RI"], eval_set=[(hold_rows[FEATURES], hold_rows["RI"])], verbose=False)
        expected.append(fold.best_iteration + 1)
    assert cv_info["fold_rounds"] == expected
    assert cv_info["n_rounds"] == int(np.median(expected))
    assert model.get_params()["n_estimators"] == cv_info["n_rounds"]
    assert model.get_params()["early_stopping_rounds"] is None
    assert model.get_booster().num_boosted_rounds() == cv_info["n_rounds"]


# ---------- calibration, thresholds, bands ----------

def test_calibration_and_bands_fit_only_on_2016_19(splits, fitted, spies):
    train, cal, test = splits
    model = fitted[0]
    xm.calibrate(model, cal)
    assert spies["platt"] == [len(cal)]
    assert len(spies["predict"]) == 1 and years_of(splits, spies["predict"][0]) <= {2016, 2017, 2018, 2019}

    bands = xm.risk_bands(cal)
    assert bands["base_rate"] == pytest.approx(cal["RI"].mean())
    assert bands["edges"] == pytest.approx([m * cal["RI"].mean() for m in settings.BAND_MULTIPLIERS])
    assert settings.BAND_MULTIPLIERS == [1, 2] and bands["labels"] == settings.BAND_LABELS

    for wrong in (train, test, pd.concat([cal, test])):
        with pytest.raises(ValueError, match="2016-2019"):
            xm.calibrate(model, wrong)
        with pytest.raises(ValueError, match="2016-2019"):
            xm.risk_bands(wrong)


def test_test_rows_never_used_before_evaluation(splits, spies):
    train, cal, test = splits
    model, _, _ = xm.train_model(train, params=SMALL)
    xm.calibrate(model, cal)
    xm.risk_bands(cal)
    for index in spies["fit"] + spies["eval"] + spies["predict"]:
        assert not set(index) & set(test.index)
        assert max(years_of(splits, index)) < 2020
    assert spies["platt"] == [len(cal)]


# ---------- determinism ----------

def test_n_jobs_pinned_and_deterministic(splits, fitted):
    assert isinstance(settings.N_JOBS, int) and settings.N_JOBS >= 1
    model = fitted[0]
    assert model.get_params()["n_jobs"] == settings.N_JOBS
    again, _, _ = xm.train_model(splits[0], params=SMALL)
    X = splits[2][FEATURES]
    np.testing.assert_array_equal(model.predict_proba(X), again.predict_proba(X))


# ---------- bootstrap CI ----------

def test_bootstrap_ci_seeded_from_config(splits, fitted, monkeypatch):
    test, result = splits[2], fitted[5]
    monkeypatch.setattr(settings, "BOOTSTRAP_N", 300)
    monkeypatch.setattr(settings, "BOOTSTRAP_SEED", 7)
    monkeypatch.setattr(settings, "CI_LEVEL", 0.9)
    args = (test["storm_id"], test["RI"], result["y_prob_cal"])
    ci = xm.bootstrap_pr_auc_ci(*args)
    assert (ci["n_boot"], ci["seed"], ci["level"]) == (300, 7, 0.9)
    explicit = xm.bootstrap_pr_auc_ci(*args, n_boot=300, seed=7, level=0.9)
    assert (ci["lower"], ci["upper"]) == (explicit["lower"], explicit["upper"])
    np.testing.assert_array_equal(ci["samples"], explicit["samples"])
    other = xm.bootstrap_pr_auc_ci(*args, n_boot=300, seed=8, level=0.9)
    assert not np.array_equal(ci["samples"], other["samples"])
    assert ci["lower"] <= ci["point"] <= ci["upper"]


def test_bootstrap_resamples_whole_storms():
    # Rows within a storm are identical, so a storm bootstrap can only produce the
    # PR-AUC of a multiset of 3 storms: at most C(5, 3) = 10 distinct values
    storms = np.repeat(["A", "B", "C"], 5)
    y = np.repeat([1, 0, 1], 5)
    p = np.repeat([0.9, 0.5, 0.1], 5)
    ci = xm.bootstrap_pr_auc_ci(storms, y, p, n_boot=500, seed=0, level=0.95)
    assert 1 < len(np.unique(np.round(ci["samples"], 12))) <= 10


# ---------- metrics and saved predictions ----------

def test_evaluate_reports_raw_and_calibrated(splits, fitted):
    train, cal, test = splits
    model, spw, cv_info, calibrator, bands, result = fitted
    metrics = xm.build_metrics(train, cal, test, spw, cv_info, bands, result)
    json.dumps(metrics)
    for key in ("pr_auc_test_raw", "pr_auc_test_cal", "brier_test_raw", "brier_test_cal",
                "pr_auc_test_ci", "risk_bands", "thresholds_cal", "band_counts_test",
                "cv_folds", "fold_rounds", "n_rounds", "n_jobs", "pr_auc_persistence"):
        assert key in metrics, key
    assert metrics["split"]["train_min_year"] == 1980
    assert metrics["n_jobs"] == settings.N_JOBS and metrics["cv_folds"] == settings.CV_FOLDS
    assert set(metrics["thresholds_cal"]) == {f"{e:.6f}" for e in bands["edges"]}
    for row in metrics["thresholds_cal"].values():
        assert {"precision", "recall", "n_predicted_positive"} <= set(row)
    assert set(metrics["pr_auc_test_ci"]) >= {"point", "lower", "upper", "level", "n_boot", "seed"}
    assert sum(metrics["band_counts_test"].values()) == len(test)
    y_cal = calibrator.predict_proba(result["y_prob"].reshape(-1, 1))[:, 1]
    np.testing.assert_allclose(result["y_prob_cal"], y_cal)


def test_save_artifacts_persists_calibrated_predictions(splits, fitted, tmp_path):
    train, cal, test = splits
    model, spw, cv_info, calibrator, bands, result = fitted
    metrics = xm.build_metrics(train, cal, test, spw, cv_info, bands, result)
    art, res = tmp_path / "a", tmp_path / "r"
    xm.save_artifacts(model, calibrator, FEATURES, test, result, metrics, artifacts_dir=art, results_dir=res)
    written = pd.read_csv(art / settings.TEST_PREDICTIONS_FILE)
    np.testing.assert_allclose(written["y_prob"], result["y_prob"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(written["y_prob_cal"], result["y_prob_cal"], rtol=0, atol=1e-7)
    with open(res / settings.METRICS_FILE) as f:
        assert json.load(f)["n_rounds"] == cv_info["n_rounds"]
    assert {p.parent for p in tmp_path.rglob("*") if p.is_file()} <= {art, res}


def test_bootstrap_raises_when_too_few_usable_resamples(monkeypatch):
    # 1 RI storm out of 20: about (19/20)^20 = 36% of resamples have no RI case
    storms = np.repeat([f"S{i}" for i in range(20)], 3)
    y = np.repeat([1] + [0] * 19, 3)
    p = np.linspace(0, 1, len(y))
    assert settings.BOOTSTRAP_MIN_VALID == 0.95
    with pytest.raises(ValueError, match="usable"):
        xm.bootstrap_pr_auc_ci(storms, y, p, n_boot=300, seed=0, level=0.95)
    monkeypatch.setattr(settings, "BOOTSTRAP_MIN_VALID", 0.5)   # threshold is read from config
    ci = xm.bootstrap_pr_auc_ci(storms, y, p, n_boot=300, seed=0, level=0.95)
    assert 0.5 * 300 <= ci["n_valid"] < 0.95 * 300


def test_band_rule_from_config_records_both_rates(splits, fitted, monkeypatch):
    train, cal, test = splits
    model, spw, cv_info, calibrator, _, result = fitted
    cal_prob = xm.predict(cal[FEATURES], model, calibrator)
    observed, mean_cal = cal["RI"].mean(), cal_prob.mean()

    assert settings.BAND_BASE_RATE == "observed"
    bands = xm.risk_bands(cal, cal_prob)
    assert bands["base_rate_rule"] == "observed"
    assert bands["observed_rate"] == pytest.approx(observed)
    assert bands["mean_calibrated_prob"] == pytest.approx(mean_cal)
    assert bands["edges"] == pytest.approx([m * observed for m in settings.BAND_MULTIPLIERS])

    metrics = xm.build_metrics(train, cal, test, spw, cv_info, bands, result)
    saved = json.loads(json.dumps(metrics))["risk_bands"]
    assert saved["observed_rate"] == pytest.approx(observed)
    assert saved["mean_calibrated_prob"] == pytest.approx(mean_cal)

    monkeypatch.setattr(settings, "BAND_BASE_RATE", "mean_calibrated")
    switched = xm.risk_bands(cal, cal_prob)
    assert switched["edges"] == pytest.approx([m * mean_cal for m in settings.BAND_MULTIPLIERS])
    with pytest.raises(ValueError, match="mean_calibrated"):
        xm.risk_bands(cal)   # that rule needs the calibrated probabilities
    monkeypatch.setattr(settings, "BAND_BASE_RATE", "median")
    with pytest.raises(ValueError, match="BAND_BASE_RATE"):
        xm.risk_bands(cal, cal_prob)
