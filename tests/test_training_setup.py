import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

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
    # A smaller config bootstrap keeps the fast test loop fast. It stays in effect while the
    # fixture is alive, so tests that compare against settings.BOOTSTRAP_N see the value used.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "BOOTSTRAP_N", 200)
        train, cal, test = splits
        model, spw, cv_info = xm.train_model(train, params=SMALL)
        calibrator = xm.calibrate(model, cal)
        bands = xm.risk_bands(cal)
        result = xm.evaluate(model, test, calibrator, bands)
        yield model, spw, cv_info, calibrator, bands, result


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


# ---------- baselines on the same resamples (AUDIT.md §8 session 6 evaluation plan) ----------

@pytest.mark.slow
def test_baseline_comparison_uses_the_same_resamples(splits, fitted):
    test, (bands, result) = splits[2], fitted[4:]
    y, p = test["RI"].to_numpy(), result["y_prob_cal"]
    persist = (test["delta_vmax_12h"] > 0).astype(float).to_numpy()
    cmp = xm.baseline_comparison(test["storm_id"], y, p, persist, bands["observed_rate"],
                                 n_boot=300, seed=3, level=0.9)
    auc = cmp["pr_auc"]
    assert cmp["bootstrap"] == {"level": 0.9, "n_boot": 300, "n_valid": len(auc["model"]["samples"]), "seed": 3}
    assert (cmp["n_rows"], cmp["n_pos"]) == (len(y), int(y.sum()))

    # The difference is taken resample by resample, so the resamples are shared
    np.testing.assert_array_equal(auc["model_minus_persistence"]["samples"],
                                  auc["model"]["samples"] - auc["persistence"]["samples"])
    # Same resamples as the headline CI
    ci = xm.bootstrap_pr_auc_ci(test["storm_id"], y, p, n_boot=300, seed=3, level=0.9)
    np.testing.assert_array_equal(auc["model"]["samples"], ci["samples"])
    assert (auc["model"]["lower"], auc["model"]["upper"]) == (ci["lower"], ci["upper"])

    # Climatology is a constant score: its PR-AUC is the positive rate of each resample
    usable = [rows for rows in xm.storm_resamples(test["storm_id"], 300, 3) if y[rows].any()]
    np.testing.assert_allclose(auc["climatology"]["samples"], [y[rows].mean() for rows in usable])
    assert auc["climatology"]["point"] == pytest.approx(y.mean())
    assert auc["persistence"]["point"] == pytest.approx(average_precision_score(y, persist))


@pytest.mark.slow
def test_brier_skill_uses_calibration_year_rate_not_test_rate(splits, fitted):
    test, (bands, result) = splits[2], fitted[4:]
    y, p = test["RI"].to_numpy(), result["y_prob_cal"]
    persist = (test["delta_vmax_12h"] > 0).astype(float).to_numpy()
    rate = bands["observed_rate"]
    assert rate != pytest.approx(y.mean())   # the synthetic 2016-19 and test rates differ
    cmp = xm.baseline_comparison(test["storm_id"], y, p, persist, rate, n_boot=300, seed=3, level=0.9)
    brier_clim = brier_score_loss(y, np.full(len(y), rate))
    assert cmp["climatology_rate"] == rate
    assert cmp["brier"]["model"] == pytest.approx(brier_score_loss(y, p))
    assert cmp["brier"]["climatology"] == pytest.approx(brier_clim)
    assert cmp["brier_skill_score"]["point"] == pytest.approx(1 - brier_score_loss(y, p) / brier_clim)
    bss = cmp["brier_skill_score"]
    assert bss["lower"] <= bss["upper"] and len(bss["samples"]) == cmp["bootstrap"]["n_valid"]


def test_evaluate_reports_baselines_with_config_bootstrap(splits, fitted):
    test, (bands, result) = splits[2], fitted[4:]
    base = result["baselines"]
    assert base["climatology_rate"] == bands["observed_rate"]
    assert base["bootstrap"]["seed"] == settings.BOOTSTRAP_SEED
    assert base["bootstrap"]["n_boot"] == settings.BOOTSTRAP_N
    assert base["pr_auc"]["model"]["lower"] == result["pr_auc_ci"]["lower"]
    assert base["pr_auc"]["model"]["upper"] == result["pr_auc_ci"]["upper"]
    assert base["pr_auc"]["persistence"]["point"] == pytest.approx(result["pr_auc_persistence"])


def test_metrics_have_baselines_and_optimistic_calibration_years(splits, fitted):
    train, cal, test = splits
    model, spw, cv_info, calibrator, bands, result = fitted
    cal_years = xm.evaluate_calibration_years(model, calibrator, cal)
    metrics = xm.build_metrics(train, cal, test, spw, cv_info, bands, result, calibration_years=cal_years)
    saved = json.loads(json.dumps(metrics))

    assert set(saved["baselines"]["pr_auc"]) == {"model", "persistence", "climatology", "model_minus_persistence"}
    assert "samples" not in json.dumps(saved)
    assert "brier_skill_score" in saved["baselines"]

    block = saved["calibration_years"]
    assert block["note"] == "used for calibration and band-setting: optimistic"
    assert block["years"] == "2016-2019" and block["n_rows"] == len(cal)
    raw = model.predict_proba(cal[FEATURES])[:, 1]
    cal_prob = xm.predict(cal[FEATURES], model, calibrator)
    assert block["pr_auc_raw"] == pytest.approx(average_precision_score(cal["RI"], raw))
    assert block["pr_auc_cal"] == pytest.approx(average_precision_score(cal["RI"], cal_prob))
    assert block["brier_cal"] == pytest.approx(brier_score_loss(cal["RI"], cal_prob))
    with pytest.raises(ValueError, match="2016-2019"):
        xm.evaluate_calibration_years(model, calibrator, test)


# ---------- model meta saved next to the model (AUDIT.md §8 follow-up for session 6) ----------

def test_save_artifacts_writes_model_meta(splits, fitted, tmp_path):
    train, cal, test = splits
    model, spw, cv_info, calibrator, bands, result = fitted
    metrics = xm.build_metrics(train, cal, test, spw, cv_info, bands, result)
    data = tmp_path / "data.parquet"
    data.write_bytes(b"some bytes")
    art, res = tmp_path / "a", tmp_path / "r"
    xm.save_artifacts(model, calibrator, FEATURES, test, result, metrics,
                      artifacts_dir=art, results_dir=res, data_path=data)

    assert (art / settings.MODEL_META_FILE).exists()
    meta = xm.load_meta(art)
    assert meta["model_version"] == settings.MODEL_VERSION
    assert meta["feature_cols"] == FEATURES
    assert meta["risk_bands"]["edges"] == pytest.approx(bands["edges"])
    assert meta["risk_bands"]["labels"] == bands["labels"]
    assert meta["risk_bands"]["base_rate"] == pytest.approx(bands["base_rate"])
    assert meta["thresholds_cal"] == pytest.approx(bands["edges"])
    assert (meta["n_rounds"], meta["n_jobs"]) == (cv_info["n_rounds"], settings.N_JOBS)
    assert meta["calibration_years"] == "2016-2019"
    assert meta["data_sha256"] == hashlib.sha256(b"some bytes").hexdigest()
    assert meta["library_versions"]["xgboost"] == xgb.__version__

    meta["feature_cols"] = FEATURES[::-1]
    (art / settings.MODEL_META_FILE).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="feature"):
        xm.load_meta(art)


def test_saved_predictions_reproduce_brier_exactly(splits, fitted, tmp_path):
    # Metrics are computed on float64 probabilities, which round-trip through the CSV exactly,
    # so the golden test can recompute Brier from the saved predictions (float32 did not).
    train, cal, test = splits
    model, spw, cv_info, calibrator, bands, result = fitted
    assert result["y_prob"].dtype == np.float64 and result["y_prob_cal"].dtype == np.float64
    metrics = xm.build_metrics(train, cal, test, spw, cv_info, bands, result)
    art, res = tmp_path / "a", tmp_path / "r"
    xm.save_artifacts(model, calibrator, FEATURES, test, result, metrics, artifacts_dir=art, results_dir=res)
    # pandas' default CSV float parser can be 1 ulp off; round_trip reads exactly what was written
    written = pd.read_csv(art / settings.TEST_PREDICTIONS_FILE, float_precision="round_trip")
    np.testing.assert_array_equal(written["y_prob"], result["y_prob"])
    np.testing.assert_array_equal(written["y_prob_cal"], result["y_prob_cal"])
    assert brier_score_loss(written["RI"], written["y_prob"]) == result["brier"]
    assert brier_score_loss(written["RI"], written["y_prob_cal"]) == result["brier_cal"]


def test_predict_matches_saved_calibrated_predictions_exactly(splits, fitted):
    # predict() and evaluate() calibrate the same float64 raw probabilities
    test = splits[2]
    model, calibrator, result = fitted[0], fitted[3], fitted[5]
    p = xm.predict(test[FEATURES], model, calibrator)
    assert p.dtype == np.float64
    np.testing.assert_array_equal(p, result["y_prob_cal"])
