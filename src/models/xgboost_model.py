import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score, brier_score_loss
import joblib
from pathlib import Path
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import sys
import hashlib
import platform
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import settings
from explainability.compute_shap import explain

def plot_calibration(y_true, y_prob, path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve (test)")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_data(path=settings.PROCESSED_DATA_PATH):
    # Load the processed data
    print("Loading data...")
    return pd.read_parquet(path)


def split_data(
    df,
    train_min_year=settings.TRAIN_MIN_YEAR,
    train_max_year=settings.TRAIN_MAX_YEAR,
    val_min_year=settings.VAL_MIN_YEAR,
    val_max_year=settings.VAL_MAX_YEAR,
    test_min_year=settings.TEST_MIN_YEAR,
):
    # Create a temporal split
    # One year per storm = year of first observation (genesis)
    storm_year = df.groupby("storm_id")["datetime"].min().dt.year
    df = df.assign(storm_year=df["storm_id"].map(storm_year))

    print("Creating temporal split...")
    training_set = df[df["storm_year"].between(train_min_year, train_max_year)]
    validation_set = df[df["storm_year"].between(val_min_year, val_max_year)]
    test_set = df[df["storm_year"] >= test_min_year]

    # No storm may appear in two sets
    train_ids = set(training_set["storm_id"])
    val_ids   = set(validation_set["storm_id"])
    test_ids  = set(test_set["storm_id"])
    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"storms overlap between train / calibration / test: {sorted(overlap)}")

    print(f"Training: {len(training_set)} | Calibration: {len(validation_set)} | Test: {len(test_set)}")
    print("Training RI rate:", training_set["RI"].mean().round(3))
    print("Calibration RI rate:", validation_set["RI"].mean().round(3))
    print("Test RI rate:", test_set["RI"].mean().round(3))

    return training_set, validation_set, test_set


def check_years(df, min_year, max_year, what):
    # Raise if any row's storm year is outside [min_year, max_year]
    years = df["storm_year"]
    if not years.between(min_year, max_year).all():
        raise ValueError(
            f"{what} must use storms from {min_year}-{max_year} only, got years {years.min()}-{years.max()}"
        )


def season_folds(training_set, n_folds=settings.CV_FOLDS):
    # CV folds grouped by season, so no season or storm is on both sides of a fold
    folds = GroupKFold(n_splits=n_folds).split(training_set, groups=training_set["storm_year"])
    return [(training_set.index[fit], training_set.index[hold]) for fit, hold in folds]


def class_weight(y):
    # Scaling factor for class imbalance: negatives / positives
    n_pos = y.sum()
    return (len(y) - n_pos) / n_pos


def train_model(
    training_set,
    feature_cols=settings.FEATURE_COLS,
    params=settings.XGB_PARAMS,
    n_jobs=settings.N_JOBS,
    n_folds=settings.CV_FOLDS,
):
    check_years(training_set, settings.TRAIN_MIN_YEAR, settings.TRAIN_MAX_YEAR, "Training")

    # Early stopping on season-grouped folds inside the train years
    print("Choosing the number of rounds by season-grouped CV...")
    fold_rounds = []
    for fit_idx, hold_idx in season_folds(training_set, n_folds):
        fit_rows, hold_rows = training_set.loc[fit_idx], training_set.loc[hold_idx]
        fold = xgb.XGBClassifier(**params, scale_pos_weight=class_weight(fit_rows["RI"]), n_jobs=n_jobs)
        fold.fit(
            fit_rows[feature_cols], fit_rows["RI"],
            eval_set=[(hold_rows[feature_cols], hold_rows["RI"])],
            verbose=False,
        )
        fold_rounds.append(fold.best_iteration + 1)
    n_rounds = int(np.median(fold_rounds))
    print(f"Fold rounds {fold_rounds} -> median {n_rounds}")

    # Refit on all train years with that fixed number of rounds, no early stopping
    scale_pos_weight = class_weight(training_set["RI"])
    print(f"scale_pos_weight = {scale_pos_weight:.1f}")
    final_params = dict(params, n_estimators=n_rounds, early_stopping_rounds=None)
    model = xgb.XGBClassifier(**final_params, scale_pos_weight=scale_pos_weight, n_jobs=n_jobs)
    model.fit(training_set[feature_cols], training_set["RI"], verbose=False)

    cv_info = {"cv_folds": n_folds, "fold_rounds": fold_rounds, "n_rounds": n_rounds}
    return model, scale_pos_weight, cv_info


def calibrate(model, validation_set, feature_cols=settings.FEATURE_COLS):
    # Platt scaling, fitted on the calibration years only
    check_years(validation_set, settings.VAL_MIN_YEAR, settings.VAL_MAX_YEAR, "Calibration")
    validation_prob = model.predict_proba(validation_set[feature_cols])[:, 1].reshape(-1, 1)

    platt = LogisticRegression()
    platt.fit(validation_prob, validation_set["RI"])
    return platt


def risk_bands(validation_set, y_prob_cal=None):
    # Band edges are multiples of a calibration-set base rate (calibrated probability scale).
    # y_prob_cal: calibrated P(RI) on the same rows, needed for the "mean_calibrated" rule.
    check_years(validation_set, settings.VAL_MIN_YEAR, settings.VAL_MAX_YEAR, "Risk bands")
    observed_rate = float(validation_set["RI"].mean())
    mean_calibrated = None if y_prob_cal is None else float(np.mean(y_prob_cal))

    rule = settings.BAND_BASE_RATE
    if rule == "observed":
        base_rate = observed_rate
    elif rule == "mean_calibrated":
        if mean_calibrated is None:
            raise ValueError("BAND_BASE_RATE 'mean_calibrated' needs the calibrated probabilities (y_prob_cal)")
        base_rate = mean_calibrated
    else:
        raise ValueError(f"Unknown BAND_BASE_RATE {rule!r}; use 'observed' or 'mean_calibrated'")

    return {
        "base_rate_rule": rule,
        "base_rate": base_rate,
        "observed_rate": observed_rate,
        "mean_calibrated_prob": mean_calibrated,
        "multipliers": list(settings.BAND_MULTIPLIERS),
        "edges": [m * base_rate for m in settings.BAND_MULTIPLIERS],
        "labels": list(settings.BAND_LABELS),
    }


def storm_resamples(storm_ids, n_boot, seed):
    # Row indices of n_boot resamples of whole storms (rows within a storm are correlated)
    storms, codes = np.unique(np.asarray(storm_ids), return_inverse=True)
    rows_of = [np.flatnonzero(codes == i) for i in range(len(storms))]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([rows_of[i] for i in rng.integers(0, len(storms), len(storms))])
        for _ in range(n_boot)
    ]


def paired_bootstrap(storm_ids, y_true, stats, n_boot=None, seed=None, level=None):
    # Percentile CIs for several statistics on the same storm resamples, so differences are paired.
    # stats: {name: function of row indices -> float}. Defaults are read from config at call time.
    # Resamples with no positive are skipped.
    n_boot = settings.BOOTSTRAP_N if n_boot is None else n_boot
    seed = settings.BOOTSTRAP_SEED if seed is None else seed
    level = settings.CI_LEVEL if level is None else level

    y_true = np.asarray(y_true)
    usable = [rows for rows in storm_resamples(storm_ids, n_boot, seed) if y_true[rows].any()]
    if len(usable) < settings.BOOTSTRAP_MIN_VALID * n_boot:
        raise ValueError(
            f"Only {len(usable)} of {n_boot} bootstrap resamples are usable (contain an RI case); "
            f"need at least {settings.BOOTSTRAP_MIN_VALID:.0%}"
        )

    alpha = (1 - level) / 2
    all_rows = np.arange(len(y_true))
    results = {}
    for name, stat in stats.items():
        samples = np.array([stat(rows) for rows in usable])
        lower, upper = np.quantile(samples, [alpha, 1 - alpha])
        results[name] = {"point": float(stat(all_rows)), "lower": float(lower), "upper": float(upper),
                         "samples": samples}
    return {"bootstrap": {"level": level, "n_boot": n_boot, "n_valid": len(usable), "seed": seed},
            "stats": results}


def bootstrap_pr_auc_ci(storm_ids, y_true, y_prob, n_boot=None, seed=None, level=None):
    # Percentile CI for PR-AUC, resampling whole storms
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    boot = paired_bootstrap(
        storm_ids, y_true, {"pr_auc": lambda rows: average_precision_score(y_true[rows], y_prob[rows])},
        n_boot, seed, level,
    )
    return {**boot["stats"]["pr_auc"], **boot["bootstrap"]}


def baseline_comparison(storm_ids, y_true, y_prob, persistence, climatology_rate,
                        n_boot=None, seed=None, level=None):
    # The model vs persistence and climatology on the same rows and the same storm resamples
    # (AUDIT.md §8 session 6 evaluation plan). climatology_rate is the 2016-19 observed RI
    # rate, not the test rate, so nothing is tuned on test.
    y, p, persist = np.asarray(y_true), np.asarray(y_prob), np.asarray(persistence)
    clim = np.full(len(y), float(climatology_rate))
    ap = average_precision_score

    def brier_skill(rows):
        return 1 - brier_score_loss(y[rows], p[rows]) / brier_score_loss(y[rows], clim[rows])

    boot = paired_bootstrap(storm_ids, y, {
        "model": lambda rows: ap(y[rows], p[rows]),
        "persistence": lambda rows: ap(y[rows], persist[rows]),
        "climatology": lambda rows: ap(y[rows], clim[rows]),
        "model_minus_persistence": lambda rows: ap(y[rows], p[rows]) - ap(y[rows], persist[rows]),
        "brier_skill_score": brier_skill,
    }, n_boot, seed, level)
    stats = boot["stats"]
    return {
        "n_rows": int(len(y)),
        "n_pos": int(y.sum()),
        "climatology_rate": float(climatology_rate),
        "bootstrap": boot["bootstrap"],
        "pr_auc": {k: stats[k] for k in ("model", "persistence", "climatology", "model_minus_persistence")},
        "brier": {"model": float(brier_score_loss(y, p)), "climatology": float(brier_score_loss(y, clim))},
        "brier_skill_score": stats["brier_skill_score"],
    }


def evaluate(model, test_set, calibrator, bands, feature_cols=settings.FEATURE_COLS):
    X_test, y_test = test_set[feature_cols], test_set["RI"]

    # Evaluate performance on raw and calibrated probabilities
    print("Evaluate model...")
    # float32 -> float64 is exact and float64 round-trips through the CSV, so the saved
    # predictions reproduce every metric exactly (predict() calibrates the same float64 values)
    y_prob = model.predict_proba(X_test)[:, 1].astype(np.float64)
    y_prob_cal = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    pr_auc_cal = average_precision_score(y_test, y_prob_cal)
    brier = brier_score_loss(y_test, y_prob)
    brier_cal = brier_score_loss(y_test, y_prob_cal)
    print(f"Test PR-AUC raw / calibrated: {pr_auc:.4f} / {pr_auc_cal:.4f}")
    print(f"Test Brier raw / calibrated: {brier:.4f} / {brier_cal:.4f}")

    pr_auc_ci = bootstrap_pr_auc_ci(test_set["storm_id"], y_test, y_prob_cal)
    print(f"Test PR-AUC {pr_auc_ci['level']:.0%} CI: [{pr_auc_ci['lower']:.4f}, {pr_auc_ci['upper']:.4f}]")

    # Thresholds are the band edges, fitted on the calibration years, applied to calibrated P(RI)
    threshold_metrics = {}
    for threshold in bands["edges"]:
        y_pred = (y_prob_cal >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall  = recall_score(y_test, y_pred, zero_division=0)

        threshold_metrics[f"{threshold:.6f}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "n_predicted_positive": int(y_pred.sum()),
        }
        print(f"Threshold {threshold:.4f} -> Precision {precision:.3f} | Recall {recall:.3f}")

    band_index = np.searchsorted(bands["edges"], y_prob_cal, side="right")
    band_counts = {label: int((band_index == i).sum()) for i, label in enumerate(bands["labels"])}

    # Compare against trivial baselines

    # Baseline A: always predict the training prevalence (or all zeros)
    pr_auc_majority = average_precision_score(y_test, np.zeros_like(y_test, dtype=float))

    # Baseline B: persistence
    # If the storm intensified over the last 12 h, predict RI
    persist_score = (X_test["delta_vmax_12h"] > 0).astype(float)
    pr_auc_persist = average_precision_score(y_test, persist_score)

    print(f"Majority / all-zero PR-AUC: {pr_auc_majority:.4f}")
    print(f"Persistence PR-AUC: {pr_auc_persist:.4f}")

    # Model vs persistence and climatology, paired on the same storm resamples
    baselines = baseline_comparison(test_set["storm_id"], y_test, y_prob_cal, persist_score, bands["observed_rate"])
    diff, bss = baselines["pr_auc"]["model_minus_persistence"], baselines["brier_skill_score"]
    print(f"PR-AUC model - persistence: {diff['point']:.4f} [{diff['lower']:.4f}, {diff['upper']:.4f}]")
    print(f"Brier skill score vs climatology: {bss['point']:.4f} [{bss['lower']:.4f}, {bss['upper']:.4f}]")

    return {
        "y_prob": y_prob,
        "y_prob_cal": y_prob_cal,
        "pr_auc": pr_auc,
        "pr_auc_cal": pr_auc_cal,
        "brier": brier,
        "brier_cal": brier_cal,
        "pr_auc_ci": pr_auc_ci,
        "thresholds_cal": threshold_metrics,
        "band_counts": band_counts,
        "pr_auc_majority": pr_auc_majority,
        "pr_auc_persistence": pr_auc_persist,
        "baselines": baselines,
    }


def evaluate_calibration_years(model, calibrator, validation_set, feature_cols=settings.FEATURE_COLS):
    # Secondary results on 2016-19. These rows fitted Platt and the bands, so they are optimistic.
    check_years(validation_set, settings.VAL_MIN_YEAR, settings.VAL_MAX_YEAR, "Calibration-year results")
    y = validation_set["RI"]
    y_prob = model.predict_proba(validation_set[feature_cols])[:, 1].astype(np.float64)
    y_prob_cal = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    return {
        "note": "used for calibration and band-setting: optimistic",
        "years": f"{settings.VAL_MIN_YEAR}-{settings.VAL_MAX_YEAR}",
        "n_rows": int(len(y)),
        "n_pos": int(y.sum()),
        "pr_auc_raw": float(average_precision_score(y, y_prob)),
        "pr_auc_cal": float(average_precision_score(y, y_prob_cal)),
        "brier_raw": float(brier_score_loss(y, y_prob)),
        "brier_cal": float(brier_score_loss(y, y_prob_cal)),
    }


def without_samples(ci):
    return {k: v for k, v in ci.items() if k != "samples"}


def build_metrics(training_set, validation_set, test_set, scale_pos_weight, cv_info, bands, eval_result,
                  feature_cols=settings.FEATURE_COLS, calibration_years=None):
    ci = without_samples(eval_result["pr_auc_ci"])
    base = eval_result["baselines"]
    baselines = {
        **base,
        "pr_auc": {name: without_samples(v) for name, v in base["pr_auc"].items()},
        "brier_skill_score": without_samples(base["brier_skill_score"]),
    }
    return {
        "model": settings.MODEL_NAME,
        "split": {
            "train_min_year": settings.TRAIN_MIN_YEAR,
            "train_max_year": settings.TRAIN_MAX_YEAR,
            "val_years": f"{settings.VAL_MIN_YEAR}-{settings.VAL_MAX_YEAR}",
            "test_min_year": settings.TEST_MIN_YEAR,
        },
        "n_train": int(len(training_set)),
        "n_val": int(len(validation_set)),
        "n_test": int(len(test_set)),
        "ri_rate_train": float(training_set["RI"].mean()),
        "ri_rate_val": float(validation_set["RI"].mean()),
        "ri_rate_test": float(test_set["RI"].mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "cv_folds": cv_info["cv_folds"],
        "fold_rounds": cv_info["fold_rounds"],
        "n_rounds": cv_info["n_rounds"],
        "n_jobs": settings.N_JOBS,
        "pr_auc_test_raw": float(eval_result["pr_auc"]),
        "pr_auc_test_cal": float(eval_result["pr_auc_cal"]),
        "pr_auc_test_ci": ci,
        "brier_test_raw": float(eval_result["brier"]),
        "brier_test_cal": float(eval_result["brier_cal"]),
        "risk_bands": bands,
        "thresholds_cal": eval_result["thresholds_cal"],
        "band_counts_test": eval_result["band_counts"],
        "pr_auc_majority": float(eval_result["pr_auc_majority"]),
        "pr_auc_persistence": float(eval_result["pr_auc_persistence"]),
        "baselines": baselines,
        "calibration_years": calibration_years,
        "features": feature_cols,
    }


def save_artifacts(
    model,
    calibrator,
    feature_cols,
    test_set,
    eval_result,
    metrics,
    artifacts_dir=settings.ARTIFACTS_DIR,
    results_dir=settings.RESULTS_DIR,
    data_path=settings.PROCESSED_DATA_PATH,
):
    artifacts_dir, results_dir, data_path = Path(artifacts_dir), Path(results_dir), Path(data_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save raw and calibrated predictions
    pred_df = test_set[["storm_id", "name", "datetime", "RI"]].copy()
    pred_df["y_prob"] = eval_result["y_prob"]
    pred_df["y_prob_cal"] = eval_result["y_prob_cal"]
    pred_df.to_csv(artifacts_dir / settings.TEST_PREDICTIONS_FILE, index=False)

    # Save model, feature list and Platt calibrator
    print("Saving model...")
    joblib.dump(model, artifacts_dir / settings.MODEL_FILE)
    joblib.dump(feature_cols, artifacts_dir / settings.FEATURE_COLS_FILE)
    joblib.dump(calibrator, artifacts_dir / settings.CALIBRATOR_FILE)
    print("Model, feature list and calibrator saved...")

    # Risk bands, thresholds and provenance next to the model, for the app
    meta = {
        "model_version": settings.MODEL_VERSION,
        "model_file": settings.MODEL_FILE,
        "calibrator_file": settings.CALIBRATOR_FILE,
        "scope": "P(RI: >=30 kt increase in 24 h | the storm stays a tropical cyclone over water for the next 24 h)",
        "feature_cols": list(feature_cols),
        "risk_bands": metrics["risk_bands"],
        "thresholds_cal": metrics["risk_bands"]["edges"],
        "calibration_years": metrics["split"]["val_years"],
        "n_rounds": metrics["n_rounds"],
        "n_jobs": metrics["n_jobs"],
        "library_versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest() if data_path.exists() else None,
    }
    with open(artifacts_dir / settings.MODEL_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    # Plot and save calibration curve
    plot_calibration(test_set["RI"], eval_result["y_prob_cal"], artifacts_dir / settings.CALIBRATION_PLOT_FILE)

    # Look at feature importance
    feature_importance = (
        pd.Series(model.feature_importances_, index=feature_cols, name="importance")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    print("Feature Importance: ", feature_importance)
    feature_importance.to_csv(artifacts_dir / settings.FEATURE_IMPORTANCE_FILE, index=False)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / settings.METRICS_FILE
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path}")


def load_model(artifacts_dir=settings.ARTIFACTS_DIR):
    artifacts_dir = Path(artifacts_dir)
    model = joblib.load(artifacts_dir / settings.MODEL_FILE)
    calibrator = joblib.load(artifacts_dir / settings.CALIBRATOR_FILE)
    feature_cols = joblib.load(artifacts_dir / settings.FEATURE_COLS_FILE)

    # The saved list and the model's own names must equal the config list, in order
    booster_cols = model.get_booster().feature_names
    if feature_cols != settings.FEATURE_COLS or booster_cols != settings.FEATURE_COLS:
        raise ValueError(
            f"Saved feature list {feature_cols} / model feature names {booster_cols} "
            f"do not match config FEATURE_COLS {settings.FEATURE_COLS}"
        )
    return model, calibrator, feature_cols


def load_meta(artifacts_dir=settings.ARTIFACTS_DIR):
    # Risk bands, thresholds and provenance saved with the model
    with open(Path(artifacts_dir) / settings.MODEL_META_FILE) as f:
        meta = json.load(f)
    if meta["model_version"] != settings.MODEL_VERSION:
        raise ValueError(f"Model meta is for {meta['model_version']}, config is {settings.MODEL_VERSION}")
    if meta["feature_cols"] != settings.FEATURE_COLS:
        raise ValueError(
            f"Model meta feature list {meta['feature_cols']} does not match config FEATURE_COLS {settings.FEATURE_COLS}"
        )
    return meta


def validate_features(X, feature_cols=settings.FEATURE_COLS):
    # X must be a DataFrame with exactly feature_cols, in that order, all numeric.
    # A numpy array skips xgboost's name check and is silently mis-ordered, so reject it.
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame with columns {feature_cols}, got {type(X).__name__}")

    columns = list(X.columns)
    missing = [c for c in feature_cols if c not in columns]
    if missing:
        raise ValueError(f"X is missing feature columns {missing}; expected {feature_cols}")
    extra = [c for c in columns if c not in feature_cols]
    if extra:
        raise ValueError(f"X has unexpected columns {extra}; expected {feature_cols}")
    if columns != list(feature_cols):
        raise ValueError(f"X columns are in the wrong order {columns}; expected order {feature_cols}")

    # Booleans count as numeric to pandas but no feature is boolean
    non_numeric = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
    ]
    if non_numeric:
        raise TypeError(f"X has non-numeric feature columns {non_numeric}: {X[non_numeric].dtypes.to_dict()}")

    # The model was trained on complete rows only; rows without enough history
    # (e.g. no 6 h / 12 h lag) must not be scored
    nan_cols = [c for c in feature_cols if X[c].isna().any()]
    if nan_cols:
        n_rows = int(X[nan_cols].isna().any(axis=1).sum())
        raise ValueError(f"X has NaN in feature columns {nan_cols} in {n_rows} rows; no forecast for those rows")


def predict(X, model=None, calibrator=None):
    # Returns calibrated P(RI). Loads the saved model and calibrator if not given.
    validate_features(X)
    if model is None or calibrator is None:
        saved_model, saved_calibrator, _ = load_model()
        model = saved_model if model is None else model
        calibrator = saved_calibrator if calibrator is None else calibrator
    raw = model.predict_proba(X)[:, 1].astype(np.float64)
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def main():
    df = load_data()
    training_set, validation_set, test_set = split_data(df)
    model, scale_pos_weight, cv_info = train_model(training_set)
    calibrator = calibrate(model, validation_set)
    bands = risk_bands(validation_set, predict(validation_set[settings.FEATURE_COLS], model, calibrator))
    calibration_years = evaluate_calibration_years(model, calibrator, validation_set)
    # Test is used once, here
    eval_result = evaluate(model, test_set, calibrator, bands)
    metrics = build_metrics(training_set, validation_set, test_set, scale_pos_weight, cv_info, bands, eval_result,
                            calibration_years=calibration_years)
    save_artifacts(model, calibrator, settings.FEATURE_COLS, test_set, eval_result, metrics)

    # Compute SHAP
    explain(model, test_set[settings.FEATURE_COLS], settings.FEATURE_COLS)


if __name__ == "__main__":
    main()
