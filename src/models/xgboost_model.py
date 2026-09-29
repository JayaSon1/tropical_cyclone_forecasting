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
from sklearn.linear_model import LogisticRegression

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
    train_max_year=settings.TRAIN_MAX_YEAR,
    val_min_year=settings.VAL_MIN_YEAR,
    val_max_year=settings.VAL_MAX_YEAR,
    test_min_year=settings.TEST_MIN_YEAR,
):
    # Create a temporal split
    # One year per storm = year of first observation (genesis)
    storm_year = (
        df.groupby("storm_id")["datetime"]
        .min()
        .dt.year
        .rename("storm_year")
    )

    df = df.merge(storm_year, left_on="storm_id", right_index=True)

    print("Creating temporal split...")
    training_set = df[df["storm_year"] <= train_max_year]
    validation_set = df[(df["storm_year"] >= val_min_year) & (df["storm_year"] <= val_max_year)]
    test_set = df[df["storm_year"] >= test_min_year]

    # Check that there are no storms that overlap in the training, validation and testing sets
    train_ids = set(training_set["storm_id"])
    val_ids   = set(validation_set["storm_id"])
    test_ids  = set(test_set["storm_id"])

    print("Train and Val :", len(train_ids & val_ids))   # must be 0
    print("Train and Test:", len(train_ids & test_ids))  # must be 0
    print("Val and Test  :", len(val_ids & test_ids))    # must be 0

    print(f"Training: {len(training_set)} | Validation: {len(validation_set)} | Test: {len(test_set)}")
    print("Training RI rate:", training_set["RI"].mean().round(3))
    print("Validation RI rate:", validation_set["RI"].mean().round(3))
    print("Test RI rate:", test_set["RI"].mean().round(3))

    return training_set, validation_set, test_set


def train_model(
    training_set,
    validation_set,
    feature_cols=settings.FEATURE_COLS,
    params=settings.XGB_PARAMS,
    n_jobs=settings.N_JOBS,
):
    X_train, y_train = training_set[feature_cols], training_set["RI"]
    X_validation, y_validation = validation_set[feature_cols], validation_set["RI"]

    # Calculate scaling factor for class imbalance
    print("Computing class imbalance scaling factor...")
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight = {scale_pos_weight:.1f}")

    # Train the model
    print("Training model...")
    model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight, n_jobs=n_jobs)

    print("Fitting model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=50,
    )

    return model, scale_pos_weight


def calibrate(model, validation_set, feature_cols=settings.FEATURE_COLS):
    # The imbalance scale factor usually pushes scores in a smooth way so Platt scaling can fix this
    validation_prob = model.predict_proba(validation_set[feature_cols])[:, 1].reshape(-1, 1)

    platt = LogisticRegression()
    platt.fit(validation_prob, validation_set["RI"])
    return platt


def evaluate(model, test_set, feature_cols=settings.FEATURE_COLS, thresholds=settings.THRESHOLDS, calibrator=None):
    X_test, y_test = test_set[feature_cols], test_set["RI"]

    # Evaluate performance
    print("Evaluate model...")
    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"Test PR-AUC: {pr_auc:.4f}")

    threshold_metrics = {}
    for threshold in thresholds:

        # If the predicted value is above the threshold, set to 1
        y_pred = (y_prob >= threshold).astype(int)

        # Evaluation metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall  = recall_score(y_test, y_pred, zero_division=0)

        threshold_metrics[str(threshold)] = {
            "precision": float(precision),
            "recall": float(recall),
            "n_predicted_positive": int(y_pred.sum()),
        }

        print(f"Threshold {threshold:.2f} -> Precision {precision:.3f} | Recall {recall:.3f}")

    # Compare against trivial baselines

    # Baseline A: always predict the training prevalence (or all zeros)
    pr_auc_majority = average_precision_score(y_test, np.zeros_like(y_test, dtype=float))

    # Baseline B: persistence
    # If the storm intensified over the last 12 h, predict RI
    persist_score = (X_test["delta_vmax_12h"] > 0).astype(float)
    pr_auc_persist = average_precision_score(y_test, persist_score)

    print(f"Majority / all-zero PR-AUC: {pr_auc_majority:.4f}")
    print(f"Persistence PR-AUC: {pr_auc_persist:.4f}")
    print(f"XGBoost PR-AUC: {pr_auc:.4f}")

    result = {
        "y_prob": y_prob,
        "pr_auc": pr_auc,
        "thresholds": threshold_metrics,
        "pr_auc_majority": pr_auc_majority,
        "pr_auc_persistence": pr_auc_persist,
    }

    if calibrator is not None:
        test_cal = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

        print("PR-AUC raw / Platt:",
            average_precision_score(y_test, y_prob),
            average_precision_score(y_test, test_cal))

        print("Brier raw / Platt:",
            brier_score_loss(y_test, y_prob),
            brier_score_loss(y_test, test_cal))

        result["y_prob_cal"] = test_cal

    return result


def build_metrics(training_set, validation_set, test_set, scale_pos_weight, eval_result, feature_cols=settings.FEATURE_COLS):
    return {
        "model": settings.MODEL_NAME,
        "split": {
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
        "pr_auc_test": float(eval_result["pr_auc"]),
        "pr_auc_majority": float(eval_result["pr_auc_majority"]),
        "pr_auc_persistence": float(eval_result["pr_auc_persistence"]),
        "thresholds": eval_result["thresholds"],
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
):
    artifacts_dir, results_dir = Path(artifacts_dir), Path(results_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_df = test_set[["storm_id", "name", "datetime", "RI"]].copy()
    pred_df["y_prob"] = eval_result["y_prob"]
    pred_df.to_csv(artifacts_dir / settings.TEST_PREDICTIONS_FILE, index=False)

    # Save model, feature list and Platt calibrator
    print("Saving model...")
    joblib.dump(model, artifacts_dir / settings.MODEL_FILE)
    joblib.dump(feature_cols, artifacts_dir / settings.FEATURE_COLS_FILE)
    joblib.dump(calibrator, artifacts_dir / settings.CALIBRATOR_FILE)
    print("Model, feature list and calibrator saved...")

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
    raw = model.predict_proba(X)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def main():
    df = load_data()
    training_set, validation_set, test_set = split_data(df)
    model, scale_pos_weight = train_model(training_set, validation_set)
    calibrator = calibrate(model, validation_set)
    eval_result = evaluate(model, test_set, calibrator=calibrator)
    metrics = build_metrics(training_set, validation_set, test_set, scale_pos_weight, eval_result)
    save_artifacts(model, calibrator, settings.FEATURE_COLS, test_set, eval_result, metrics)

    # Compute SHAP
    explain(model, test_set[settings.FEATURE_COLS], settings.FEATURE_COLS)


if __name__ == "__main__":
    main()
