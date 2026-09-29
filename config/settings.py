"""Single source of settings for training, evaluation and saved artifacts.

Values are the v2 settings (AUDIT.md §8 session 6). Changing any of them changes results.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paths
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "hurdat2_processed_observations.parquet"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

# Artifact names
MODEL_VERSION = "v2"
MODEL_NAME = f"xgb_ri_{MODEL_VERSION}"
MODEL_FILE = f"{MODEL_NAME}.joblib"
FEATURE_COLS_FILE = f"feature_cols_{MODEL_VERSION}.joblib"
CALIBRATOR_FILE = f"platt_calibrator_{MODEL_VERSION}.joblib"
TEST_PREDICTIONS_FILE = f"test_predictions_{MODEL_VERSION}.csv"
CALIBRATION_PLOT_FILE = f"calibration_{MODEL_VERSION}.png"
FEATURE_IMPORTANCE_FILE = f"feature_importance_{MODEL_VERSION}.csv"
METRICS_FILE = f"metrics_{MODEL_VERSION}.json"
SHAP_IMPORTANCE_FILE = f"shap_importance_{MODEL_VERSION}.csv"
SHAP_SUMMARY_FILE = f"shap_summary_{MODEL_VERSION}.png"
# Risk bands, thresholds and provenance saved next to the model for the app
MODEL_META_FILE = f"model_meta_{MODEL_VERSION}.json"

# Model features, in the order the model was trained on
FEATURE_COLS = [
    "vmax", "mslp",
    "delta_vmax_6h", "delta_vmax_12h",
    "latitude", "longitude",
    "translation_speed",
    "storm_age_hours",
    "month",
]

# Statuses that count as a tropical cyclone, required at t and at t+24 (AUDIT.md §6 decision 4)
TROPICAL_STATUSES = ["TD", "TS", "HU", "SD", "SS"]

# Split by storm genesis year (AUDIT.md §6 decisions 12, 12b). The VAL_* years are the
# calibration years: Platt, thresholds and risk bands are fitted on them only.
TRAIN_MIN_YEAR = 1980
TRAIN_MAX_YEAR = 2015
VAL_MIN_YEAR = 2016
VAL_MAX_YEAR = 2019
TEST_MIN_YEAR = 2020

# Early stopping: season-grouped CV folds inside the train years, median best rounds, refit
CV_FOLDS = 5

# XGBoost hyperparameters (scale_pos_weight is computed from the train split)
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,
    random_state=42,
)

# xgboost hist results depend on the thread count (n_jobs=1 vs -1 changed v1's
# best_iteration from 94 to 167, see CLAUDE.md), so it is pinned. Never use -1.
N_JOBS = 1

# Risk bands on calibrated P(RI): edges = BAND_MULTIPLIERS x a 2016-19 base rate.
# Low < 1x, Elevated 1-2x, High >= 2x. The edges are also the reported thresholds.
# BAND_BASE_RATE: "observed" (RI rate of the calibration rows) or "mean_calibrated"
# (mean calibrated P(RI) on those rows). Both are recorded in the metrics.
BAND_BASE_RATE = "observed"
BAND_MULTIPLIERS = [1, 2]
BAND_LABELS = ["Low", "Elevated", "High"]

# Bootstrap CI for test PR-AUC, resampling whole storms. Resamples with no RI case
# are skipped; fewer than BOOTSTRAP_MIN_VALID usable resamples is an error.
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95
BOOTSTRAP_MIN_VALID = 0.95
