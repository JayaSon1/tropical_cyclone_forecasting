"""Single source of settings for training, evaluation and saved artifacts.

Values are the v1 settings. Changing any of them changes results.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paths
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "hurdat2_processed_observations.parquet"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

# Artifact names
MODEL_VERSION = "v1"
MODEL_NAME = f"xgb_ri_{MODEL_VERSION}"
MODEL_FILE = f"{MODEL_NAME}.joblib"
FEATURE_COLS_FILE = f"feature_cols_{MODEL_VERSION}.joblib"
CALIBRATOR_FILE = f"platt_calibrator_{MODEL_VERSION}.joblib"
TEST_PREDICTIONS_FILE = f"test_predictions_{MODEL_VERSION}.csv"
CALIBRATION_PLOT_FILE = f"calibration_{MODEL_VERSION}.png"
FEATURE_IMPORTANCE_FILE = f"feature_importance_{MODEL_VERSION}.csv"
METRICS_FILE = f"metrics_{MODEL_VERSION}.json"

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

# Split by storm genesis year
TRAIN_MAX_YEAR = 2015
VAL_MIN_YEAR = 2016
VAL_MAX_YEAR = 2019
TEST_MIN_YEAR = 2020

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

# v1 value, kept for parity. xgboost hist results depend on the thread count
# (n_jobs=1 vs -1 changed best_iteration from 94 to 167, see CLAUDE.md), so v1 is
# only reproducible on the machine that trained it. Session 5 pins this.
N_JOBS = -1

# Fixed decision thresholds on raw probabilities (v1 reporting)
THRESHOLDS = [0.10, 0.20, 0.30, 0.50]
