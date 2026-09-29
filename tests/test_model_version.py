import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from config import settings
from src.models import xgboost_model as xm

# After the session-6 retrain the code loads v2 only. The v1 files stay on disk,
# untouched, but no code path may open them (AUDIT.md §8 session 6).
ARTIFACT_NAMES = [v for k, v in vars(settings).items() if k.endswith("_FILE")]


def test_config_names_are_v2_only():
    assert settings.MODEL_VERSION == "v2"
    assert ARTIFACT_NAMES and all("_v2" in name for name in ARTIFACT_NAMES), ARTIFACT_NAMES
    assert not any("v1" in name for name in ARTIFACT_NAMES)


def test_load_model_opens_only_v2_files(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, len(settings.FEATURE_COLS))), columns=settings.FEATURE_COLS)
    y = np.arange(50) % 2
    fake = {
        settings.MODEL_FILE: xgb.XGBClassifier(n_estimators=2, n_jobs=1).fit(X, y),
        settings.CALIBRATOR_FILE: LogisticRegression().fit(X[["vmax"]], y),
        settings.FEATURE_COLS_FILE: list(settings.FEATURE_COLS),
    }
    opened = []

    def spy_load(path):
        opened.append(path)
        return fake[path.name]

    monkeypatch.setattr(xm.joblib, "load", spy_load)
    xm.load_model()
    assert sorted(p.name for p in opened) == sorted(fake)
    assert all(p.parent == settings.ARTIFACTS_DIR and "_v2" in p.name for p in opened), opened


def test_saved_model_is_v2():
    files = [settings.ARTIFACTS_DIR / f for f in
             (settings.MODEL_FILE, settings.CALIBRATOR_FILE, settings.FEATURE_COLS_FILE, settings.MODEL_META_FILE)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        pytest.skip(f"v2 artifacts missing: {missing}")
    model, _, _ = xm.load_model()
    meta = xm.load_meta()
    assert meta["model_version"] == "v2"
    assert model.get_booster().num_boosted_rounds() == meta["n_rounds"]
