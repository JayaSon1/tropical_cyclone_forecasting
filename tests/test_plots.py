import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from PIL import Image

from config import settings

# Plots must not come out blank (AUDIT.md §2.5 Check L, §6 decision 8).
ROOT = Path(__file__).resolve().parent.parent


def grey_levels(png):
    return len(np.unique(np.asarray(Image.open(png).convert("L"))))


@pytest.fixture
def interactive_show(monkeypatch):
    # With an interactive backend, show() blocks until the window is closed and the
    # figure is gone afterwards. That is why savefig-after-show wrote blank PNGs.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))


def test_calibration_plot_not_blank(tmp_path, interactive_show):
    from src.models.xgboost_model import plot_calibration

    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, 500)
    y_true = (rng.uniform(0, 1, 500) < y_prob).astype(int)
    out = tmp_path / "cal.png"

    plot_calibration(y_true, y_prob, out)

    assert out.exists()
    assert grey_levels(out) > 1


def test_shap_summary_not_blank(tmp_path, monkeypatch, interactive_show):
    from explainability.compute_shap import explain

    rng = np.random.default_rng(0)
    cols = ["a", "b", "c"]
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=cols)
    y = (X["a"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1, random_state=0)
    model.fit(X, y)

    # The output folder is an argument; it is created only when a plot is saved.
    out_dir = tmp_path / "out"
    explain(model, X, cols, artifacts_dir=out_dir)

    out = out_dir / settings.SHAP_SUMMARY_FILE
    assert out.exists()
    assert grey_levels(out) > 1
    assert (out_dir / settings.SHAP_IMPORTANCE_FILE).exists()


def test_importing_shap_module_creates_no_folders(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", "import explainability.compute_shap"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_scripts_force_agg(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = (
        "import explainability.compute_shap, src.models.xgboost_model, matplotlib;"
        "print(matplotlib.get_backend().lower())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "agg"
