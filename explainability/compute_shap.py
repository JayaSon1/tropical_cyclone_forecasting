import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import settings


def explain(model, X_test, feature_cols, artifacts_dir=settings.ARTIFACTS_DIR):
    # Explain log-odds (default, more stable than probabilities)
    explainer = shap.TreeExplainer(model)
    X_explain = X_test

    shap_values = explainer.shap_values(X_explain)
    # For binary XGBClassifier this is a 2D array: (n_rows, n_features)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Create the output folder only when something is saved
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    shap_importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    shap_importance.to_csv(artifacts_dir / settings.SHAP_IMPORTANCE_FILE, index=False)
    print(shap_importance)

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_cols,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(artifacts_dir / settings.SHAP_SUMMARY_FILE, dpi=150, bbox_inches="tight")
    plt.close()

