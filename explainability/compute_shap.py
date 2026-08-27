import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("artifacts").mkdir(exist_ok=True)

def explain(model, X_test, feature_cols):
    # Explain log-odds (default, more stable than probabilities)
    explainer = shap.TreeExplainer(model)
    X_explain = X_test  # or X_test.sample(500, random_state=42) if you want it faster

    shap_values = explainer.shap_values(X_explain)
    # For binary XGBClassifier this is a 2D array: (n_rows, n_features)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    shap_importance = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    shap_importance.to_csv("artifacts/shap_importance_v1.csv", index=False)
    print(shap_importance)
        
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_cols,
        show=False,
    )
    plt.tight_layout()
    plt.show()
    plt.savefig("artifacts/shap_summary_v1.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    