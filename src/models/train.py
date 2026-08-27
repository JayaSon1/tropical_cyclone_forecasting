import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score
import joblib
from pathlib import Path
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import json
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from explainability.compute_shap import explain

if __name__ == "__main__":
    
    # Load the processed data
    print("Loading data...")
    df = pd.read_parquet("data/processed/hurdat2_processed_observations.parquet")

    feature_cols = [
        "vmax", "mslp",
        "delta_vmax_6h", "delta_vmax_12h",
        "latitude", "longitude",
        "translation_speed",
        "storm_age_hours",
        "month",
    ]
    
    # Create a temporal 
    # One year per storm = year of first observation (genesis)
    storm_year = (
        df.groupby("storm_id")["datetime"]
        .min()
        .dt.year
        .rename("storm_year")
    )

    df = df.merge(storm_year, left_on="storm_id", right_index=True)

    print("Creating temporal split...")
    training_set = df[df["storm_year"] <= 2015]
    validation_set = df[(df["storm_year"] >= 2016) & (df["storm_year"] <= 2019)]
    test_set = df[df["storm_year"] >= 2020]
    
    # Check that there are no storms that overlap in the training, validation and testing sets
    train_ids = set(training_set["storm_id"])
    val_ids   = set(validation_set["storm_id"])
    test_ids  = set(test_set["storm_id"])

    print("Train ∩ Val :", len(train_ids & val_ids))   # must be 0
    print("Train ∩ Test:", len(train_ids & test_ids))  # must be 0
    print("Val ∩ Test  :", len(val_ids & test_ids))    # must be 0
        
    print(f"Training: {len(training_set)} | Validation: {len(validation_set)} | Test: {len(test_set)}")
    print("Training RI rate:", training_set["RI"].mean().round(3))
    print("Validation RI rate:", validation_set["RI"].mean().round(3))
    print("Test RI rate:", test_set["RI"].mean().round(3))
        
    # Split into data and labels
    X_train, y_train = training_set[feature_cols], training_set["RI"]
    X_validation, y_validation = validation_set[feature_cols], validation_set["RI"]
    X_test, y_test = test_set[feature_cols], test_set["RI"]
    
    # Calculate scaling factor for class imbalance
    print("Computing class imbalance scaling factor...")
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight = {scale_pos_weight:.1f}")
    
    # Train the model
    print("Training model...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    print("Fitting model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=50,
    )
    
    # Evaluate performance
    print("Evaluate model...")
    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"Test PR-AUC: {pr_auc:.4f}")
    
    # Save predictions
    pred_df = test_set[["storm_id", "name", "datetime", "RI"]].copy()
    pred_df["y_prob"] = y_prob
    pred_df.to_csv("artifacts/test_predictions_v1.csv", index=False)
        
    threshold_metrics = {}
    for threshold in [0.10, 0.20, 0.30, 0.50]:
        
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

   
    # Save artefacts
    print("Saving model...")
    Path("artifacts").mkdir(exist_ok=True)
    
    joblib.dump(model, "artifacts/xgb_ri_v1.joblib")
    joblib.dump(feature_cols, "artifacts/feature_cols_v1.joblib")
    
    print("Model and feature list saved...")
    
    
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
            
    # Calibration
    # The imabalance scale factor usually pushes scores in a smooth way so Platt scaling can fix this
    validation_prob  = model.predict_proba(X_validation)[:, 1].reshape(-1, 1)
    test_prob = model.predict_proba(X_test)[:, 1].reshape(-1, 1)

    platt = LogisticRegression()
    platt.fit(validation_prob, y_validation)

    val_cal  = platt.predict_proba(validation_prob)[:, 1]
    test_cal = platt.predict_proba(test_prob)[:, 1]

    print("PR-AUC raw / Platt:",
        average_precision_score(y_test, test_prob),
        average_precision_score(y_test, test_cal))

    print("Brier raw / Platt:",
        brier_score_loss(y_test, test_prob.ravel()),
        brier_score_loss(y_test, test_cal))
    
    # Save Platt calibrator
    joblib.dump(platt, "artifacts/platt_calibrator_v1.joblib")
    joblib.dump(feature_cols, "artifacts/feature_cols_v1.joblib")

    # Plot calibration curve
    prob_true, prob_pred = calibration_curve(y_test, test_cal, n_bins=10, strategy="quantile")

    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve (test)")
    plt.legend()
    plt.show()
    
    # Save calibration plot
    plt.savefig("artifacts/calibration_v1.png", dpi=150)
    
    # Look at feature importance
    feature_importance = (
    pd.Series(model.feature_importances_, index=feature_cols, name="importance")
      .sort_values(ascending=False)
      .reset_index()
      .rename(columns={"index": "feature"})
)
    
    print("Feature Importance: ", feature_importance)
    
    # Save feature importance
    feature_importance.to_csv("artifacts/feature_importance_v1.csv", index=False)
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    
    results = {
        "model": "xgb_ri_v1",
        "split": {"train_max_year": 2015, "val_years": "2016-2019", "test_min_year": 2020},
        "n_train": int(len(training_set)),
        "n_val": int(len(validation_set)),
        "n_test": int(len(test_set)),
        "ri_rate_train": float(y_train.mean()),
        "ri_rate_val": float(y_validation.mean()),
        "ri_rate_test": float(y_test.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "pr_auc_test": float(pr_auc),
        "pr_auc_majority": float(pr_auc_majority),
        "pr_auc_persistence": float(pr_auc_persist),
        "thresholds": threshold_metrics,
        "features": feature_cols,
    }

    with open("results/metrics_v1.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved artifacts/metrics_v1.json")
    
    # Compute SHAP
    explain(model, X_test, feature_cols)