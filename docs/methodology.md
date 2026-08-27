V1 Notes:
Results:
The HURDAT2-only XGBoost model is useful for ranking (test PR-AUC 0.376 vs 0.129 persistence). Predicted probabilities are not well calibrated: scale_pos_weight produces over-confident scores. For example, predictions near 0.75 correspond to an observed RI frequency of about 0.30. Scores should be treated as a risk rank, not as a true probability, unless recalibrated.

Calibration:
Raw XGBoost probabilities were over-confident because of scale_pos_weight. Platt scaling fitted on the 2016–2019 validation storms brings predicted probabilities in line with observed RI frequencies on the 2020+ test set. Calibrated scores top out near 0.30, consistent with the rarity of the event.

SHAP:
On the 2020+ test storms, mean |SHAP| is highest for latitude, 6-hour intensity change, and longitude, then central pressure and storm age. In a HURDAT2-only model this is expected: location acts as a proxy for the missing large-scale environment, and recent intensification is the main dynamical persistence signal. Current vmax and 12-hour change add less once 6-hour change and pressure are included, consistent with collinear intensity features.