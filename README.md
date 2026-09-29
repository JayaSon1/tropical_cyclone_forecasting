# tropical_cyclone_forecasting

Atlantic rapid-intensification (RI) classifier: XGBoost on HURDAT2 best-track data, with Platt calibration.

## Scope
The model gives **P(RI | the storm stays a tropical cyclone over water for the next 24 h)**, where RI is an increase of at least 30 kt in maximum wind over 24 h.
- Rows are excluded when the status at t or t+24 h is not TD/TS/HU/SD/SS, or when a landfall (`L` record) falls in [t, t+24 h].
- Those filters use future information. The metrics below therefore describe performance **within this scope**, not operational performance on every storm.
- Rows without a 6 h / 12 h history or without `mslp` get no forecast.

## Run
From the repo root, with the project venv:
```
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m data.parse_hurdat2        # raw -> labels -> features -> parquet
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe src/models/xgboost_model.py     # split, train, calibrate, evaluate, SHAP
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

- **Data files are not tracked.** On a fresh clone, run `python -m data.parse_hurdat2` first. It reads the tracked raw file `data/raw/hurdat2-1851-2025-02272026.txt` and regenerates `data/processed/hurdat2_raw.parquet` and `data/processed/hurdat2_processed_observations.parquet` (paths in `config/settings.py`). Until then, the golden-metrics and model tests skip.
- **Outputs:**
  - `artifacts/`: `xgb_ri_v2.joblib`, `platt_calibrator_v2.joblib`, `feature_cols_v2.joblib`, `model_meta_v2.json` (risk bands, thresholds, library versions, data hash), `test_predictions_v2.csv` (raw and calibrated), plots, SHAP.
  - `results/metrics_v2.json`: all metrics.
- **Scoring:** `src.models.xgboost_model.predict(X)` returns calibrated P(RI) for a DataFrame with exactly the columns in `config.settings.FEATURE_COLS`, in that order.

## Setup (v2)
- **Data:** synoptic times only (00/06/12/18 UTC). t+24 h and the 6 h / 12 h lags are exact timestamp matches within a storm.
- **Split** by storm genesis year:
  - train 1980–2015: 7,684 rows
  - calibration 2016–2019: 1,088 rows, 88 RI
  - test 2020+: 1,516 rows, 114 RI
- **Number of boosting rounds:** season-grouped 5-fold CV inside 1980–2015 picked 44, 61, 93, 10 and 167 rounds. The median, **61**, is used for a refit on all of 1980–2015. `n_jobs=1` is pinned.
- **Calibration years:** Platt calibration and the risk bands (Low < 1×, Elevated 1–2×, High ≥ 2× the 2016–19 RI rate of 0.081) are fitted on 2016–2019 only.

## Results (test, 2020+, used once)
All rows below use the same 1,516 rows and the same 2,000 storm-level bootstrap resamples, with 95% CIs.

| | PR-AUC | 95% CI |
|---|---|---|
| v2 (calibrated) | **0.398** | 0.248 – 0.539 |
| Persistence (`delta_vmax_12h > 0`) | 0.138 | 0.090 – 0.185 |
| Climatology (constant score) | 0.075 | 0.051 – 0.100 |
| v2 − persistence (paired) | **+0.260** | 0.135 – 0.389 |

| | Brier |
|---|---|
| v2 calibrated | 0.0603 |
| v2 raw (class-weighted, uncalibrated) | 0.1692 |
| Climatology (the 2016–19 rate, 0.081, not the test rate) | 0.0696 |
| **Brier skill score vs climatology** | **0.134** (95% CI 0.035 – 0.199) |

Raw and calibrated PR-AUC are equal because Platt scaling is monotonic.

**Secondary line: 2016–2019. These years were used for calibration and band-setting, so treat the result as optimistic.** PR-AUC 0.333, calibrated Brier 0.064 (1,088 rows, 88 RI). These years were not used for early stopping, and Platt scaling doesn't change the ranking, so the PR-AUC here is not inflated. In fact it is lower than on test.

## Limitations
- **Scope, not operations.** The metrics are within the scope above. The scope filters use future information (landfall and status at t+24 h), so operational performance on every storm is not measured.
- **Unstable number of boosting rounds.** The five season-grouped CV folds picked 10, 44, 61, 93 and 167 rounds. The median, 61, is used, but the spread means the chosen number depends strongly on which seasons are held out.
- **Few positives for calibration.** Only 88 RI cases fall in 2016–2019, so the Platt fit and the band edges are noisy. For the same reason the bands are deliberately simple (1× and 2× the base rate).
- **Wide test CI.** The test set has 114 RI cases. The PR-AUC CI (0.248–0.539) is wide, although the paired difference against persistence stays above zero.

## v1 is superseded
The audit of v1 (`AUDIT.md`) found three problems:
1. The t+24 h label and the lag and speed features used row offsets on a series that isn't 6-hourly. Off-synoptic records put hundreds of labels and lags on the wrong time base.
2. The scope filters (landfall, non-tropical status at t+24) were applied by row count rather than by time, and weren't documented as using future information.
3. The same 2016–19 years were used for both early stopping and Platt calibration.

v1's metrics are **superseded** and are not a baseline for v2. They are preserved only at the git tag `pre-fixes`.
