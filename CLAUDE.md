# CLAUDE.md

Atlantic rapid-intensification (RI) classifier: XGBoost on HURDAT2 best-track data, with Platt calibration.
**Model scope:** P(RI, i.e. ≥30 kt increase in 24 h | the storm stays a tropical cyclone over water for the next 24 h).
The scope filters use future information, so test metrics are within-scope, not operational.
The audit, the decisions and the evidence are in `AUDIT.md` (§6 Decisions, §8 fix-session order). Read it before changing modelling code.

## Commands
Run from the repo root. Always use the venv: the system `python` lacks pytest and joblib.
```
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m data.parse_hurdat2          # raw -> labels -> features -> parquet
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe src/models/xgboost_model.py     # split, train, calibrate, metrics, SHAP (overwrites artifacts/, results/)
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m pytest tests/test_irene.py -q -p no:cacheprovider
```

## Pipeline map (v1; details and line numbers in AUDIT.md §1)
1. Parse: `data/parse_hurdat2.py:parse_hurdat2`
2. Label: `src/features/labels.py:ri_labels`
3. Features: `src/features/build_features.py:extract_features`
4. Filter and save parquet: the `parse_hurdat2.py` `__main__` block
5. Split, train, calibrate, evaluate, save: the `src/models/xgboost_model.py` `__main__` block
6. SHAP: `explainability/compute_shap.py:explain`

`config/` and `src/interpret/` are empty. Everything is hardcoded until fix session 2.

## Modelling rules (Jaya's decisions, AUDIT.md §6)
- Keep only synoptic times (00/06/12/18 UTC) before labelling. t+24 and the 6 h / 12 h lags use an **exact-timestamp match within the storm**, never `shift(n)`.
- Landfall markers come from the raw `L` records **before** off-synoptic rows are dropped. Exclude a row if there is any `L` in [t, t+24 h].
- Exclude a row if its status at t or at t+24 is not TD/TS/HU/SD/SS.
- Training uses storm genesis year ≥ 1980. Train is 1980–2015, calibration 2016–2019, test 2020+. The split is by storm, never by row, and no storm may appear in two sets. Assert this; don't just print it.
- Early stopping uses season-grouped CV inside 1980–2015 and the median best iteration, then a refit on all of 1980–2015. The fallback is ES on 2016 and calibration on 2017–19.
- **Never tune anything on test years.** Platt calibration, thresholds and risk bands are fitted on 2016–2019 only. Test is used once, at the end.
- Report metrics on calibrated probabilities, with raw alongside. Persist calibrated predictions. Report test PR-AUC with a bootstrap 95% CI.
- Risk bands are simple and relative to the calibration-set base rate. Don't fine-tune many cut-offs.
- There is one feature list, in one place, saved with the model. `predict()` only accepts a pandas DataFrame in the saved feature order and raises on a wrong order or missing columns.
- `n_jobs` is pinned in config. Never use `-1` for a model you save.
- Plots use the Agg backend, and `savefig` is called before any `show`.
- Never put a future-derived column (`vmax_24h`, `delta_vmax_24h`, `landfall_next_24h`, t+24 status) into the features.
- v1 vs v2 test metrics are **not like-for-like**, because the test rows change. Compare on validation and say so.

## Workflow rules
- One fix per session, in the order of AUDIT.md §8. Retrain only once, in the final session.
- **Test first:** write a failing test, run it and show the failure, then fix, then show it passing.
- Never change modelling code without a failing test first.
- **Never edit or delete a test just to make it pass.** Stop and ask Jaya.
- Show the pytest output as evidence. Don't just claim the tests pass.
- The v1 golden-metrics test must keep passing until the retrain session. Re-baselining it needs Jaya's approval.
- Keep commits small, one logical change each, with a clear message. Don't commit data or artifacts.

## Gotchas
- **Thread-count reproducibility:** xgboost hist results depend on `n_jobs`. On the v1 data, `n_jobs=1` vs `-1` changed `best_iteration` from 94 to 167 and predictions by up to 0.34.
- **Feature order fails silently with numpy:** xgboost rejects a mis-ordered DataFrame, but a numpy array in the wrong column order is accepted and changed predictions by up to 0.85.
- The system Python (3.13) has no pytest or joblib, so use `hurricane-env/Scripts/python.exe`.
- `data/`, `artifacts/`, `*.txt`, `*.csv` and `*.parquet` are git-ignored. Test fixtures must be force-added or live under `tests/fixtures/` with an ignore exception.
- HURDAT2 includes off-synoptic records (L, I, P, R, T, …): 1,062 in the raw file. Row offsets are not time offsets.
- Only 88 RI positives fall in 2016–2019, so threshold and band estimates are noisy. Keep them simple.
- Paths are relative, so run everything from the repo root. `-m data.parse_hurdat2` is needed for the `src.*` imports.
