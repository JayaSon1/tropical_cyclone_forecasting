# Pipeline Audit: RI XGBoost model (v1)

**Audited commit:** `1ecf4acb` ("Working model"). Audit date: 2026-09-29.

**Environment used for all checks:**
- Windows 11, project venv `hurricane-env/Scripts/python.exe`: Python 3.12.10, pandas 3.0.5, numpy 2.5.2, xgboost 3.4.1, scikit-learn 1.9.0, shap 0.52.0, pytest 9.1.1.
- The system `python` (3.13) has neither pytest nor joblib, so it cannot run the pipeline or the tests.
- `requirements.txt` pins no versions and lists `flask` twice. `pyyaml` and `python-dotenv` are listed but nothing imports them.

**Method:**
- Read all of the source.
- Rebuilt the labelled table **in memory** from the raw file and compared it with the saved parquet: it matches (14,528 rows, 868 RI).
- Recomputed metrics from the saved artifacts.
- Retrained in memory to test determinism.
- No repo file was modified except this one. All scripts ran with `PYTHONDONTWRITEBYTECODE=1`. Every snippet is in the [Appendix](#appendix-re-runnable-read-only-checks).

**Scope notes:**
- `src/interpret/` exists but is **empty**. The SHAP code is in `explainability/compute_shap.py`.
- `config/` is **empty**, and there are no YAML, JSON or constants files. Every parameter is hardcoded in `src/models/xgboost_model.py`.
- `artifacts/`, `data/processed/` and `data/raw/test_irene_2011.txt` are **git-ignored** (`.gitignore:7` `data/`, `.gitignore:19` `artifacts/`). Only the parser, the raw HURDAT2 file and `results/metrics_v1.json` are tracked.

---

## 1. Pipeline map

| # | Step | File : function (lines) | Input → output |
|---|------|-------------------------|----------------|
| 1 | Parse raw HURDAT2 | `data/parse_hurdat2.py : parse_hurdat2` (39–126) | `data/raw/hurdat2-1851-2025-02272026.txt` → DataFrame (`storm_id, name, datetime, record_id, status, latitude, longitude, vmax, mslp`), sorted by storm and time (125). Saved as `data/processed/hurdat2_raw.parquet` (138). Keeps **every** record, including off-synoptic L/I/P/… rows. |
| 2 | Label | `src/features/labels.py : ri_labels` (18–61), `landfall_in_next_24h` (4–16) | Adds `vmax_24h` (`shift(-4)` per storm, 26), `delta_vmax_24h`, `is_landfall`, `landfall_next_24h`, and `RI` (`>=30`, 57). `RI` is `NA` when the row is excluded (55). |
| 3 | Features | `src/features/build_features.py : extract_features` (36–66) | Adds `delta_vmax_6h/12h` (grouped `shift(1)`/`shift(2)`, 40–44), `translation_speed` (haversine distance / 6.0, 50–54), `storm_age_hours` (57–60), `month`, `day_of_year` (63–64). Runs on the full labelled frame (`parse_hurdat2.py:163`), so the lags can step through non-tropical rows. |
| 4 | Filter and save | `data/parse_hurdat2.py` `__main__` (166–191) | Drops rows whose RI or any of **10** features is NaN (180). This drops each storm's first 2 rows and all rows with missing `mslp`. Writes `data/processed/hurdat2_processed_observations.parquet`: 14,528 rows. |
| 5 | Split | `src/models/xgboost_model.py` `__main__` (37–49) | Storm year = year of the storm's first **remaining** row. Train ≤2015 (11,459 rows), validation 2016–2019 (1,283), test ≥2020 (1,786). |
| 6 | Train | `xgboost_model.py` (70–98) | `XGBClassifier` using **9** hardcoded features (26–33); `scale_pos_weight` computed from train (72–74); early stopping on validation (88, 96). Saved model: `best_iteration=94`. |
| 7 | Evaluate (raw probabilities) | `xgboost_model.py` (102–127, 143–152) | Test PR-AUC, precision/recall at fixed thresholds 0.1/0.2/0.3/0.5, and majority and persistence baselines. |
| 8 | Calibrate | `xgboost_model.py` (156–174) | Platt `LogisticRegression` fitted on **validation** predictions (159–160) and applied to test. Calibrated metrics are **printed only** (165–171). |
| 9 | Artifacts | `xgboost_model.py` (109, 134–135, 174–175, 189, 202) | `artifacts/test_predictions_v1.csv`, `xgb_ri_v1.joblib`, `feature_cols_v1.joblib` (saved twice), `platt_calibrator_v1.joblib`, `calibration_v1.png`, `feature_importance_v1.csv` |
| 10 | Metrics | `xgboost_model.py` (207–225) | `results/metrics_v1.json`. The print at 227 wrongly says `artifacts/`. |
| 11 | SHAP | `explainability/compute_shap.py : explain` (9–39), called at `xgboost_model.py:230` | `TreeExplainer` on the raw model (log-odds), run on the test split → `artifacts/shap_importance_v1.csv`, `shap_summary_v1.png`. Importing the module creates `artifacts/` as a side effect (7). |

**Running it end to end.** There is no CLI and no config, and every path is relative, so run from the repo root with the venv:
```
hurricane-env\Scripts\python.exe -m data.parse_hurdat2        # steps 1-4 (needs -m for the src.* imports)
hurricane-env\Scripts\python.exe src\models\xgboost_model.py  # steps 5-11; overwrites artifacts/ and results/
```
- Step 2 of that command blocks on `plt.show()` (`xgboost_model.py:186`, `compute_shap.py:37`) when an interactive backend is active.
- There is **no inference / prediction entry point**.

---

## 2. Correctness checks

### 2.1 RI label
| Check | Verdict | Evidence |
|---|---|---|
| Threshold is +30 kt | **OK** | `labels.py:57`: `delta_vmax_24h >= 30`. The comment at :54 says "more than 30" and names "Maria and Kaplan"; the code (≥30) matches Kaplan & DeMaria 2003. |
| "Next 24 h" really is 24 h | **PROBLEM** | `labels.py:26` uses `shift(-4)`, a row offset, grouped by storm, with no check on the timestamp. The raw file has **1,062 records not at 00/06/12/18Z** (981 L, 33 I, 11 R, 10 P, 9 T, 8 S, 5 C, 4 W, 1 G). **219 labelled rows** have a real t→t+24 gap other than 24 h (121 at 18 h, 45 at 21 h, the rest 10–23 h), and 23 of them are RI=1. (Check B) |
| Storms don't bleed into each other | **OK** | Every shift is `groupby("storm_id")` (`labels.py:26, 35`). |
| Storm end / missing t+24 excluded | **OK** | The last 4 rows of each storm get NaN `vmax_24h`, which excludes them (`labels.py:41, 52`). |
| Missing current vmax excluded | **OK** | `labels.py:44`. |
| Non-tropical at **t** excluded | **OK** | `labels.py:47` only keeps TD/TS/HU/SD/SS. |
| Non-tropical at **t+24** handled | **PROBLEM** (policy) | Not checked. **1,522 labelled rows** are EX/LO/WV/DB at t+24 (944 EX, 492 LO, 43 WV, 43 DB) and all but 3 are labelled RI=0. They are "easy negatives" and they inflate precision. (Check C) |
| Landfall excluded | **UNSURE** (intent) | A row is excluded if it or any of the next **4 rows** is an `L` record (`labels.py:9–16, 49`). That is 4 rows, not 24 h, and because L records are themselves off-synoptic rows, the window is often < 24 h. **4,804** otherwise-valid tropical rows are dropped this way (Check H). The exclusion also uses future information, so the model is never trained or evaluated on storms about to make landfall, yet at prediction time you won't know a storm is about to. Whether that is intended is a decision (see §5). |
| Landfall rows absent from labelled set | **OK** | Implied by `labels.py:16` (`| current`). `parse_hurdat2.py:160` prints the count. |

### 2.2 Leakage
| Check | Verdict | Evidence |
|---|---|---|
| Features at t use only data ≤ t | **OK** | `build_features.py:40–48` are grouped `shift(1)`/`shift(2)`; :57–60 subtracts the storm's first time; :63–64 are calendar fields. There are no negative shifts, rolling/centred windows, `bfill` or global scaling. |
| Future-derived columns kept out of X | **OK**, but fragile | `vmax_24h`, `delta_vmax_24h` and `landfall_next_24h` **are in the saved parquet**. X is built only from the explicit list (`xgboost_model.py:66–68`). Any future "use all numeric columns" change would leak. |
| Lag features have correct time spacing | **PROBLEM** (quality, not leakage) | `delta_vmax_6h` spans ≠6 h in **412** labelled rows and `delta_vmax_12h` spans ≠12 h in **723** (Check D). `translation_speed` always divides by 6.0 (`build_features.py:51`), so it is wrong in those same 412 rows. |
| Split by storm/season, not by random row | **OK** | Storm-level year split (`xgboost_model.py:37–49`). |
| No storm in both train and test | **OK**, not enforced | Overlap is 0/0/0 (Check E). The code only prints it (56–58) and never asserts. One storm crosses a calendar year: AL312005 (2005-12-30 → 2006-01-06), which is kept whole in train. |
| Calibration fitted on unseen data | **OK**, with caveat | Platt is fitted on validation (`xgboost_model.py:156–160`). But validation was also used for early stopping (96), so it is not fully independent. That is a mild optimistic bias; it does not leak into test. |
| Thresholds not tuned on test | **OK** | Fixed list (`xgboost_model.py:112`). **Note:** they are applied to **raw**, class-weighted probabilities. Raw 0.1 corresponds to about 0.020 calibrated and raw 0.5 to about 0.104. The maximum calibrated probability on test is 0.379 (Check I). |
| Early stopping not using test | **OK** | `eval_set` is validation only (96). |

### 2.3 Feature order
| Check | Verdict | Evidence |
|---|---|---|
| Same order at training and saved | **OK** | The saved `feature_cols_v1.joblib` equals the model's `feature_names_in_` and the list in code (Check G). |
| Order enforced at prediction time | **UNSURE → risk** | There is no prediction code to audit. A pandas DataFrame in the wrong order **is rejected** by xgboost (`ValueError: feature_names mismatch`). A **numpy array with reversed columns is silently accepted**, and predictions change by up to **0.845** (Check J). |
| Single source of truth for features | **PROBLEM** | There are two lists: `parse_hurdat2.py:166–177` (10 features, including `day_of_year`) and `xgboost_model.py:26–33` (9 features). The parquet is filtered on the 10-feature list, so the model's data depends on a list the model doesn't use. |
| SHAP / importance use the right names | **OK** | Both label by position from `feature_cols` on the same `X_test` (`xgboost_model.py:193, 230`; `compute_shap.py:21`). The SHAP CSV **does** contain feature names (`reset_index` at :24). |

### 2.4 Class imbalance
| Check | Verdict | Evidence |
|---|---|---|
| `scale_pos_weight` = neg/pos from **train only** | **OK** | `xgboost_model.py:72–74`. Value 16.23 (`metrics_v1.json`), matching the train RI rate of 0.058. |
| Consequence handled | **OK** | Platt calibration corrects the resulting over-confidence. Test Brier: 0.150 raw → 0.051 calibrated; PR-AUC is unchanged at 0.3855 because the mapping is monotonic (Check I). |

### 2.5 Reproducibility
| Check | Verdict | Evidence |
|---|---|---|
| Metrics reproducible from saved artifacts | **OK** | From `test_predictions_v1.csv` + parquet: `n_test`, `ri_rate_test`, `pr_auc_test` (0.385453), all threshold precision/recall/n_pos, and `pr_auc_persistence` (0.128907) match `metrics_v1.json` exactly. The saved model re-predicts the saved probabilities to within 3e-8 (Checks F, G). |
| Calibrated metrics saved | **PROBLEM** (minor) | Calibrated metrics are only printed (`xgboost_model.py:165–171`), not saved. They can be recomputed from the calibrator (Check I). |
| Seeds fixed | **OK** | `random_state=42` (89) is the only randomness. Platt/lbfgs is deterministic. |
| Retraining reproduces the model | **PROBLEM** | On this machine, `n_jobs=-1` twice gives identical results that equal the saved model. **`n_jobs=1` gives a different model**: `best_iteration` 167 vs 94, predictions differ by up to 0.34 (Check K). Results therefore depend on the core count, so another machine may not reproduce v1. |
| Environment reproducible | **PROBLEM** | `requirements.txt` has no pinned versions. The artifacts are joblib pickles tied to xgboost 3.4.1 / sklearn 1.9.0. |
| Artifacts intact | **PROBLEM** | `calibration_v1.png` and `shap_summary_v1.png` are **blank**: each has a single grey level (Check L). `savefig` is called after `plt.show()` (`xgboost_model.py:186→189`, `compute_shap.py:37→38`). The calibration plot also never calls `plt.figure()` or `close()`, so a later run could overlay onto it. |
| Data provenance | **UNSURE** | The training data comes from a git-ignored parquet. It is regenerable from the tracked raw file (the rebuild matches exactly), but nothing records which file version or hash produced the artifacts. |

---

## 3. Existing tests

**Result:** `2 passed in 1.13s`. Command: `PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`. Git status was the same before and after the run. The ` M data/__pycache__/parse_hurdat2.cpython-312.pyc` was already there.

| Test | Covers | Notes |
|---|---|---|
| `tests/test_irene.py::test_irene_basic_parsing` (10–39) | `parse_hurdat2` on one storm: columns, id/name, sort order, first record values, an L record exists, hemisphere signs | It depends on `data/raw/test_irene_2011.txt`, which is **git-ignored** (`.gitignore:7`), so it will fail on a fresh clone. There is no `sys.path` setup, so it only works via `python -m pytest` from the root. `__main__` (47) is missing `()`. |
| `tests/test_parse_hurdat2_full.py::test_full_hurdat2_parsing` (14–94) | `parse_hurdat2` on the full file: row count > 50k, all ids start with AL, > 1,500 storms, sort order within storms, lat/lon/vmax ranges, Irene present, L records exist | Several asserts are duplicated (29/39, 30/36, 44/88). `min_size >= 1` (50) is always true. The sort check always passes because the parser sorts (125). |

**What the tests miss.** Everything after parsing has no tests:
- the RI label, landfall and status exclusions, and t+24 timing
- feature correctness and leakage
- the split
- class weighting
- calibration
- feature order and the saved artifact
- metric reproducibility
- SHAP
- the malformed-field path in the parser (the bare `int()` at `parse_hurdat2.py:94,99` raises before `handle_none` runs)

There is also no `conftest.py` or pytest config, and the tests use relative paths.

---

## 4. Ranked risks

Severity is the impact on the scientific validity or usability of results (H/M/L). Effort is the size of the fix (S ≈ hours, M ≈ a day, L ≈ multi-day).

| # | Risk | Evidence | Sev | Effort | Action |
|---|------|----------|-----|--------|--------|
| 1 | The t+24 label and the lag/speed features use row offsets on a series that isn't 6-hourly: 219 labels, 412 / 723 lag features and all speeds next to off-synoptic points are on the wrong time base. | `labels.py:26`, `build_features.py:40–51` | H | S–M | **Fix before building the app** |
| 2 | No prediction path enforces feature order; a numpy input is silently mis-ordered (Δp up to 0.85); there are two diverging feature lists. | `xgboost_model.py:26`, `parse_hurdat2.py:166` | H | S | **Fix before building the app** |
| 3 | Rows that are EX/LO/DB/WV at t+24 are labelled 0 (1,522 rows), which inflates the negatives and precision. | `labels.py:47` | M–H | S | **Fix before building the app** (after the §5 decision) |
| 4 | The landfall exclusion uses future information and removes 4,804 rows, so evaluation doesn't reflect deployment where landfall is unknown. | `labels.py:4–16, 49` | M–H | S | **Decide, then fix or document** |
| 5 | Thresholds and reported metrics use raw class-weighted probabilities; calibrated outputs aren't saved; an app that shows "probability" would show inflated numbers. | `xgboost_model.py:112–115, 165–171` | M | S | **Fix before building the app** |
| 6 | Retraining isn't reproducible across thread counts (n_jobs=1 vs -1 gives a different model); no pinned versions; the artifacts are pickles. | `xgboost_model.py:90`, `requirements.txt` | M | S | **Fix before building the app** |
| 7 | The split's no-overlap guarantee is only printed, never asserted. | `xgboost_model.py:56–58` | M | S | **Fix before building the app** (cheap) |
| 8 | Blank PNG artifacts; `plt.show()` blocks headless runs. | `xgboost_model.py:186–189`, `compute_shap.py:37–38` | L | S | Fix before the app if the plots are displayed; otherwise defer |
| 9 | Training spans 1853–2015 (2,884 train rows before 1980, the pre-satellite era) with mixed data quality. | `xgboost_model.py:47` | M | S | **Document and defer** (experiment) |
| 10 | Validation is reused for both early stopping and calibration. | `xgboost_model.py:96, 160` | L–M | M | **Document and defer** |
| 11 | Everything is hardcoded (paths, years, hyperparameters, thresholds); `config/` and `src/interpret/` are empty; there are no entry points. | the whole of `xgboost_model.py` | M | M | **Fix before building the app** (the app needs importable functions) |
| 12 | The test fixture is git-ignored, committed `.pyc` files are tracked, and the tests use relative paths. | `.gitignore:7`, `git ls-files` | L | S | **Fix before building the app** (so CI works) |
| 13 | Future-derived columns sit in the modelling parquet next to the features. | `parse_hurdat2.py:163, 180` | L | S | Document and defer (guard with a test) |

### Proposed tests to add
1. **`test_labels.py`** (synthetic storms, no data files needed):
   - +29 / +30 / +31 kt give 0 / 1 / 1
   - the last 4 rows of a storm give NA
   - an off-synoptic L record inside the window: the expected behaviour is that the timestamp-based t+24 is used, so the test fails today
   - an EX status at t+24, per the decision in §5
   - a landfall at t and at t+k
   - two storms back to back never mix
   - a missing `vmax` gives NA
2. **`test_features_no_leakage.py`:**
   - Perturb all rows after t, recompute the features, and assert the features at t are unchanged.
   - Assert that no column derived from `vmax_24h` or `landfall_next_24h` appears in the model feature list.
3. **`test_feature_time_spacing.py`:** the lag features and `translation_speed` on a storm with an off-synoptic record use the correct Δt. This will fail today.
4. **`test_split.py`:**
   - storm-id sets are disjoint
   - year boundaries are correct
   - AL312005 is wholly in one split
   - split row counts on the real parquet are 11,459 / 1,283 / 1,786
5. **`test_feature_contract.py`:**
   - Saved `feature_cols_v1.joblib` == `model.feature_names_in_` == the single canonical list.
   - The prediction helper reindexes by name and rejects missing or extra columns.
6. **`test_class_weight.py`:** `scale_pos_weight` equals neg/pos computed on the train rows only.
7. **`test_calibration.py`:**
   - The calibrator is fitted only on validation indices (refactor it into a function first).
   - The calibrated output is monotonic in the raw output.
8. **`test_metrics_golden.py`:** recompute from `test_predictions_v1.csv` and match `metrics_v1.json` to within 1e-9 (the Check F logic). Skip when the artifacts are missing.
9. **`test_determinism.py`:**
   - Train twice on a small sample with a fixed `n_jobs` and get identical predictions.
   - Document the thread-count dependence.
10. **`test_shap.py`:** the SHAP matrix shape equals `X.shape`; the CSV has a `feature` column equal to the feature set; the PNG is non-blank.
11. **`test_parser_edge_cases.py`:** `-99` / `-999` / blank fields, a malformed `vmax`, a southern / eastern hemisphere row, and a single-storm file committed under `tests/fixtures/`.

---

## 5. Decisions needed from Jaya

> **Answered 2026-09-29. See [§6 Decisions](#6-decisions-2026-09-29).** The questions are kept below for the record.

Answer yes/no. The findings each question depends on are cited.

1. **Time base:** Should the label and lag features be based on timestamps, i.e. t+24 is the record at exactly t+24 h within the storm and the lags are at exactly t−6 h and t−12 h? (§2.1, §2.2; risk 1)
2. **Off-synoptic rows:** Should rows not at 00/06/12/18Z (1,062 raw) be dropped from the modelling table entirely? The alternative is to keep them only as landfall markers. (risk 1)
3. **Dissipation / EX at t+24:** Should a row whose t+24 status is EX/LO/WV/DB be **excluded**? It is currently labelled 0 (1,522 rows). (risk 3)
4. **Landfall scope (a):** Should pre-landfall rows remain excluded at all, given that the app will score storms without knowing whether they will make landfall? (risk 4)
5. **Landfall scope (b):** If they are kept excluded, should the window be "any L record within the next 24 h by time" rather than "within the next 4 rows"? (risk 4)
6. **Pre-1980 data:** Should training start at 1980 or later? That would drop 2,884 of 11,459 train rows. (risk 9)
7. **Probability type:** Should thresholds, reported precision/recall and anything shown in the app use **calibrated** probabilities? (risk 5)
8. **Threshold choice:** Should an operating threshold be selected on validation (for example, a target recall) instead of reporting fixed raw cut-offs? (risk 5)
9. **Validation reuse:** Is reusing the 2016–2019 validation set for both early stopping and Platt fitting acceptable for v1? (risk 10)
10. **Reproducibility:** Should training pin `n_jobs` to a fixed number, with library versions pinned in `requirements.txt`? (risk 6)
11. **Repo hygiene:** Should test fixtures be committed under `tests/fixtures/` and the tracked `.pyc` files be removed from git? (risk 12)

---

## 6. Decisions (2026-09-29)

All decisions below are Jaya's. None of them has been implemented yet. Numbers refer to the risk table in §4.

### Model scope
The model predicts **the probability of RI (≥30 kt increase in 24 h) given that the storm remains a tropical cyclone over water for the next 24 h.**
- Decisions 4 and 5 below enforce this scope.
- Enforcing it uses future information (status and landfall at t+24), so test metrics describe performance **within the scope**, not operational performance on every storm.
- Operational (unfiltered) evaluation is **deferred**.

### Fix now
1. **Time base (risks 1, 3):** keep only synoptic times (00/06/12/18 UTC) before labels and features are computed, so t+24 and the 6 h / 12 h lags are true time offsets.
   - *Implementation note (from Check A):* even after this filter, the series has gaps. t+24 and the lags must therefore use an **exact-timestamp match** within the storm, not `shift(n)`.
2. **Feature order (risk 2):** one feature list, defined in one place and saved with the model.
   - `predict()` accepts only a pandas DataFrame and validates column names and order. A wrong order or a missing column raises a clear error.
   - Remove the duplicate list at `parse_hurdat2.py:166–177`.
   - *Implementation note (session 3):* `predict()` also rejects boolean or non-numeric columns and any NaN in a feature, naming the column(s) and the row count. The model was trained only on complete rows. A storm's first rows have no 6 h / 12 h lag, and some rows have no `mslp`. **For such rows the app must show "no forecast available" and must not call `predict()`.**
3. **Determinism (risk 6):** pin `n_jobs` to a fixed value stored in one config location, and document it.
4. **Non-tropical at t+24 (risk 3):** exclude rows whose t+24 status is not TD/TS/HU/SD/SS, e.g. EX, LO, WV, DB. This enforces the scope above.
5. **Landfall (risk 4):** keep the landfall exclusion as a scope filter and document that it uses future information.
   - *Implementation note:* because of decision 1, landfall markers must come from the raw `L` records **before** off-synoptic rows are dropped.
   - The window is by time: any `L` record with a timestamp in [t, t+24 h] in the same storm.
6. **Metrics (risk 5):**
   - Report metrics on **calibrated** probabilities, keeping the raw ones for comparison.
   - Choose thresholds and risk bands on the calibration years only, never on test.
   - Persist the calibrated predictions.
   - Risk bands are **simple, defined relative to the calibration-set base rate**; no fine-tuning of many cut-offs.
   - Report a **bootstrap 95% CI for test PR-AUC**.
7. **Early stopping vs calibration (risk 10).** Superseded by 12b below.
8. **Blank plots (risk 8):** call `savefig` before `show`, and use the non-interactive **Agg** backend in scripts and tests.
9. **Irene test (risk 12):** make it pass on a fresh clone using a small tracked fixture.
10. **Hygiene (risk 12):** `git rm --cached` the committed `.pyc` files, and ignore `__pycache__/` and `.pytest_cache/`.
11. **Importable training (risk 11):**
    - Functions for load, split, train, calibrate, evaluate and save.
    - `load_model()`, plus `predict()` returning calibrated probabilities.

### Decided after Checks A and B (§7)
12. **Early data (risk 9): YES.** Training uses storm genesis year ≥ 1980, so train = **1980–2015**.
    - 12b. **Calibration block size: option (c).**
      - Early stopping via **season-grouped CV within 1980–2015**: a season is never split across folds, so no storm appears in two folds. Take the **median best iteration** across folds.
      - Then retrain on all of 1980–2015 with that fixed number of rounds.
      - Platt calibration, thresholds and risk bands are fitted on **all of 2016–2019**.
      - Test (2020+) is used **once**, at the end.
      - **Fallback** if (c) proves impractical: option (b), i.e. early stopping on 2016 and calibration on 2017–2019.
    - This replaces decision 7's 2016–17 / 2018–19 split, which Check A showed leaves only 31 RI positives for calibration.

### Reporting rule for v1 vs v2
v1 and v2 test metrics are **not like-for-like**: the time base and the scope filters change which test rows exist. The final comparison and the README must state this. Compare on validation where possible, and explain the difference.

---

## 7. Checks A and B (2026-09-29)

Both checks ran read-only. The script is in the [Appendix](#checks-a-and-b-script) and was piped to the venv Python on stdin. "Simulated" means decisions 1 + 4 (+ 5 by time) applied **in memory** as described in §6. The storm year is the year of the storm's first remaining modelling row, matching the current split.

### Check A: validation split feasibility
| Block | Current: rows / RI+ / rate | Simulated d1+d4: rows / RI+ / rate |
|---|---|---|
| train ≤2015 | 11,459 / 665 / 0.058 | 10,337 / 656 / 0.063 |
| early stopping 2016–2017 | 667 / 57 / 0.085 | 578 / 57 / 0.099 |
| **calibration 2018–2019** | **616 / 31 / 0.050** | **510 / 31 / 0.061** |
| all of 2016–2019 | 1,283 / 88 / 0.069 | 1,088 / 88 / 0.081 |
| test ≥2020 | 1,786 / 115 / 0.064 | 1,516 / 114 / 0.075 |

Simulated RI positives by year, 2012–2019: 12, 0, 2, 8, 19, 38, 19, 12.

**FLAG:** 2018–2019 has 31 positives, below the ~50 threshold. These were the alternatives presented:

| Option | Early stopping / calibration | Pros | Cons |
|---|---|---|---|
| (a) | 2016–17 / 2018–19 as decided | Simple; matches decision 7 | Thresholds and bands rest on 31 positives, so they are noisy |
| (b) | 2016 / 2017–19 | 69 calibration positives; recent climate | Early stopping on 19 positives is noisy |
| (c) | grouped CV in train / all of 2016–19 | 88 calibration positives; calibration fully independent of early stopping | More compute and code; the final model is refit with a fixed number of rounds |
| (d) | 2016–19 / CV Platt on out-of-fold training predictions | About 650 positives | Calibrated to the training era and the fold models, not the final model |

**Jaya chose (c), with (b) as the fallback** (§6, 12b).

### Check B: early data
| Era | Current: rows / RI+ / rate | Simulated: rows / RI+ / rate |
|---|---|---|
| pre-1944 | 233 / 4 / 0.017 | 199 / 3 / 0.015 |
| 1944–1969 | 1,476 / 87 / 0.059 | 1,334 / 84 / 0.063 |
| 1970–1979 | 1,175 / 101 / 0.086 | 1,120 / 101 / 0.090 |
| 1980+ | 11,644 / 676 / 0.058 | 10,288 / 670 / 0.065 |

**Effect of the mslp `dropna`.** Rows below are ones that pass everything except mslp, under the simulated labels:

| Era | Dropped (mslp missing) | Kept | Kept % |
|---|---|---|---|
| pre-1944 | 12,384 | 199 | 1.6% |
| 1944–1969 | 3,640 | 1,334 | 26.8% |
| 1970–1979 | 1,728 | 1,120 | 39.3% |
| **pre-1980 total** | **17,752** | **2,653** | **13.0%** |
| 1980+ | 464 | 10,288 | 95.7% |

Under the current pipeline, pre-1980 drops 19,028 and keeps 2,884.

**Representativeness:**

| Subset | n | RI rate | vmax q25/50/75 (kt) | Storm-max vmax median | Latitude median | Landfall later in storm | Landfall within 48 h | Median h to landfall |
|---|---|---|---|---|---|---|---|---|
| pre-1980 kept | 2,653 | 0.071 | 40 / 60 / 80 | 90 | 26.3 | 0.21 | 0.08 | 66 |
| pre-1980 dropped | 17,752 | 0.032 | 40 / 50 / 75 | 85 | 23.7 | 0.28 | 0.08 | 77 |
| 1980+ | 10,752 | 0.062 | 35 / 50 / 70 | 80 | 23.6 | 0.28 | 0.10 | 60 |

**Interpretation:**
- The pre-1980 rows that survive are a pressure-observed subset.
- They are stronger (median 60 kt vs 50 kt) and further north (26.3° vs 23.6°).
- They are less often followed by a landfall (21% vs 28%).
- They have a higher RI rate than 1980+, especially 1970–79 (0.090).

The recommendation was to start at 1980. **Jaya decided YES** (§6, #12). That removes about 2,650 rows and about 188 positives, roughly 22% of train positives.

---

## 8. Fix-session order

Retraining happens only once, in session 6. Every session is test-first: write the test, show it fails, then fix.

| # | Session | Changes results? | Test-first plan |
|---|---|---|---|
| 0 | v1 golden-metrics safety net | no | Recompute from the v1 artifacts and match `metrics_v1.json` (Check F logic). It must pass before session 1 and keep passing until session 6 deliberately re-baselines it. |
| 1 | Hygiene + Irene fixture + plots (#8, #9, #10) | no | The Irene test points at the tracked `tests/fixtures/` file; it fails because the file is missing → add the fixture, `git rm --cached` the `.pyc` files, fix `.gitignore`. The plot helper under Agg writes a PNG with more than one grey level; it fails today. |
| 2 | Importable refactor + config (#11) | no | A parity test: `load → split → train → evaluate` through the new functions, with v1 settings (`n_jobs=-1`), reproduces v1 test PR-AUC 0.385453 in memory. It fails on import → refactor. |
| 3 | Feature contract (#2) | no | `predict()` raises on a reordered DataFrame, a missing column or a numpy array. The saved feature list equals the config list. Modelling row count is unchanged after removing the duplicate list. |
| 4 | Label scope (#1, #4, #5) | **yes** (data) | Synthetic storms: an off-synoptic `L` record inside the window; a missing synoptic step (no t+24 → NA); exact 6/12 h lags and speed Δt; EX/LO at t+24 → NA; `L` in [t, t+24 h] → excluded; +29/+30 kt boundary. |
| 5 | Training setup (#3, #6, #12, 12b) | **yes** (model) | Small synthetic data: the split has 1980 ≤ train ≤ 2015 and disjoint storms; CV folds never share a season; rounds = median best iteration; the calibrator, thresholds and bands see only 2016–19 rows; the config `n_jobs` is used and two fits are identical; the calibrated-prediction CSV is written; bootstrap CI is deterministic with a seed. **No full retrain.** |
| 6 | Single retrain → v2 | — | Regenerate the data, v2 artifacts, metrics (raw + calibrated, CI) and SHAP. Re-baseline the golden test to v2 **with Jaya's approval**. Compare v1 vs v2 on validation; the test comparison is not like-for-like (§6). Update the README. |

### Follow-ups
- `compute_shap.explain` uses hard-coded `artifacts/` paths and creates `artifacts/` on import. Move the paths to `config/settings.py` and remove the import-time side effect (do with session 3 or before the app). *(Found in session 2.)*

---

## Appendix: re-runnable read-only checks

Run from the repo root. Neither script writes to the repo: they read the raw file, the parquet and the artifacts, and retrain only in memory.
```
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe audit_checks.py
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe audit_checks2.py
PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
git ls-files --error-unmatch data/raw/test_irene_2011.txt; git check-ignore -v data/raw/test_irene_2011.txt artifacts/xgb_ri_v1.joblib
```
Save the two scripts below anywhere outside the repo (or pipe them in) and run them with the repo root as the working directory.

### `audit_checks.py` (Checks A–G, L)
```python
import sys, json
sys.path.insert(0, ".")
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score
from data.parse_hurdat2 import parse_hurdat2
from src.features.labels import ri_labels
from src.features.build_features import extract_features

print("== A. Rebuild labelled table in memory (no writes) ==")
raw = parse_hurdat2("data/raw/hurdat2-1851-2025-02272026.txt")
lab = ri_labels(raw)
feat = extract_features(lab)
g = feat.groupby("storm_id")
feat["dt_24"] = (g["datetime"].shift(-4) - feat["datetime"]).dt.total_seconds() / 3600
feat["dt_6"] = (feat["datetime"] - g["datetime"].shift(1)).dt.total_seconds() / 3600
feat["dt_12"] = (feat["datetime"] - g["datetime"].shift(2)).dt.total_seconds() / 3600
feat["status_24"] = g["status"].shift(-4)
FEATS10 = ["vmax","mslp","delta_vmax_6h","delta_vmax_12h","latitude","longitude",
           "translation_speed","storm_age_hours","month","day_of_year"]
m = feat.dropna(subset=["RI"] + FEATS10).copy()
m["RI"] = m["RI"].astype(int)
proc = pd.read_parquet("data/processed/hurdat2_processed_observations.parquet")
print("rebuilt modelling rows:", len(m), "| saved parquet rows:", len(proc),
      "| RI sum rebuilt/saved:", int(m["RI"].sum()), int(proc["RI"].sum()))

print("\n== B. t+24 gap (hours) in labelled rows ==")
print("rows with t+24 gap != 24h:", int((m["dt_24"] != 24).sum()))
print(m.loc[m["dt_24"] != 24, "dt_24"].value_counts().sort_index().to_string())
print("RI positives among them:", int(m.loc[m["dt_24"] != 24, "RI"].sum()))

print("\n== C. status at t+24 for labelled rows ==")
nt = ~m["status_24"].isin(["TD","TS","HU","SD","SS"])
print("labelled rows non-tropical at t+24:", int(nt.sum()), "| of which RI=1:", int(m.loc[nt,"RI"].sum()))
print(m.loc[nt, "status_24"].value_counts().to_string())

print("\n== D. lag-gap mismatches in labelled rows ==")
print("delta_vmax_6h gap != 6h:", int((m["dt_6"] != 6).sum()))
print("delta_vmax_12h gap != 12h:", int((m["dt_12"] != 12).sum()))

print("\n== E. split overlap / boundaries (saved parquet) ==")
sy = proc.groupby("storm_id")["datetime"].min().dt.year
tr, va, te = set(sy[sy<=2015].index), set(sy[(sy>=2016)&(sy<=2019)].index), set(sy[sy>=2020].index)
print("overlaps tr&va, tr&te, va&te:", len(tr&va), len(tr&te), len(va&te))
span = proc.groupby("storm_id")["datetime"].agg(["min","max"])
print("storms spanning calendar years:\n", span[span["min"].dt.year != span["max"].dt.year].to_string())
s = proc.merge(sy.rename("sy"), left_on="storm_id", right_index=True)
print("n train/val/test:", int((s.sy<=2015).sum()), int(((s.sy>=2016)&(s.sy<=2019)).sum()), int((s.sy>=2020).sum()))

print("\n== F. metrics recomputed from artifacts/test_predictions_v1.csv ==")
p = pd.read_csv("artifacts/test_predictions_v1.csv")
met = json.load(open("results/metrics_v1.json"))
print("n_test", len(p), met["n_test"], "| ri_rate", p.RI.mean(), met["ri_rate_test"])
print("pr_auc", average_precision_score(p.RI, p.y_prob), met["pr_auc_test"])
for t, v in met["thresholds"].items():
    yp = (p.y_prob >= float(t)).astype(int)
    print(t, "prec", round(precision_score(p.RI, yp, zero_division=0),6), round(v["precision"],6),
          "rec", round(recall_score(p.RI, yp),6), round(v["recall"],6), "npos", int(yp.sum()), v["n_predicted_positive"])
te_rows = s[s.sy>=2020]
print("persistence pr_auc from parquet:", average_precision_score(te_rows.RI, (te_rows.delta_vmax_12h>0).astype(float)), met["pr_auc_persistence"])

print("\n== G. saved artifacts ==")
import joblib
fc = joblib.load("artifacts/feature_cols_v1.joblib"); mdl = joblib.load("artifacts/xgb_ri_v1.joblib")
print("feature_cols_v1:", fc)
print("model feature_names_in_ == fc:", list(mdl.feature_names_in_) == fc, "| best_iteration:", mdl.best_iteration)
print("max |model(parquet test) - saved y_prob|:", float(np.abs(mdl.predict_proba(te_rows[fc])[:,1] - p.y_prob.values).max()))
try:
    mdl.predict_proba(te_rows[fc[::-1]]); print("reordered DataFrame: silently accepted")
except Exception as e:
    print("reordered DataFrame rejected:", type(e).__name__)
print(pd.read_csv("artifacts/shap_importance_v1.csv").head(3).to_string())

print("\n== L. PNG artifacts ==")
from PIL import Image
for f in ["artifacts/calibration_v1.png", "artifacts/shap_summary_v1.png"]:
    a = np.asarray(Image.open(f).convert("L")); print(f, a.shape, "unique grey levels:", len(np.unique(a)))
```

**Output (2026-09-29)**, trimmed to the lines that matter:
```
rebuilt modelling rows: 14528 | saved parquet rows: 14528 | RI sum rebuilt/saved: 868 868
rows with t+24 gap != 24h: 219      (18h:121, 21h:45, 20h:8, 22h:7, 23h:6, ... 10.3h:1) | RI positives among them: 23
labelled rows non-tropical at t+24: 1522 | of which RI=1: 3   (EX 944, LO 492, WV 43, DB 43)
delta_vmax_6h gap != 6h: 412
delta_vmax_12h gap != 12h: 723
overlaps tr&va, tr&te, va&te: 0 0 0
storms spanning calendar years: AL312005 2005-12-30 12:00 -> 2006-01-06 12:00
n train/val/test: 11459 1283 1786
n_test 1786 1786 | ri_rate 0.06438969764837627 0.06438969764837627
pr_auc 0.38545263568634536 0.38545263568634536
0.1 prec 0.084435 0.084435 rec 1.0 1.0 npos 1362 1362
0.2 prec 0.104967 0.104967 rec 0.973913 0.973913 npos 1067 1067
0.3 prec 0.130383 0.130383 rec 0.947826 0.947826 npos 836 836
0.5 prec 0.177551 0.177551 rec 0.756522 0.756522 npos 490 490
persistence pr_auc from parquet: 0.12890723479060512 0.12890723479060512
feature_cols_v1: ['vmax', 'mslp', 'delta_vmax_6h', 'delta_vmax_12h', 'latitude', 'longitude', 'translation_speed', 'storm_age_hours', 'month']
model feature_names_in_ == fc: True | best_iteration: 94
max |model(parquet test) - saved y_prob|: 2.97e-08
reordered DataFrame rejected: ValueError
shap_importance_v1.csv head: latitude 0.5418, delta_vmax_6h 0.5224, longitude 0.3279
artifacts/calibration_v1.png (720, 960) unique grey levels: 1
artifacts/shap_summary_v1.png (750, 990) unique grey levels: 1
```

### `audit_checks2.py` (environment, Checks H–K)
```python
import sys, platform
sys.path.insert(0, ".")
import pandas as pd, numpy as np, sklearn, xgboost as xgb, shap, joblib
from sklearn.metrics import average_precision_score, brier_score_loss
from data.parse_hurdat2 import parse_hurdat2
from src.features.labels import ri_labels

print("python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__,
      "| xgboost", xgb.__version__, "| sklearn", sklearn.__version__, "| shap", shap.__version__)

raw = parse_hurdat2("data/raw/hurdat2-1851-2025-02272026.txt")
off = ~((raw.datetime.dt.hour % 6 == 0) & (raw.datetime.dt.minute == 0))
print("raw rows not at 00/06/12/18:00Z:", int(off.sum()))
print(raw.loc[off, "record_id"].fillna("blank").value_counts().to_string())

# H. landfall-window exclusions
lab = ri_labels(raw)
trop = lab.status.isin(["TD","TS","HU","SD","SS"]); cur = lab.vmax.notna(); fut = lab.vmax_24h.notna()
print("tropical rows w/ vmax & t+24 but excluded only by landfall window:",
      int((trop & cur & fut & lab.landfall_next_24h).sum()))

proc = pd.read_parquet("data/processed/hurdat2_processed_observations.parquet")
fc = joblib.load("artifacts/feature_cols_v1.joblib")
sy = proc.groupby("storm_id")["datetime"].transform("min").dt.year
tr, va, te = proc[sy <= 2015], proc[(sy >= 2016) & (sy <= 2019)], proc[sy >= 2020]
print("train rows with storm_year < 1980:", int((sy[sy <= 2015] < 1980).sum()), "of", len(tr))

# J. numpy input bypasses the feature-name check
mdl = joblib.load("artifacts/xgb_ri_v1.joblib")
ref = mdl.predict_proba(te[fc])[:, 1]
alt = mdl.predict_proba(te[fc[::-1]].to_numpy())[:, 1]
print("numpy input with reversed columns accepted; max |diff| vs correct:", float(np.abs(ref - alt).max()))

# I. calibrated metrics
platt = joblib.load("artifacts/platt_calibrator_v1.joblib")
cal = platt.predict_proba(ref.reshape(-1, 1))[:, 1]
print("test PR-AUC raw/cal:", average_precision_score(te.RI, ref), average_precision_score(te.RI, cal))
print("test Brier raw/cal:", brier_score_loss(te.RI, ref), brier_score_loss(te.RI, cal))
print("max calibrated prob on test:", float(cal.max()), "| raw thresholds 0.1/0.5 map to calibrated:",
      platt.predict_proba(np.array([[0.1], [0.5]]))[:, 1].round(4).tolist())

# K. determinism (in-memory retrain with the exact v1 hyperparameters)
def fit(nj):
    m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=(len(tr) - tr.RI.sum()) / tr.RI.sum(), n_estimators=500, learning_rate=0.05,
        max_depth=4, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30, random_state=42, n_jobs=nj)
    m.fit(tr[fc], tr.RI, eval_set=[(va[fc], va.RI)], verbose=False)
    return m.predict_proba(te[fc])[:, 1], m.best_iteration
a, ia = fit(-1); b, ib = fit(-1); c, ic = fit(1)
print("retrain n_jobs=-1 twice: max|diff|", float(np.abs(a - b).max()), "best_iter", ia, ib)
print("retrain n_jobs=1 vs -1:  max|diff|", float(np.abs(a - c).max()), "best_iter", ic)
print("retrain vs saved model:  max|diff|", float(np.abs(a - ref).max()), "| PR-AUC retrained", average_precision_score(te.RI, a))
```

**Output (2026-09-29):**
```
python 3.12.10 | pandas 3.0.5 | numpy 2.5.2 | xgboost 3.4.1 | sklearn 1.9.0 | shap 0.52.0
raw rows not at 00/06/12/18:00Z: 1062   (L 981, I 33, R 11, P 10, T 9, S 8, C 5, W 4, G 1)
tropical rows w/ vmax & t+24 but excluded only by landfall window: 4804
train rows with storm_year < 1980: 2884 of 11459
numpy input with reversed columns accepted; max |diff| vs correct: 0.8448691368103027
test PR-AUC raw/cal: 0.38545263568634536 0.38545263568634536
test Brier raw/cal: 0.14958971738815308 0.051113311201334
max calibrated prob on test: 0.37930500507354736 | raw thresholds 0.1/0.5 map to calibrated: [0.0202, 0.1043]
retrain n_jobs=-1 twice: max|diff| 0.0 best_iter 94 94
retrain n_jobs=1 vs -1:  max|diff| 0.3406079411506653 best_iter 167
retrain vs saved model:  max|diff| 0.0 | PR-AUC retrained 0.38545263568634536
```

### Checks A and B script
Pipe it on stdin so no file is written: `PYTHONDONTWRITEBYTECODE=1 hurricane-env/Scripts/python.exe - < checks_ab.py`. Save the script outside the repo.
```python
import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from data.parse_hurdat2 import parse_hurdat2
from src.features.labels import ri_labels
from src.features.build_features import extract_features, point_distance_km
TROP = ["TD","TS","HU","SD","SS"]
F9 = ["vmax","mslp","delta_vmax_6h","delta_vmax_12h","latitude","longitude","translation_speed","storm_age_hours","month"]
raw = parse_hurdat2("data/raw/hurdat2-1851-2025-02272026.txt")

# CURRENT pipeline (identical to parse_hurdat2.__main__)
cur = extract_features(ri_labels(raw))
cur = cur.dropna(subset=["RI"] + F9 + ["day_of_year"]).copy(); cur["RI"] = cur["RI"].astype(int)
cur["sy"] = cur.groupby("storm_id")["datetime"].transform("min").dt.year

# SIMULATED decisions 1 + 4 (+5 landfall by time, L markers taken before dropping off-synoptic rows)
syn = raw[(raw.datetime.dt.hour % 6 == 0) & (raw.datetime.dt.minute == 0)].copy().reset_index(drop=True)
def at(offset_h, cols):
    k = syn[["storm_id","datetime"] + cols].copy(); k["datetime"] = k["datetime"] - pd.Timedelta(hours=offset_h)
    return syn[["storm_id","datetime"]].merge(k, on=["storm_id","datetime"], how="left")[cols].values
syn[["vmax_24","status_24"]] = at(24, ["vmax","status"])
syn[["vmax_m6","lat_m6","lon_m6"]] = at(-6, ["vmax","latitude","longitude"])
syn[["vmax_m12"]] = at(-12, ["vmax"])
L = raw.loc[raw.record_id == "L", ["storm_id","datetime"]].rename(columns={"datetime":"tL"})
j = syn.reset_index()[["index","storm_id","datetime"]].merge(L, on="storm_id")
hit = j[(j.tL >= j.datetime) & (j.tL <= j.datetime + pd.Timedelta(hours=24))]["index"].unique()
syn["landfall_24"] = syn.index.isin(hit)
syn["vmax_24"] = pd.to_numeric(syn["vmax_24"])
for c in ["vmax_m6","lat_m6","lon_m6","vmax_m12"]: syn[c] = pd.to_numeric(syn[c])
valid = syn.vmax.notna() & syn.vmax_24.notna() & syn.status.isin(TROP) & syn.status_24.isin(TROP) & ~syn.landfall_24
syn["RI"] = np.where(valid, (syn.vmax_24 - syn.vmax >= 30).astype(float), np.nan)
syn["delta_vmax_6h"] = syn.vmax - syn.vmax_m6; syn["delta_vmax_12h"] = syn.vmax - syn.vmax_m12
syn["translation_speed"] = [point_distance_km(a,b,c,d)/6.0 if pd.notna(a) else np.nan for a,b,c,d in zip(syn.lat_m6, syn.lon_m6, syn.latitude, syn.longitude)]
syn["storm_age_hours"] = syn.groupby("storm_id")["datetime"].transform(lambda x: (x - x.min()).dt.total_seconds()/3600)
syn["month"] = syn.datetime.dt.month
sim = syn.dropna(subset=["RI"] + F9).copy(); sim["RI"] = sim["RI"].astype(int)
sim["sy"] = sim.groupby("storm_id")["datetime"].transform("min").dt.year

def summ(d, lo, hi):
    s = d[(d.sy >= lo) & (d.sy <= hi)]
    return f"rows={len(s):5d}  RI+={int(s.RI.sum()):4d}  rate={s.RI.mean():.3f}  storms={s.storm_id.nunique()}"
print("=== CHECK A: validation split feasibility ===")
for name, d in [("CURRENT", cur), ("SIM d1+d4", sim)]:
    print(f"-- {name}: total rows={len(d)} RI+={int(d.RI.sum())}")
    for lab_, lo, hi in [("train <=2015", 0, 2015), ("ES 2016-2017", 2016, 2017), ("CAL 2018-2019", 2018, 2019), ("VAL 2016-2019", 2016, 2019), ("TEST >=2020", 2020, 9999)]:
        print(f"   {lab_:14s} {summ(d, lo, hi)}")
print("-- per-year RI+ (SIM), 2008-2025:")
print(sim[sim.sy >= 2008].groupby("sy").agg(rows=("RI","size"), pos=("RI","sum")).T.to_string())

print("\n=== CHECK B: eras ===")
eras = [("pre-1944", 0, 1943), ("1944-1969", 1944, 1969), ("1970-1979", 1970, 1979), ("1980+", 1980, 9999)]
for name, d in [("CURRENT", cur), ("SIM d1+d4", sim)]:
    print(f"-- {name}")
    for e, lo, hi in eras: print(f"   {e:10s} {summ(d, lo, hi)}")

# mslp drop: SIM rows that pass label + all other features, split by mslp missing
pre = syn.dropna(subset=["RI"] + [c for c in F9 if c != "mslp"]).copy()
pre["yr"] = pre.groupby("storm_id")["datetime"].transform("min").dt.year
nxt = pre.reset_index()[["index","storm_id","datetime"]].merge(L, on="storm_id")
nxt = nxt[nxt.tL > nxt.datetime].assign(h=lambda x: (x.tL - x.datetime).dt.total_seconds()/3600).groupby("index").h.min()
pre["h_to_L"] = pre.index.map(nxt)
smax = raw.groupby("storm_id").vmax.max(); pre["storm_max_vmax"] = pre.storm_id.map(smax)
def prof(s, tag):
    lf = s.h_to_L
    print(f"   {tag:22s} n={len(s):6d} RI+={int(s.RI.sum()):4d} rate={s.RI.mean():.3f} | vmax q25/50/75={s.vmax.quantile([.25,.5,.75]).tolist()} "
          f"| storm-max vmax med={s.storm_max_vmax.median():.0f} | lat med={s.latitude.median():.1f} "
          f"| landfall later in storm={lf.notna().mean():.2f} | landfall within 48h={(lf<=48).mean():.2f} | med h_to_L={lf.median():.0f}")
print("-- mslp dropna effect (SIM label; rows passing everything except mslp)")
for e, lo, hi in eras[:3] + [("pre-1980 all", 0, 1979), ("1980+", 1980, 9999)]:
    s = pre[(pre.yr >= lo) & (pre.yr <= hi)]
    print(f"  {e}: dropped (mslp NaN)={int(s.mslp.isna().sum())}  kept={int(s.mslp.notna().sum())}  kept%={s.mslp.notna().mean():.1%}")
print("-- profile")
p80 = pre[pre.yr < 1980]
prof(p80[p80.mslp.notna()], "pre-1980 KEPT (mslp)")
prof(p80[p80.mslp.isna()], "pre-1980 DROPPED")
prof(pre[pre.yr >= 1980], "1980+")
print("-- CURRENT-pipeline mslp drop pre-1980 (label valid, other feats ok):")
c2 = extract_features(ri_labels(raw)).dropna(subset=["RI"] + [c for c in F9 if c != "mslp"] + ["day_of_year"])
c2["yr"] = c2.groupby("storm_id")["datetime"].transform("min").dt.year
c2 = c2[c2.yr < 1980]; print(f"  dropped={int(c2.mslp.isna().sum())} kept={int(c2.mslp.notna().sum())}")
```

**Output (2026-09-29):**
```
=== CHECK A: validation split feasibility ===
-- CURRENT: total rows=14528 RI+=868
   train <=2015   rows=11459  RI+= 665  rate=0.058  storms=933
   ES 2016-2017   rows=  667  RI+=  57  rate=0.085  storms=32
   CAL 2018-2019  rows=  616  RI+=  31  rate=0.050  storms=32
   VAL 2016-2019  rows= 1283  RI+=  88  rate=0.069  storms=64
   TEST >=2020    rows= 1786  RI+= 115  rate=0.064  storms=111
-- SIM d1+d4: total rows=12941 RI+=858
   train <=2015   rows=10337  RI+= 656  rate=0.063  storms=884
   ES 2016-2017   rows=  578  RI+=  57  rate=0.099  storms=32
   CAL 2018-2019  rows=  510  RI+=  31  rate=0.061  storms=30
   VAL 2016-2019  rows= 1088  RI+=  88  rate=0.081  storms=62
   TEST >=2020    rows= 1516  RI+= 114  rate=0.075  storms=106
-- per-year RI+ (SIM), 2008-2025:
sy    2008  2009  2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023  2024  2025
rows   233    84   277   266   340   129    97   165   292   286   300   210   365   251   174   341   220   165
pos     24     7    28    11    12     0     2     8    19    38    19    12    33    13     7    10    28    23

=== CHECK B: eras ===
-- CURRENT
   pre-1944   rows=  233  RI+=   4  rate=0.017  storms=152
   1944-1969  rows= 1476  RI+=  87  rate=0.059  storms=219
   1970-1979  rows= 1175  RI+= 101  rate=0.086  storms=98
   1980+      rows=11644  RI+= 676  rate=0.058  storms=639
-- SIM d1+d4
   pre-1944   rows=  199  RI+=   3  rate=0.015  storms=133
   1944-1969  rows= 1334  RI+=  84  rate=0.063  storms=206
   1970-1979  rows= 1120  RI+= 101  rate=0.090  storms=97
   1980+      rows=10288  RI+= 670  rate=0.065  storms=616
-- mslp dropna effect (SIM label; rows passing everything except mslp)
  pre-1944: dropped (mslp NaN)=12384  kept=199  kept%=1.6%
  1944-1969: dropped (mslp NaN)=3640  kept=1334  kept%=26.8%
  1970-1979: dropped (mslp NaN)=1728  kept=1120  kept%=39.3%
  pre-1980 all: dropped (mslp NaN)=17752  kept=2653  kept%=13.0%
  1980+: dropped (mslp NaN)=464  kept=10288  kept%=95.7%
-- profile
   pre-1980 KEPT (mslp)   n=  2653 RI+= 188 rate=0.071 | vmax q25/50/75=[40.0, 60.0, 80.0] | storm-max vmax med=90 | lat med=26.3 | landfall later in storm=0.21 | landfall within 48h=0.08 | med h_to_L=66
   pre-1980 DROPPED       n= 17752 RI+= 569 rate=0.032 | vmax q25/50/75=[40.0, 50.0, 75.0] | storm-max vmax med=85 | lat med=23.7 | landfall later in storm=0.28 | landfall within 48h=0.08 | med h_to_L=77
   1980+                  n= 10752 RI+= 670 rate=0.062 | vmax q25/50/75=[35.0, 50.0, 70.0] | storm-max vmax med=80 | lat med=23.6 | landfall later in storm=0.28 | landfall within 48h=0.10 | med h_to_L=60
-- CURRENT-pipeline mslp drop pre-1980 (label valid, other feats ok):
  dropped=19028 kept=2884
```
