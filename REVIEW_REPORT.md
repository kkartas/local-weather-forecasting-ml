# Implementation Review Report

Reviewer role: Senior ML Engineer, Research Software Engineer, and Code Reviewer.
Repository: `local-weather-forecasting-ml` (commit `aba77d6`, branch `master`).
Review date: 2026-05-05.
Environment: Windows 11, Python 3.11.9, MetDataPy 1.2.0, pytest 7.4.0.

## 1. Executive Summary

**Verdict: Acceptable with minor fixes.**

The implementation faithfully follows the dissertation methodology: it forecasts a single station's `temp_c` from observation-only inputs, uses chronological splits, fits scalers on the training partition only, computes past-only lag and rolling features (`closed="left"`), and trains/evaluates baselines, traditional ML, and deep-learning models per horizon. MetDataPy is treated as the data-preparation authority, the adapter layer is thin, and `METDATAPY.md` documents missing/incomplete MetDataPy capabilities.

Notable findings that should be addressed before final dissertation runs:
- A real but small look-ahead leakage from MetDataPy QC features (`qc_*_spike`, `qc_*_flatline`) computed with `center=True` rolling windows in MetDataPy 1.2.0; these flags are currently passed to models as features.
- `optional_horizons` defined in `configs/default.yaml` (`m10`, `h24`) are silently ignored by the training loop; only `horizons` is iterated.
- DL models are trained on scaled features but unscaled targets, which slows convergence and degrades comparability across horizons.
- README install instructions assume `pip install -e .`; without it, `python -m weather_forecasting_pipeline ...` fails with `No module named ...`.
- DL models in [src/weather_forecasting_pipeline/training/pipeline.py:209-215](src/weather_forecasting_pipeline/training/pipeline.py#L209-L215) ignore `train_dl_model`'s training-loop `shuffle=False` over chronological windows — that is correct — but the docstring/report claim of "validation early stopping" matches behavior; nonetheless, the smoke result demonstrates DL was effectively unfit for one-epoch smoke runs.
- A few documentation/integrity issues (gitignore line `/AGENTS.md`, missing `pip install -e .` in CI/test path, `data/raw/smoke_weathercloud.csv` is committed even though `data/raw/*` is gitignored — protected only by an unstated history exception).

There is **no critical leakage** and **no random splitting**; the methodology is intact. With the minor fixes below the implementation is ready for Chapter 4 experimental runs.

## 2. Dissertation Methodology Alignment

**Aligned**
- Single-station, observation-only inputs; only Weathercloud CSVs are read (`source: weathercloud` enforced in [config.py:109-110](src/weather_forecasting_pipeline/config.py#L109-L110)).
- No NWP or external forecast inputs anywhere in the pipeline.
- Chronological train/validation/test split via `metdatapy.mlprep.time_split` with fraction-derived boundaries ([metdatapy_adapter.py:126-145](src/weather_forecasting_pipeline/metdatapy_adapter.py#L126-L145)).
- Fraction sums to 1.0 enforced ([config.py:103-106](src/weather_forecasting_pipeline/config.py#L103-L106)).
- Scaler fit on `splits["train"]` features only ([metdatapy_adapter.py:148-168](src/weather_forecasting_pipeline/metdatapy_adapter.py#L148-L168)).
- Independent training per `(target, horizon)` pair ([pipeline.py:99-162](src/weather_forecasting_pipeline/training/pipeline.py#L99-L162)).
- Required model families present: persistence, moving-average, linear regression, random forest, gradient boosting, SVR, LSTM, GRU, TCN.
- MAE, RMSE, and safe MAPE implemented and reported per (model, horizon).
- Outputs (CSV, JSON, Markdown summary, plots, model & scaler artifacts) all produced.

**Deviations / Soft mismatches**
- `optional_horizons` is parsed from the YAML but never iterated; horizons in that section are silently skipped. `default.yaml` puts `m10` and `h24` there, so they will not run unless moved into `horizons`. This breaks the "multi-horizon" claim for the default config.
- The dissertation's "moving average" baseline is implemented as the mean of the first 4 lag columns sorted by lag number ([baselines.py:36-44](src/weather_forecasting_pipeline/models/baselines.py#L36-L44)). With default lags `[1, 3, 6, 12, 24, 72, 144]` it averages `t-1, t-3, t-6, t-12` — i.e., a **non-consecutive** rolling average. A standard MA over `t-1..t-k` would be more defensible; document the choice or change the implementation.
- DL models receive **scaled features but unscaled target** (target column is excluded from `scale_columns`). With Adam + MSE this is workable but slows convergence and makes hyperparameters horizon-dependent. The dissertation should either scale the target and inverse-scale predictions or document the choice.
- Persistence baseline computes prediction from the unscaled `splits["test"]` while ML/DL predictions come from the scaled split; this is fine because the target column is unscaled in both, but the asymmetry is worth a comment.

## 3. MetDataPy Compliance

MetDataPy is correctly treated as the data-preparation authority. The adapter ([metdatapy_adapter.py](src/weather_forecasting_pipeline/metdatapy_adapter.py)) is thin and delegates:
- Weathercloud directory ingestion → `metdatapy.weathercloud.read_weathercloud_directory`
- canonical mapping & timezone normalization → MetDataPy `Mapper` + ts.timezone
- gap insertion, QC, derived variables, calendar features, wind cyclic encoding, rolling features → `WeatherSet` methods
- supervised dataset creation → `metdatapy.mlprep.make_supervised`
- chronological boundary split → `metdatapy.mlprep.time_split`
- scaler fit/apply → `metdatapy.mlprep.fit_scaler` / `apply_scaler`
- IO → `metdatapy.io.read_parquet` / `to_parquet`

No MetDataPy responsibility is duplicated locally except:
1. Duplicate-timestamp drop after ingestion ([metdatapy_adapter.py:58-62](src/weather_forecasting_pipeline/metdatapy_adapter.py#L58-L62)) — **already documented** in [METDATAPY.md](METDATAPY.md) ("Duplicate timestamp handling policy").
2. Fraction → boundary conversion before calling `time_split` ([metdatapy_adapter.py:126-145](src/weather_forecasting_pipeline/metdatapy_adapter.py#L126-L145)) — **already documented** in METDATAPY.md ("Fraction-based chronological split utility").
3. The DST-safe ingestion gap is **already documented** in METDATAPY.md.

`METDATAPY.md` is well structured (active issues + resolved-in-version sections) and actively maintained. **One additional missing-feature should be added** (see §4): MetDataPy QC functions use `center=True` rolling windows; a `causal=True` option is required so QC features are leakage-safe when used as model inputs.

## 4. Data Leakage Assessment

| Severity | Finding | Evidence |
|----------|---------|----------|
| **Medium** | QC spike/flatline features use centered rolling windows | `metdatapy/qc.py` lines 123 / 187 use `center=True`. The 33 `qc_*` boolean columns end up in `feature_columns` ([split_metadata_h01.json:14-37](data/processed/split_metadata_h01.json#L14-L37)) and reach all ML and DL models. With window=9 (spike) / 5 (flatline) this implies a ~4-step / ~2-step look-ahead. At 10-min cadence ≈ 40 / 20 minutes of future information per row. Information content is small (booleans, mostly `False`) but it does technically violate "no future data in features." |
| Low | Duplicate-timestamp policy applied **after** ingestion only keeps first ([metdatapy_adapter.py:58-62](src/weather_forecasting_pipeline/metdatapy_adapter.py#L58-L62)) | Acceptable, documented in METDATAPY.md, deterministic. |
| None | Random / shuffled split | Confirmed absent. `train_dl_model` uses `DataLoader(..., shuffle=False)` ([dl_models.py:109](src/weather_forecasting_pipeline/models/dl_models.py#L109)). |
| None | Scaler fit on test/val | Verified: `fit_scaler(splits["train"][scale_columns], ...)` only. Test confirms (`test_chronological_split_and_scaler_fit_only_train`). |
| None | Lag features using future values | Verified empirically: `temp_c_lag1` at row 1 equals `temp_c` at row 0; `make_supervised` uses positive lags. |
| None | Rolling features using future values | Verified empirically: `temp_c_roll6_mean` at row 6 equals mean of rows 0–5 (excludes current and future). |
| None | Target leakage through horizon shift | Target `temp_c_t+6` at row 0 equals `temp_c` six rows later. Confirmed. |
| None | `_t+` columns leaking into features | `select_feature_columns` excludes any column containing `_t+` ([splits.py:33-45](src/weather_forecasting_pipeline/datasets/splits.py#L33-L45)). |
| None | Validation/test info during preprocessing | Preprocessing is deterministic (gap insertion, QC, derived variables, calendar, encoding, rolling) and does not fit any parameters; only the QC center-window issue above touches future data. |

**Recommended Fix for the Medium finding:** Either (a) drop QC flag columns from the feature set in `select_feature_columns` (2-line filter) and document this in CHANGES.md, or (b) request `causal=True` in MetDataPy `qc_spike`/`qc_flatline` and add an entry under "Active Missing Or Incomplete Features" in `METDATAPY.md`. Until MetDataPy supports causal QC, option (a) is the safest local mitigation that respects the "no MetDataPy reimplementation" rule.

## 5. Time-Series and Forecasting Correctness

- Timestamps parsed as `Date (Europe/Athens)`, localized to Europe/Athens, converted to UTC by MetDataPy mapping (`ts.timezone: Europe/Athens`). Verified by `test_metdatapy_mapping_timezone_normalizes_local_time` (`2024-01-01 00:00 Athens → 2023-12-31 22:00 UTC`).
- Sort order enforced (`df.sort_index()` after ingestion, monotonic check in `split_by_fraction_with_metdatapy`).
- Duplicate timestamps logged + dropped (keep first); deterministic but should move into MetDataPy.
- Missing timestamps inserted via `WeatherSet.insert_missing(expected_frequency)` and flagged with `gap`.
- DST handling: known limitation tracked in METDATAPY.md (High priority for full-year exports).
- Forecast horizon shift via `make_supervised(targets=[target], horizons=[h])` produces `temp_c_t+h` correctly.
- Lag and rolling features verified past-only.
- Sequence construction in `sequence_arrays_from_split` is leakage-free: for end position `e`, the sequence is rows `[e-seq+1 .. e]` and the label is `y_tab[e]` which corresponds to the already-shifted MetDataPy target at `t+h` — i.e., features come strictly from observations at or before forecast origin `e`.
- Test-set timestamp alignment for DL predictions: `scaled_splits["test"].index[-len(y_test):]` correctly maps to forecast origins (verified by sequence count: 52 test rows, sequence_length 12 → 41 sequences, indices `[-41:]`).

## 6. Model Implementation Review

| Family | Model | Implementation | Notes |
|--------|-------|----------------|-------|
| Baseline | persistence | OK | Uses current `temp_c` as forecast for any horizon. |
| Baseline | moving_average | OK but unusual | Averages first 4 lag columns sorted by lag number — non-consecutive. With default lags this is `mean(t-1, t-3, t-6, t-12)`. |
| ML | linear_regression | OK | Default `LinearRegression`. |
| ML | random_forest | OK | `n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=-1`. |
| ML | gradient_boosting | OK | sklearn defaults + `random_state=seed`. |
| ML | svr | OK | `RBF, C=10, epsilon=0.1`. |
| DL | lstm | OK | Single-layer LSTM 64-hidden + 32→1 head, MSE loss, Adam, early stopping on validation. |
| DL | gru | OK | Same architecture as LSTM with GRU cell. |
| DL | tcn | OK | 3 dilated causal blocks (dilation 1/2/4) with residuals; final timestep linear head. |

Observations:
- DL `RecurrentRegressor.head` is `Linear(hidden, 32) → ReLU → Linear(32, 1)` and consumes `output[:, -1, :]` — final-step semantics are correct for next-h forecasting given that the target is the already-shifted `temp_c_t+h`.
- TCN block padding/truncation (`y[..., :x.size(-1)]`) preserves causal alignment for the last timestep.
- Random seeds: `set_random_seed` covers Python, NumPy, and PyTorch (CPU + CUDA + cudnn deterministic). Seeds for sklearn estimators are passed via `random_state=seed`. SVR has no randomness.
- DL receives unscaled targets — see §2 deviation. Recommend adding optional target scaling for DL only.
- DL training writes `epochs_trained` and `best_validation_loss` to the metrics CSV — good.
- All models save artifacts (`.joblib` / `.pt`) per (model, horizon). Per-row predictions saved as `predictions_<model>_<horizon>.csv` for reproducibility.

## 7. Evaluation and Metrics Review

- MAE: `mean(|y_true - y_pred|)` ✓ unit-tested and matches `1.0` on `[1,2,3] vs [1,4,2]`.
- RMSE: `sqrt(mean((y_true - y_pred)**2))` ✓ unit-tested (`sqrt(5/3)`).
- Safe MAPE: masks `|y_true| <= epsilon`; returns `None` when fully masked. ✓ unit-tested with mixed and zero arrays.
- `evaluate_predictions` returns a dict containing all three metrics plus aligned/finite filtering ([metrics.py:35-52](src/weather_forecasting_pipeline/evaluation/metrics.py#L35-L52)).
- Metrics computed only on the test partition; no test-set use during training/early stopping (early stopping uses `splits["val"]`).
- Exports verified in `artifacts/metrics/`: `metrics.csv` (one row per model/horizon), `metrics.json`, `artifacts/reports/summary.md`.
- Plots produced per the smoke run: model comparison (MAE), error-by-horizon, actual-vs-predicted (top 4 prediction files), residual histograms.
- Plot environment compatibility check (`numpy>=2 + matplotlib<3.8`) prevents a known binary incompatibility — pragmatic.

## 8. Code Quality Review

- Project structure is clear and idiomatic (`src/` layout, dataclass-based config, dedicated `models/`, `evaluation/`, `datasets/`, `plotting/`, `training/` packages).
- Type hints throughout, frozen dataclasses for config sections.
- Logging via the standard `logging` module; CLI sets level from `--log-level`.
- No hardcoded absolute paths; YAML paths are anchored relative to the parent of the `configs/` directory ([config.py:101](src/weather_forecasting_pipeline/config.py#L101)).
- Docstrings present and concise; comments explain non-obvious logic (sequence construction, TCN axis transpose).
- Exception handling: `_require` validates required keys; explicit errors for unknown models, unsupported source, empty splits, missing target column.
- One robustness wart: the package is consumed by `python -m weather_forecasting_pipeline ...`, which requires either `pip install -e .` or `PYTHONPATH=src`. Tests use `pythonpath = ["src"]` from `pyproject.toml`. The README does mention `pip install -e .`, but most error paths point to the same fix; not a code defect.
- Minor: `select_feature_columns` does not exclude boolean / QC flag columns, so 33 `qc_*` and `gap` flags end up as features. See §4 for the leakage implication.
- Minor: `_scaler_to_dict` returns `{"repr": repr(scaler)}` for non-dataclass scalers — the joblib artifact then loses the actual fitted parameters, which is misleading because the file is named `scaler_*.joblib` but contains only metadata, not a usable scaler. Consider `joblib.dump(scaler, ...)` directly so the scaler can be reloaded for inference.

## 9. Test Suite Review

**Existing (11 tests, all pass):**
- `test_metrics.py` — MAE, RMSE, safe-MAPE correctness and zero-handling.
- `test_datasets.py` — horizon target shifting alignment; sequence-array past-only ordering.
- `test_splits.py` — chronological split monotonicity; scaler fitted only on train.
- `test_models.py` — persistence + linear regression smoke; GRU end-to-end smoke.
- `test_metdatapy_integration.py` — MetDataPy import; semicolon directory ingestion + UTC index + Athens-to-UTC conversion; preprocess gap/QC/calendar/derived/cyclic/rolling.

**Missing tests recommended before final runs:**
1. **DL leakage regression:** assert that for a synthetic ramp, `sequence_arrays_from_split` never includes a feature value strictly later than the forecast-origin timestamp.
2. **QC-flag leakage check:** with a known-good synthetic series, verify either (a) `qc_*` columns are excluded from `select_feature_columns`, or (b) compute and assert the magnitude of the temporal leakage.
3. **CLI smoke test:** invoke `weather_forecasting_pipeline.cli.main(["run-all", "--config", "configs/smoke.yaml"])` from a temp dir and assert artifacts are written.
4. **Optional-horizons handling:** add an assertion (or a config-validator test) that `optional_horizons` either runs or is rejected with a clear error.
5. **Scaler artifact integrity:** load `scaler_h01.joblib` and assert it contains the fitted parameters (currently it only stores `repr(scaler)` for non-dataclass scalers).
6. **Per-horizon independence:** verify predictions for horizon `h01` don't change when `h12` is added to the config (no cross-horizon contamination).
7. **Persistence/MA on baseline split:** assert that `splits["train"]` is unscaled (it is) and that baseline predictions are in original units.

## 10. Documentation Review

- `README.md` is concise and accurate. Install steps include `pip install -e .` (good), but the project name in `pyproject.toml` is `weather-forecasting-pipeline` while the importable package is `weather_forecasting_pipeline`. The README makes this clear by example.
- `AGENTS.md` is excellent: dissertation constraints, MetDataPy mandate, data-leakage rules, modeling rules, documentation rules, coding style, testing expectations, commit rules.
- `CHANGES.md` correctly limits itself to methodology divergences (one entry: feature-set follows installed MetDataPy capabilities).
- `METDATAPY.md` is well organized (active vs. resolved-by-version) and currently captures three active gaps and several resolved items.
- `docs/` (GitHub Pages) covers all stages: pipeline overview, ingestion, preprocessing/features, datasets/splits, training, evaluation, artifacts, configuration, development. Cross-references match the code.
- Missing: a short note in `docs/preprocessing-features.md` (or `docs/datasets-splits.md`) acknowledging the QC-feature centered-window caveat.
- Missing: a note about `optional_horizons` semantics — either how to enable them or why they exist as a separate section.

## 11. Git History Review

- 16 commits, conventional-commit style (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- Repository was initialized with structure first (`b25797d chore: initialize project structure`), then capabilities were added incrementally: MetDataPy integration → datasets/metrics/models → orchestration/reporting → tests → smoke config → entry point → robustness fixes → MetDataPy 1.1/1.2 upgrades → docs.
- No giant single "implementation" commit; history reflects real evolution.
- Two minor cleanups recommended:
  - `.gitignore` line `/AGENTS.md` is inert because `AGENTS.md` is already tracked, but it is misleading. Remove.
  - `data/raw/smoke_weathercloud.csv` is committed even though `data/raw/*` is gitignored. This is fine because Git tracks files that were added before the ignore rule, but the intent is unclear; either explicitly allow `!data/raw/smoke_weathercloud.csv` (so contributors don't accidentally remove it) or move the smoke fixture under `tests/fixtures/`.
- Untracked files in working tree: `.claude/`, `CLAUDE.md` — local Claude Code state, ignore.

## 12. Reproducibility Check

Commands actually run in this review:

| # | Command | Result |
|---|---------|--------|
| 1 | `python -m pytest -v` | **11 passed** in 5.38 s |
| 2 | `pip show metdatapy` | `Version: 1.2.0` (matches `requirements.txt`) |
| 3 | `python -m weather_forecasting_pipeline run-all --config configs/smoke.yaml` | **Failed** with `No module named weather_forecasting_pipeline` because the package was not installed editably in this fresh environment |
| 4 | `PYTHONPATH=src python -m weather_forecasting_pipeline run-all --config configs/smoke.yaml` | **Succeeded**: ingested 360 rows, prepared 360 rows, trained baselines + linear_regression + GRU for `m10` and `h01`, wrote metrics/plots/reports |
| 5 | Verified artifacts (`artifacts/metrics/`, `artifacts/models/`, `artifacts/plots/`, `artifacts/reports/`, `data/processed/`) — all present |
| 6 | Spot-checks on supervised dataset: target shift `temp_c_t+6` aligns with future temp; lag features past-only; rolling mean past-only with `closed="left"` | **All correct** |

Reproducibility from a clean checkout therefore depends on `pip install -e .` (documented in README) **or** running with `PYTHONPATH=src`. The README path is correct; the only friction is that running pytest works without `pip install -e .` (because of `pyproject.toml`'s `pythonpath = ["src"]`), but the CLI does not — a slight discoverability hazard for a new developer.

## 13. Issues Found

| Priority | Area | Issue | Evidence | Recommended Fix |
|----------|------|-------|----------|-----------------|
| Medium | Data leakage | QC `qc_*_spike` and `qc_*_flatline` features use `center=True` rolling windows in MetDataPy 1.2.0; these flags are passed to all models as features, leaking ~2–4 future steps per row. | `metdatapy/qc.py:123` and `:187` use `center=True`; `data/processed/split_metadata_h01.json` lists 33 `qc_*` columns under `feature_columns`. | Either (a) exclude `qc_*` and `gap` boolean columns from `select_feature_columns`; or (b) request `causal=True` in MetDataPy QC and document in `METDATAPY.md`. Preferred: (a) immediately + (b) tracked in METDATAPY.md. |
| Medium | Methodology / config | `optional_horizons` parsed but never trained; `default.yaml` puts `m10` and `h24` there, so they are silently skipped. | `train()` iterates only `config.data.horizons` ([pipeline.py:99](src/weather_forecasting_pipeline/training/pipeline.py#L99)); `default.yaml` lines 23–25. | Either iterate `{**horizons, **optional_horizons}`, or remove `optional_horizons` and document a single horizons block. |
| Medium | Modeling | DL models trained on scaled features but unscaled targets; convergence is target-magnitude dependent. Smoke run shows GRU MAE ≈ 13.7 (vs persistence 0.10) due to single-epoch convergence. | `fit_apply_scaler_with_metdatapy` excludes target from `scale_columns`; `arrays_from_split` reads the unscaled target from `scaled_splits["test"]`. | For DL only, scale the target with the same training-only scaler and inverse-transform predictions before metric computation. |
| Medium | Artifact integrity | `scaler_*.joblib` files stored as `{"repr": repr(scaler)}` for non-dataclass scalers — not reloadable for inference. | [pipeline.py:373-376](src/weather_forecasting_pipeline/training/pipeline.py#L373-L376). | Persist the scaler object directly via `joblib.dump(scaler, ...)`. |
| Low | Methodology / baselines | `MovingAverageModel` averages the first 4 lag columns sorted by lag number → non-consecutive average like `mean(t-1, t-3, t-6, t-12)`. | [baselines.py:36-44](src/weather_forecasting_pipeline/models/baselines.py#L36-L44). | Document explicitly, or change to a true rolling mean over `t-1..t-k` consecutive observations. |
| Low | Reproducibility | CLI requires `pip install -e .` or `PYTHONPATH=src`. Tests work without install (via pyproject `pythonpath`), CLI does not — discoverability hazard. | Reproducing step (3)/(4) above. | Add a note to README "Quick test without install" + "Running the CLI requires `pip install -e .`". Consider adding a smoke CI job that `pip install -e .` then runs the CLI. |
| Low | Repo hygiene | `.gitignore` line `/AGENTS.md` is inert but misleading; `data/raw/smoke_weathercloud.csv` is committed despite `data/raw/*` ignore. | `.gitignore` last and middle sections; `git ls-files` shows both. | Remove the `/AGENTS.md` line; add an explicit `!data/raw/smoke_weathercloud.csv` allow-list **or** move the fixture under `tests/fixtures/`. |
| Low | Documentation | No mention of the QC centered-window caveat in `docs/preprocessing-features.md`; no mention of `optional_horizons` semantics anywhere. | Reading [docs/preprocessing-features.md](docs/preprocessing-features.md) and [docs/configuration.md](docs/configuration.md). | Add a one-paragraph caveat to each. |
| Low | Test coverage | No CLI test, no QC-leakage regression, no scaler-artifact reload test, no per-horizon independence test. | `tests/` listing. | See §9 recommended tests. |

No **Critical** or **High** issues were found.

## 14. Recommended Fix Plan

- [ ] Exclude `qc_*` and `gap` boolean columns from `select_feature_columns` and add a short comment explaining why.
- [ ] Open a "QC causal-window option" entry in `METDATAPY.md` and reference it from `docs/preprocessing-features.md`.
- [ ] Decide and document the `optional_horizons` semantics — either iterate them in `train()` and document, or drop the section and put everything under `horizons`.
- [ ] Replace the `_scaler_to_dict` indirection with `joblib.dump(scaler, ...)` so saved scalers are reloadable.
- [ ] (Optional but recommended) Add target-only scaling for DL models with inverse transform before metric computation.
- [ ] Document or change the `MovingAverageModel` lag-selection behavior.
- [ ] Add the missing tests listed in §9 (at least: CLI smoke, QC-leakage regression, scaler-reload).
- [ ] Clean up `.gitignore` (`/AGENTS.md`) and either explicitly allow `data/raw/smoke_weathercloud.csv` or move it under `tests/fixtures/`.
- [ ] Add a "Reproducing without `pip install -e .`" hint to `README.md`.

## 15. Dissertation Updates Needed

- **No methodology rewrite required.** The implementation matches the dissertation methodology (observation-only, single-station, multi-horizon, chronological split, baseline + ML + DL comparison, MAE/RMSE/safe-MAPE).
- **Mention** in the methodology chapter that QC flags from MetDataPy 1.2.0 use centered rolling windows and were therefore excluded from / carefully handled in the model feature set (after applying the §13 Medium fix). A 2-3 line note keeps the dissertation defensible against a leakage challenge.
- **Mention** the `MovingAverageModel` definition explicitly (which lag indices are averaged) so the baseline is reproducible from the dissertation text alone.
- **Clarify** in the methodology chapter the DL target-scaling decision (whichever way it is finally implemented).
- **DST handling** is currently flagged in `METDATAPY.md` as High priority; if final experiments include the spring DST transition day, mention how DST-transition rows were treated.

## 16. Final Verdict

The implementation is **methodologically valid** and **ready for experimental runs and dissertation Chapter 4 results** subject to the small fixes in §14. Tests pass (11/11), the smoke experiment runs end-to-end, MetDataPy is correctly used as the data-preparation authority, and the data-leakage posture is sound except for the Medium-severity QC-flag center-window issue, which is locally fixable with a 2-line filter. With the recommended fixes applied, the repository is acceptable for full-scale yearly Weathercloud runs.

**Smoke experiment outcome:** 360 input rows ingested, 340 supervised rows after lag/horizon, two horizons trained (m10, h01), 4 models per horizon (persistence, moving_average, linear_regression, gru). Persistence dominates at this synthetic-data size (MAE 0.10 at m10 / 0.60 at h01); linear_regression beats persistence at h01 (MAE 0.24); GRU underfits because smoke runs only 1 epoch with patience 1 (expected).

**Tests:** 11 passed, 0 failed, 0 skipped.
**Smoke experiment:** completed end-to-end; all artifacts produced.
**Issues by priority:** Critical 0 / High 0 / Medium 4 / Low 5.
**Methodologically acceptable:** Yes, with minor fixes.
**Report location:** `REVIEW_REPORT.md` at the repository root.
