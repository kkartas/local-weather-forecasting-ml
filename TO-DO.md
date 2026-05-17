# Project TODO

Track dissertation pipeline work here. Check items off as they are completed.

## Logging and observability

- [x] Add structured logging so long runs show clear progress instead of a single "Preparing horizon …" line with a long silent gap.
  - [x] Use ISO timestamps on every log line (CLI now installs an ISO-8601 UTC handler with format `%(asctime)s %(levelname)s %(name)s: %(message)s` and `datefmt=%Y-%m-%dT%H:%M:%SZ`).
  - [x] Log run context at start: config path, `project.name`, resolved horizon list, model families/names (`Run start: ...` and `Train context: ...`).
  - [x] Log **start** and **finish** (and elapsed seconds) for: `ingest`, `preprocess`, supervised build, chronological split, feature-scaler fit, target-scaler fit, `evaluate`/report writing (via `_log_stage`).
  - [x] For each **horizon**: log label and step count before supervised build, split, and scaler steps (`Stage start: horizon h01 steps=6 target=temp_c_t+6`).
  - [x] For each **model**: log **start** and **finish** with `horizon_label`, `model_family`, and `model_name` (`Stage start: train model family=ml model=random_forest horizon=h01 ...`).
  - [x] For DL: log epochs trained, best validation loss, and explicit skip reasons (`Skip model: family=dl model=lstm horizon=h12 reason=min_dl_train_rows ...`).
  - [x] Keep existing warnings; no change to training logic, hyperparameters, or methodology—logging only.

## Experiments and pipeline

- [x] Document when to clean `data/processed/` and `artifacts/` before a train run (same config re-run vs smoke→default vs failed partial run).
- [x] (optional) Add a `clean` CLI command or `--fresh` flag on `train` / `run-all` to remove generated outputs before writing new ones.

## Parallel training (horizon-level)

Long full-config runs are dominated by single-threaded RBF `SVR` (~95k train rows × ~1.6k features per horizon). Train horizons sequentially today; parallelizing **horizons** (not nested horizon×ML at full blast) is the preferred quality-preserving speedup.

- [x] Add **parallel horizon training** to `train` (configurable worker count, default `1` = current behaviour).
  - [x] Each worker runs the full per-horizon pipeline for one `(horizon_label, horizon_steps)`: supervised build, split, scalers, baselines, ML, DL, predictions, and horizon-scoped artifacts (`supervised_<horizon>.parquet`, `scaler_<horizon>.joblib`, `models/*_<horizon>.*`, etc.).
  - [x] Require `prepared.parquet` once up front; do not parallelize ingest/preprocess inside `train`.
  - [x] Cap workers at `min(configured, n_horizons, cpu_count)`; use a process pool (not threads) so sklearn/torch fits stay isolated.
  - [x] When `workers > 1`, set `RandomForestRegressor(n_jobs=1)` (or equivalent) to avoid outer×inner CPU oversubscription; keep sequential-run behaviour unchanged when `workers == 1`.
  - [x] Merge per-horizon metric rows after all workers finish, then run existing `_attach_persistence_skill_score` and `_write_metrics_and_plots` once on the combined table (same outputs as today).
  - [x] Preserve deterministic seeds per horizon/model; document any acceptable float drift (e.g. RF) in `CHANGES.md` only if observed.
  - [x] Log which horizon runs in which worker (`horizon=`, `worker=`, PID) using the existing `_log_stage` / ISO timestamp format.
  - [x] Add pytest coverage: two horizons in parallel on a tiny fixture; assert both horizons’ artifacts and metrics rows exist and match sequential semantics on small data.
- [x] **Documentation (required when implementing the above)** — update in the **same commit** as the code:
  - [x] `docs/training.md` — how parallel horizons work, config key(s), default `workers: 1`, RF `n_jobs` rule, memory/CPU expectations, that SVR still dominates wall time per worker.
  - [x] `docs/running-the-experiment.md` — when to use parallel training, suggested worker count vs machine cores, warning not to nest with a second full `train` on the same artifact dirs.
  - [x] `docs/configuration.md` — new YAML key(s), defaults, and valid values.
  - [x] `README.md` — one-line mention if CLI/config surface changes.
  - [x] `docs/development.md` — point to the new parallel-horizon test(s).
  - [x] Do **not** change dissertation methodology in `CHANGES.md` unless parallel runs produce materially different metrics; if they do, record why and whether results were regenerated. (Not changed; identical metrics on smoke fixture.)
- [ ] (optional, later) Parallel ML models **within** a horizon — only if horizon parallelism stays off or with a global job cap; avoid nested 6×4 process explosion on laptops.

## Deep learning memory (feature selection and batched sequences)

Full-config DL training currently fails with `numpy.core._exceptions._ArrayMemoryError` when
`sequence_arrays_from_split` tries to allocate the full train tensor in one block
(e.g. shape `(95211, 144, 1632)` float32 ≈ **83 GiB** on `m10`). Tabular ML uses the
same ~1.6k `select_feature_columns()` list; DL should not stack every lag column at
every timestep when `sequence_length` already encodes history.

- [x] **DL feature selection** — add `select_dl_feature_columns()` (or equivalent) used only by DL:
  - [x] Include per-timestep signals: canonical variables, derived metrics, calendar cyclic
    features, wind-direction encoding, rolling stats at `t`, and causal `qc_*` flags if desired.
  - [x] **Exclude** `_lag*` columns from DL inputs (redundant with the sequence axis).
  - [x] Keep tabular ML/baselines on the existing wide feature set unchanged.
  - [x] Optionally allow an explicit allow-list in config (e.g. `dl_feature_columns` or
    `dl_exclude_lag_features: true`) for dissertation transparency.
  - [x] Log `n_dl_features` vs `n_tabular_features` at DL train start per horizon/model.
- [x] **Batched / lazy sequence loading** — stop materializing all windows in `sequence_arrays_from_split`:
  - [x] Refactor so PyTorch `DataLoader` (or a small `SequenceDataset`) builds each
    `(sequence_length, n_features)` window on demand from the scaled split frame.
  - [x] Train/val/test iterators must remain leakage-free (windows end at forecast origin `t` only).
  - [x] Preserve existing `batch_size`, early stopping, target scaling, and metric reporting in original units.
  - [x] Add a conservative RAM estimate log before DL fit (approx batch × seq × features) without allocating the full tensor.
- [x] **Tests** — pytest on a tiny fixture: DL path completes without allocating a multi‑GiB array; optional regression test that DL feature count ≪ tabular feature count on synthetic data with lags.
- [x] **Documentation (required when implementing the above)** — same commit as code:
  - [x] `docs/training.md` — DL feature policy vs ML, why lags are excluded, memory expectations.
  - [x] `docs/configuration.md` — any new YAML keys and defaults.
  - [x] `docs/datasets-splits.md` — update `sequence_arrays_from_split` / dataset behaviour if API changes.
  - [x] `docs/running-the-experiment.md` — note RAM requirements dropped; full run no longer needs 80+ GiB for DL.
  - [x] `CHANGES.md` — DL feature set narrowing recorded as a methodology change.
- [ ] (optional) Resume or skip logic so a failed DL step does not force redoing completed SVR/ML artifacts for the same horizon.

## Documentation

- [ ] (add items as needed)

## Testing

- [ ] (add items as needed)
