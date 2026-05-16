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

- [ ] Document when to clean `data/processed/` and `artifacts/` before a train run (same config re-run vs smoke→default vs failed partial run).
- [ ] (optional) Add a `clean` CLI command or `--fresh` flag on `train` / `run-all` to remove generated outputs before writing new ones.

## Documentation

- [ ] (add items as needed)

## Testing

- [ ] (add items as needed)
