# Project TODO

Track dissertation pipeline work here. Check items off as they are completed.

## Logging and observability

- [ ] Add structured logging so long runs show clear progress instead of a single "Preparing horizon …" line with a long silent gap.
  - [ ] Use ISO timestamps on every log line (update CLI `logging.basicConfig` format, e.g. `%(asctime)s %(levelname)s:%(name)s:%(message)s`).
  - [ ] Log run context at start: config path, `project.name`, resolved horizon list, model families/names.
  - [ ] Log **start** and **finish** (and elapsed seconds) for: `ingest`, `preprocess`, supervised build, chronological split, feature-scaler fit, target-scaler fit, `evaluate`/report writing.
  - [ ] For each **horizon**: log label and step count before supervised build, split, and scaler steps (not only one generic "Preparing horizon …" line).
  - [ ] For each **model**: log **start** and **finish** with `horizon_label`, `model_family`, and `model_name` (e.g. `baseline/persistence`, `ml/random_forest`, `dl/lstm` for `h01`).
  - [ ] For DL: log epochs trained, best validation loss, and explicit skip reasons (`min_dl_train_rows`, insufficient sequences after windowing).
  - [ ] Keep existing warnings; do not change training logic, hyperparameters, or methodology—logging only.

## Experiments and pipeline

- [ ] Document when to clean `data/processed/` and `artifacts/` before a train run (same config re-run vs smoke→default vs failed partial run).
- [ ] (optional) Add a `clean` CLI command or `--fresh` flag on `train` / `run-all` to remove generated outputs before writing new ones.

## Documentation

- [ ] (add items as needed)

## Testing

- [ ] (add items as needed)
