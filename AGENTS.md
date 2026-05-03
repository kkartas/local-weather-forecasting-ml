# Agent Contribution Guide

## Project Purpose

This repository implements the practical forecasting pipeline for an MSc dissertation in Computer Science on local short-term weather forecasting from a single local weather station using machine learning and neural networks.

## Dissertation Constraints

- The scientific purpose must remain observation-only, station-local forecasting.
- Do not add Numerical Weather Prediction outputs or external weather forecast products as model inputs unless the dissertation methodology is formally changed and documented.
- Do not silently alter forecast targets, horizons, splits, metrics, or assumptions.
- Any methodology divergence belongs in `CHANGES.md`.

## Mandatory MetDataPy Rule

MetDataPy is the official data preparation layer for this project. Treat it as mandatory, not optional.

MetDataPy owns reusable meteorological time-series preparation logic:

- Weathercloud ingestion where supported
- canonical schema mapping
- unit normalization
- timestamp normalization
- quality-control flagging
- gap detection and insertion
- derived meteorological features
- lag and horizon creation
- rolling and cyclical feature generation where supported
- ML-ready dataset preparation where supported

This repository owns experiment orchestration:

- configuration
- model training
- evaluation
- artifact management
- plotting
- reports
- CLI commands
- dissertation experiment outputs

Never reimplement MetDataPy responsibilities locally. If a required feature is missing or incomplete, document it in `METDATAPY.md` with the expected API and blocking status, then update MetDataPy separately.

## Data Leakage Rules

- Never use future data in training or preprocessing.
- Never shuffle time-series splits.
- Never fit scalers on validation or test data.
- Fit scalers after chronological splitting, using only the training split.
- Rolling or lagged features must not use future observations.
- Validation is for model selection and early stopping; final metrics are reported on test data.

## Modeling Rules

- Train models independently for each target and forecast horizon.
- Compare models under the same dataset, preprocessing logic, split, and metrics.
- Use deterministic random seeds where possible.
- Comment non-trivial ML, preprocessing, sequence, scaling, and time-series logic.
- Document any model family excluded from final experiments in `CHANGES.md` if the dissertation methodology is affected.

## Documentation Rules

- Update `README.md` when commands, configuration, or artifact locations change.
- Update the detailed documentation under `docs/` for every codebase addition or behavior change that affects users, experiments, configuration, data flow, models, metrics, artifacts, or development workflow.
- Keep the GitHub Pages documentation synchronized with the code in the same commit as the code change whenever practical.
- Update `docs/configuration.md` when configuration keys, defaults, or expected values change.
- Update `docs/pipeline-overview.md`, `docs/data-ingestion.md`, `docs/preprocessing-features.md`, `docs/datasets-splits.md`, `docs/training.md`, `docs/evaluation-reporting.md`, or `docs/artifacts.md` when the corresponding pipeline stage changes.
- Update `docs/development.md` when testing, contribution workflow, package structure, or GitHub Pages setup changes.
- Update `METDATAPY.md` for missing MetDataPy functionality.
- Update `CHANGES.md` for methodology, experiment design, or assumption changes.
- Do not use `CHANGES.md` for ordinary implementation notes.
- Do not leave documentation stale after code changes. If documentation does not need updating, say why in the final response or commit context.

## Coding Style

- Use Python 3.10+ compatible code.
- Prefer type hints and dataclasses for configuration and structured objects.
- Use logging for pipeline operations.
- Avoid hardcoded absolute paths.
- Avoid hidden global state.
- Raise clear exceptions for invalid configuration, missing columns, or unavailable required APIs.

## Testing Expectations

Add or update pytest tests for:

- MetDataPy import and integration
- canonical schema expectations
- timestamp and split behavior
- horizon shifting
- scaler fit/apply leakage prevention
- metrics edge cases
- baseline, ML, and DL smoke paths

Tests should use small synthetic fixtures where possible. Real Weathercloud files may be used for integration smoke tests but should not be the only coverage.

## Commit Rules

Use clear, professional commit messages. Conventional commits are preferred, for example:

- `docs: add agent contribution guidelines`
- `feat: add experiment configuration system`
- `test: add time split coverage`

Do not collapse the whole project into one final commit.

## Handling Uncertainty

If ownership of a feature is ambiguous, assume reusable meteorological time-series preparation belongs in MetDataPy. Document the missing feature in `METDATAPY.md` and keep local code as a thin adapter or orchestration layer.
