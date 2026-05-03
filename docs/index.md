# Local Weather Forecasting ML Pipeline

This documentation describes the complete practical pipeline for short-term local weather forecasting from single-station Weathercloud observations.

The project is designed for reproducible experiments. It ingests raw Weathercloud CSV exports, prepares canonical meteorological time-series data, creates supervised forecasting datasets, trains baseline, machine-learning, and deep-learning models, evaluates them on chronological test splits, and writes all outputs under configured artifact directories.

## Main Constraints

- Inputs are historical station observations only.
- Numerical Weather Prediction outputs and external forecast products are not used.
- Time-series rows are split chronologically.
- Scalers are fit on training data only.
- Each forecast horizon is trained and evaluated independently.
- MetDataPy owns reusable meteorological data preparation logic.

## Documentation Map

- [Getting Started](getting-started.md)
- [Configuration](configuration.md)
- [Pipeline Overview](pipeline-overview.md)
- [Data Ingestion](data-ingestion.md)
- [Preprocessing And Feature Engineering](preprocessing-features.md)
- [Supervised Datasets And Splits](datasets-splits.md)
- [Model Training](training.md)
- [Evaluation And Reporting](evaluation-reporting.md)
- [Artifacts](artifacts.md)
- [Development And Testing](development.md)

## Command Summary

```powershell
python -m weather_forecasting_pipeline ingest --config configs/default.yaml
python -m weather_forecasting_pipeline preprocess --config configs/default.yaml
python -m weather_forecasting_pipeline train --config configs/default.yaml
python -m weather_forecasting_pipeline evaluate --config configs/default.yaml
python -m weather_forecasting_pipeline run-all --config configs/default.yaml
```

Use `configs/smoke.yaml` for fast verification.
