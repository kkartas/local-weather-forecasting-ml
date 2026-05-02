# Methodology Change Log

This file records divergences between the written dissertation methodology and the implemented forecasting pipeline.

Use this file only for changes affecting methodology, experimental design, or scientific assumptions. General implementation notes and missing MetDataPy features belong elsewhere.

## 2026-05-02 - Executable feature set limited by MetDataPy 1.0.0

- Affected component:
  Feature engineering and raw Weathercloud ingestion.
- What changed:
  The executable pipeline uses only data-preparation functionality currently exposed by MetDataPy 1.0.0. Rolling meteorological features and wind-direction cyclic encoding are not locally reimplemented. Robust Weathercloud multi-file, delimiter, and encoding handling are also deferred to MetDataPy.
- Why it changed:
  The dissertation requires MetDataPy to be the authoritative data preparation layer. Reimplementing missing reusable meteorological preparation logic inside this forecasting repository would violate the project architecture.
- Methodology impact:
  The final dissertation methodology remains valid, but full final experiments require the missing MetDataPy APIs documented in `METDATAPY.md`. Interim smoke runs may use a reduced supported feature set.
- Dissertation update required:
  No if MetDataPy is updated before final experiments; yes if final reported experiments use the reduced feature set.
