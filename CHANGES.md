# Methodology Change Log

This file records divergences between the written dissertation methodology and the implemented forecasting pipeline.

Use this file only for changes affecting methodology, experimental design, or scientific assumptions. General implementation notes and missing MetDataPy features belong elsewhere.

## 2026-05-02 - Executable feature set follows installed MetDataPy capabilities

- Affected component:
  Feature engineering and raw Weathercloud ingestion.
- What changed:
  The executable pipeline uses only data-preparation functionality currently exposed by MetDataPy. After updating to MetDataPy 1.2.0, Weathercloud directory ingestion, delimiter/encoding handling, timezone-aware source mapping, `rain_rate_mmh`, wind-direction cyclic encoding, and rolling features are used directly.
- Why it changed:
  The dissertation requires MetDataPy to be the authoritative data preparation layer. Reimplementing missing reusable meteorological preparation logic inside this forecasting repository would violate the project architecture.
- Methodology impact:
  The final dissertation methodology remains valid. MetDataPy 1.2.0 removes the previous reduced-feature limitation for Weathercloud ingestion and the default feature-engineering set.
- Dissertation update required:
  No.
