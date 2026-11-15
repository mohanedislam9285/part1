# PM2.5 Data Audit, EDA and Fold Preparation

This repository contains the data-preparation stage of a PM2.5 study:

- read and validate the configured source file;
- clean the daily PM2.5 target under explicit rules;
- audit station identity, missingness, distributions and temporal coverage;
- reserve an independent future holdout;
- write temporal, station-grouped and diagnostic fold definitions;
- check seven-day sequence availability against the temporal folds.

The repository stops at validated, leakage-aware inputs. It produces no predictions and reports no performance scores.

> **Data licensing:** the compiled Kaggle bundle is labelled *Other (specified in description)*, and its redistribution terms are not explicit. Use it for local academic analysis only until source-by-source attribution and redistribution rights are confirmed. See [`data/manifest.yaml`](data/manifest.yaml).

## What the saved evidence shows

These figures come from the saved artifacts under [`results/`](results/) and must be reproduced from the source file before being cited as new results.

| Item | Recorded value |
|---|---|
| Source rows | 41,718 |
| Cleaned station-days | 40,983 |
| Stations after cleaning | 95 |
| Date span | 2016-11-25 to 2026-04-06 (UTC) |
| Removed: target ≤ 0 | 727 |
| Removed: target ≥ 500 | 8 |
| Duplicate station-days | 0 |
| Composite station identities before target cleaning | 98 |
| Distinct source sensor names | 94 |
| Cleaned PM2.5 median / mean | 7.85 / 9.46 µg/m³ |
| Cleaned PM2.5 P95 / maximum | 21.65 / 484.98 µg/m³ |
| Development partition | 30,651 rows, 73 stations, ending 2024-12-07 |
| Development stations with an uninterrupted run ≥ 14 days | 63 |
| Development stations with an uninterrupted run ≥ 30 days | 6 |

Cleaning rule: retain only `0 < PM2.5 < 500 µg/m³`. Rows outside that interval are removed as sentinel or invalid values, not as statistical outliers; high but valid values are kept. The recorded raw and cleaned counts are in [`results/part1_evidence_summary.json`](results/part1_evidence_summary.json).

The saved summary also records the holdout status. The `folds` command writes a fresh reservation with status `reserved`; the holdout period is excluded from every development fold.

## Data source and placement

The configured source is `data/raw/extracted/combined_gee_pm25.csv`, from the Kaggle dataset [Air Quality – Mediterranean Basin (2016–2026)](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026). The raw CSV and the upstream `pm25.csv` / `gee_features_daily.csv` files are **not included** in this repository.

1. Obtain the dataset from the Kaggle page and review its terms.
2. Place `combined_gee_pm25.csv` at `data/raw/extracted/combined_gee_pm25.csv`.
3. Confirm its SHA-256 matches the value recorded in [`data/manifest.yaml`](data/manifest.yaml). The audit refuses to run if the checksum or the manifest fields are missing.
4. If you also have `pm25.csv` and `gee_features_daily.csv`, keep them in the same folder to run the optional upstream join audit.

## Setup on Windows

Python 3.12 is the recorded environment; the package metadata allows 3.11 and 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e src
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window only, then activate again.

## Commands

Run these from the repository root, in order.

```powershell
# 1. Validate the configuration.
python -m pm25_project.cli validate-config --config src/configs/project.yaml

# 2. Read, validate, clean and audit the source CSV.
python -m pm25_project.cli audit --config src/configs/project.yaml

# 3. Write exploratory tables, diagnostics and figures.
python -m pm25_project.cli eda --config src/configs/project.yaml

# 4. Reserve the future holdout and write fold definitions.
python -m pm25_project.cli folds --config src/configs/project.yaml

# 5. Check seven-day sequence availability against the temporal folds.
python -m pm25_project.cli sequence-coverage --config src/configs/project.yaml
```

`audit` writes `data/processed/cleaned_daily.csv`, `data/processed/audit.json`, and the station identity reconciliation files. `eda` reads the cleaned CSV, so run `audit` first. `sequence-coverage` requires the fold files written by `folds`.

### Optional scripts

```powershell
# Audit + EDA + the two template summary figures.
python scripts/run_eda.py

# Development-only value screening; requires audit and folds.
python scripts/audit_cleaning_quality.py

# Report evidence tables and figures; also requires the upstream audit below.
python scripts/generate_part1_report_evidence.py

# Row-level join check; runs only when pm25.csv and gee_features_daily.csv exist.
python scripts/audit_upstream_join.py
```

Each script derives its paths from `src/configs/project.yaml` and stops with an actionable message when a prerequisite is missing.

### Notebook

[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) reproduces the exploratory tables and charts from the cleaned panel. It expects `audit` to have run first.

## Cleaning and validation rules

- **Timestamps** are converted to UTC and floored to the day. Naive timestamps require an explicit `data.timestamp_timezone`; mixed timezone formats are rejected.
- **Target** must be numeric and strictly between the configured exclusive bounds (0 and 500 µg/m³).
- **Duplicate station-days** are an error by default. The project does not silently aggregate them.
- **Station identity** is the sensor name plus latitude and longitude rounded to six decimals. This keeps nearby sites with the same sensor name separate. The audit reports how many composite identities collapse only after target cleaning.
- **Source manifest** requires a URL, title, retrieval date, license note and a matching SHA-256 for the configured file.
- **Fold-local preprocessing:** `SequencePreprocessor` fits median imputation and scaling on a fitting partition only. It is checked by the test suite and is not applied to the saved panel.

## Fold design

`folds` first reserves the latest consecutive dates as an independent future holdout, then builds folds on the earlier development partition:

| File | Scheme | Purpose |
|---|---|---|
| `folds/folds_temporal.json` | Expanding temporal folds | Primary development fold family; five folds with a one-day embargo |
| `folds/folds_spatial.json` | Station-grouped folds | Held-out station groups |
| `folds/folds_standard.json` | Random diagnostic folds | Optimism diagnostic only |
| `results/future_holdout.json` | Independent future holdout | Reserved before development work; never used in the folds |

Fold sizes, date boundaries and embargo days are recorded in each JSON file. The holdout uses `project.holdout_fraction` (0.2) and `project.holdout_min_days` (30) from the configuration.

## Artifacts

| Path | Contents |
|---|---|
| [`results/Exploratory_Data.csv`](results/Exploratory_Data.csv) | Per-variable descriptive and missingness table |
| [`results/Exploratory_temporal.csv`](results/Exploratory_temporal.csv) | Per-station cadence, ACF/PACF, stationarity and trend diagnostics |
| [`results/station_identity_reconciliation.csv`](results/station_identity_reconciliation.csv) | Sensor name and coordinate mapping for each composite station |
| [`results/station_identity_reconciliation_summary.json`](results/station_identity_reconciliation_summary.json) | Reconciliation counts |
| [`results/eda/`](results/eda/) | Daily coverage, missingness, monthly trend, descriptive statistics and station summary |
| [`results/correlations.csv`](results/correlations.csv) | Pairwise Pearson and Spearman correlations |
| [`results/station_summary.csv`](results/station_summary.csv) | Station-level PM2.5 and exceedance summary |
| [`results/monthly_summary.csv`](results/monthly_summary.csv) | Monthly PM2.5 summary |
| [`results/seasonal_summary.csv`](results/seasonal_summary.csv) | Seasonal levels and exceedance frequencies |
| [`results/cleaning_quality_audit/`](results/cleaning_quality_audit/) | Development-only value screening |
| [`results/provenance_audit/`](results/provenance_audit/) | Upstream join summary |
| [`results/pollutant_missingness_station_month_development.csv`](results/pollutant_missingness_station_month_development.csv) | NO2/CO/O3 missingness by station-month |
| [`results/part1_evidence_summary.json`](results/part1_evidence_summary.json) | Saved summary of the evidence above |
| [`figures/`](figures/) | Distribution, seasonal, missingness, coverage and summary charts |

The `folds/` directory, `results/future_holdout.json` and `results/lookback_coverage.csv` are generated when the corresponding command runs.

## Reports and presentation

- [`reports/part_1/Part_1_Report.pdf`](reports/part_1/Part_1_Report.pdf) — Part 1 report.
- [`reports/part_1/source/part_1_report.typ`](reports/part_1/source/part_1_report.typ) — Typst source.
- [`presentation/Part_1_Presentation.pptx`](presentation/Part_1_Presentation.pptx) — Part 1 slides.

Build the PDF from the repository root with:

```powershell
typst compile --root . reports/part_1/source/part_1_report.typ reports/part_1/Part_1_Report.pdf
```

Typst is not a Python dependency.

## Tests

```powershell
python -m pytest -c src/pyproject.toml src/tests -q
```

The suite covers cleaning rules, data contracts, EDA artifacts, sequence boundaries, fold construction, leakage checks and temporal features.

## Repository layout

```text
.
├── data/
│   ├── manifest.yaml                  source metadata, checksums and license caveat
│   ├── provenance/                    saved Kaggle metadata snapshot
│   ├── raw/                           source CSVs (not shipped)
│   └── processed/                     cleaned panel and audit (generated)
├── notebooks/
│   └── 01_eda.ipynb                   exploratory notebook
├── src/
│   ├── config.py                      notebook path and config helpers
│   ├── configs/
│   │   └── project.yaml               data, cleaning, feature and path settings
│   ├── pm25_project/
│   │   ├── cli.py                     command-line entry points
│   │   ├── config.py                  configuration loading and validation
│   │   ├── data_audit.py              source read, canonicalisation and cleaning
│   │   ├── eda.py                     exploratory tables and figures
│   │   ├── features.py                past-only lags, rolling means and calendar terms
│   │   ├── preprocessing.py           fold-local sequence imputation and scaling
│   │   ├── sequence_audit.py          lookback coverage against the folds
│   │   ├── sequences.py               station-contained daily windows
│   │   ├── splits.py                  holdout and fold definitions
│   │   ├── temporal_diagnostics.py    per-station temporal diagnostics
│   │   └── schemas.py                 exploratory table column definitions
│   └── tests/                         automated checks
├── scripts/
│   ├── run_eda.py                     audit + EDA + summary figures
│   ├── audit_cleaning_quality.py      development-only value screening
│   ├── audit_upstream_join.py         upstream join reconstruction
│   └── generate_part1_report_evidence.py
├── results/                           saved tables and audit outputs
├── figures/                           saved charts
├── reports/part_1/                    Part 1 report and Typst source
└── presentation/                      Part 1 slides
```

## Limitations

1. **Licensing and provenance.** The bundle is compiled from OpenAQ, Copernicus and NASA sources. Its redistribution terms are not explicit, and the project does not include a complete transformation log from the original providers to the compiled CSV. Do not redistribute the data without confirming source-by-source rights.
2. **Saved artifacts are evidence, not a fresh run.** The tables under `results/` were produced from the source file described above. Re-running the commands overwrites them; cite only values you have reproduced or can trace to the saved files.
3. **No predictive results.** This repository covers data preparation and fold validation only. It reports no forecast skill and contains no prediction artifact.
4. **Coverage is uneven.** Many stations have short or interrupted daily histories; the development partition has only six stations with an uninterrupted run of at least 30 days. Any use of the folds must account for this.
