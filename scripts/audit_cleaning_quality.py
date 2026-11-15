"""Development-only data quality audit: flags unusual values, never drops them.

Reproduces the 352 station-relative flags on 60 of the 71 screened stations cited
in the report. Row-level flag tables are written under ``data/processed``, which is
outside version control.

Prerequisites: ``python -m pm25_project.cli audit`` and ``... folds``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm25_project.config import load_config  # noqa: E402

CONFIG = ROOT / "src" / "configs" / "project.yaml"
cfg, project_root = load_config(CONFIG)
RESULTS = project_root / cfg["paths"]["results_root"] / cfg["project"].get("group_id", "")
CLEANED = project_root / "data" / "processed" / "cleaned_daily.csv"
LOCK = RESULTS / "future_holdout.json"

for path, prerequisite in ((CLEANED, "audit"), (LOCK, "folds")):
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Run `python -m pm25_project.cli {prerequisite} "
            f"--config {CONFIG}` first."
        )

OUT = RESULTS / "cleaning_quality_audit"
OUT.mkdir(parents=True, exist_ok=True)
LOCAL = project_root / "data" / "processed" / "cleaning_quality_row_level"
LOCAL.mkdir(parents=True, exist_ok=True)
lock = json.loads(LOCK.read_text(encoding="utf-8"))
df = pd.read_csv(CLEANED, parse_dates=["timestamp"])
dev = df.iloc[np.asarray(lock["development_indices"], dtype=int)].copy()
dev["timestamp"] = pd.to_datetime(dev.timestamp, utc=True)
if dev.timestamp.max() > pd.to_datetime(lock["development_end"], utc=True):
    raise RuntimeError("Development audit rows cross the holdout cutoff")
# Development-only target validity, key integrity, coordinate range/stability.
duplicates = int(dev.duplicated(["station_id", "timestamp"]).sum())
invalid = int((~np.isfinite(dev.pm25) | (dev.pm25 <= 0) | (dev.pm25 >= 500)).sum())
coord_invalid = int(
    (
        ~np.isfinite(dev.latitude)
        | ~np.isfinite(dev.longitude)
        | ~dev.latitude.between(-90, 90)
        | ~dev.longitude.between(-180, 180)
    ).sum()
)
summary = {
    "scope": "cleaned development partition only",
    "development_rows": int(len(dev)),
    "stations": int(dev.station_id.nunique()),
    "date_min": dev.timestamp.min().isoformat(),
    "date_max": dev.timestamp.max().isoformat(),
    "duplicate_station_day_keys": duplicates,
    "nonfinite_or_out_of_rule_pm25": invalid,
    "invalid_coordinates": coord_invalid,
    "holdout_values_used": False,
    "statistical_flags_are_removed": False,
}
summary["target_distribution"] = {
    k: float(v) for k, v in dev.pm25.quantile([.5, .9, .95, .99, 1]).items()
}
summary["target_counts"] = {f"gt_{v}": int((dev.pm25 > v).sum()) for v in (25, 50, 100, 250)}
# A z-score screen is reported for comparison; the IQR screen is the cited one.
z_scores = dev.groupby("station_id").pm25.transform(
    lambda values: (values - values.mean()) / values.std(ddof=0)
)
z_rows = dev.assign(z_score=z_scores)
z_rows = z_rows[z_rows.z_score.abs() > 3]
summary["station_zscore_screen"] = {
    "threshold": "|z| > 3 within station",
    "flagged_rows": int(len(z_rows)),
    "flagged_stations": int(z_rows.station_id.nunique()),
    "policy": "diagnostic screening only; no automatic deletion",
}
predictor_cols = [name for name in cfg["data"].get("predictors", {}) if name in df.columns]
summary["predictor_missingness"] = {
    c: {
        "missing_n": int(dev[c].isna().sum()),
        "missing_fraction": float(dev[c].isna().mean()),
        "finite_n": int(np.isfinite(dev[c]).sum()),
    }
    for c in predictor_cols
}
# Station-specific Tukey upper fences (Q3 + 3*IQR) are screening flags only.
rows = []
for station, group in dev.groupby("station_id", sort=True):
    x = group.pm25.dropna()
    if len(x) < 30:
        continue
    q1, q3 = x.quantile([.25, .75])
    iqr = q3 - q1
    fence = float(q3 + 3 * iqr)
    flagged = group[group.pm25 > fence]
    for _, row in flagged.iterrows():
        rows.append(
            {
                "station_id": station,
                "timestamp": row.timestamp.isoformat(),
                "pm25": float(row.pm25),
                "q3_plus_3iqr_fence": fence,
                "station_n": int(len(x)),
                "flag": "station-relative statistical high value; review only; retained",
            }
        )
flag_frame = pd.DataFrame(rows)
summary["station_tukey_screen"] = {
    "stations_screened_min_n30": int(dev.groupby("station_id").pm25.count().ge(30).sum()),
    "flagged_stations": int(flag_frame.station_id.nunique()) if not flag_frame.empty else 0,
    "flagged_rows": len(rows),
    "policy": "IQR Q3+3*IQR; diagnostic screening only; no automatic deletion",
}
summary["cleaning_policy_assessment"] = {
    "configured_pm25_rule": "retain 0 < PM2.5 < 500 ug/m3",
    "source_measurements_removed_for_statistical_outlier_only": 0,
    "high_values_are_retained_unless_source_QA_invalidates_them": True,
    "high_value_validation_note": (
        "Rows above 100 ug/m3 are retained; statistical extremeness alone cannot "
        "establish an instrument fault."
    ),
}
(OUT / "cleaning_quality_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
flag_frame.to_csv(LOCAL / "station_relative_high_value_flags.csv", index=False)
z_rows.to_csv(LOCAL / "station_zscore_flags.csv", index=False)
pd.DataFrame(
    [{"variable": c, **summary["predictor_missingness"][c]} for c in predictor_cols]
).to_csv(LOCAL / "development_predictor_missingness.csv", index=False)
screen = summary["station_tukey_screen"]
print(
    f"{screen['flagged_rows']} flags on {screen['flagged_stations']} stations; "
    f"z-score screen {summary['station_zscore_screen']['flagged_rows']} rows"
)
print(f"Wrote {OUT / 'cleaning_quality_audit.json'} (row-level tables in {LOCAL})")
