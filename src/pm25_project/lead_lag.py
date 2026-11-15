"""Station-level lead-lag correlations between PM2.5 and environmental variables.

Positive ``lag_days`` means the predictor leads PM2.5: PM2.5 at day ``t`` is
correlated with the predictor at ``t - lag_days``. Negative values mean the
predictor lags PM2.5. Correlations are computed only inside uninterrupted daily
segments, so no pair crosses a date gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DEFAULT_VARIABLES = (
    "aod",
    "no2",
    "co",
    "o3",
    "wind_u",
    "wind_v",
    "wind_speed",
    "temperature_celsius",
)


def _correlation(x: np.ndarray, y: np.ndarray, min_pairs: int = 3) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < max(3, min_pairs):
        return np.nan, np.nan, n
    xs = x[mask]
    ys = y[mask]
    if xs.std(ddof=0) == 0 or ys.std(ddof=0) == 0:
        return np.nan, np.nan, n
    result = pearsonr(xs, ys)
    return float(result.statistic), float(result.pvalue), n


def lead_lag_detail(
    frame: pd.DataFrame,
    variables: list[str] | tuple[str, ...] | None = None,
    max_lag: int = 7,
    min_pairs: int = 30,
) -> pd.DataFrame:
    """Return station-level correlations for every variable and lag."""
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    working = frame.copy()
    if "wind_speed" not in working and {"wind_u", "wind_v"}.issubset(working):
        working["wind_speed"] = np.hypot(working["wind_u"], working["wind_v"])
    columns = [name for name in (variables or DEFAULT_VARIABLES) if name in working]
    rows = []
    for station, group in working.groupby("station_id", sort=True):
        group = group.sort_values("timestamp")
        days = group["timestamp"].to_numpy(dtype="datetime64[D]")
        breaks = np.flatnonzero(np.diff(days).astype("timedelta64[D]").astype(int) != 1) + 1
        for positions in np.split(np.arange(len(group)), breaks):
            target = group["pm25"].to_numpy(dtype=float)[positions]
            for variable in columns:
                predictor = group[variable].to_numpy(dtype=float)[positions]
                for lag in range(-max_lag, max_lag + 1):
                    if lag > 0:
                        x, y = predictor[:-lag], target[lag:]
                    elif lag < 0:
                        x, y = predictor[-lag:], target[:lag]
                    else:
                        x, y = predictor, target
                    r, p, n = _correlation(x, y, min_pairs)
                    if n >= min_pairs:
                        rows.append(
                            {
                                "station_id": str(station),
                                "variable": variable,
                                "lag_days": lag,
                                "n_pairs": n,
                                "pearson_r": r,
                                "p_value": p,
                            }
                        )
    return pd.DataFrame(
        rows,
        columns=["station_id", "variable", "lag_days", "n_pairs", "pearson_r", "p_value"],
    )


def lead_lag_summary(
    detail: pd.DataFrame,
    variables: list[str] | tuple[str, ...] | None = None,
    max_lag: int = 7,
) -> pd.DataFrame:
    """Summarise the station-level correlations for every variable and lag."""
    columns = list(variables or DEFAULT_VARIABLES)
    grid = pd.MultiIndex.from_product([columns, range(-max_lag, max_lag + 1)], names=["variable", "lag_days"])
    if detail.empty:
        summary = pd.DataFrame(index=grid).reset_index()
        for name in ("n_stations", "median_r", "median_abs_r", "share_positive", "median_pairs"):
            summary[name] = np.nan
        summary["n_stations"] = 0
        return summary[["variable", "lag_days", "n_stations", "median_r", "median_abs_r", "share_positive", "median_pairs"]]
    grouped = detail.groupby(["variable", "lag_days"]).agg(
        n_stations=("station_id", "nunique"),
        median_r=("pearson_r", "median"),
        median_abs_r=("pearson_r", lambda values: float(np.abs(values).median())),
        share_positive=("pearson_r", lambda values: float((values > 0).mean())),
        median_pairs=("n_pairs", "median"),
    )
    summary = grouped.reindex(grid).reset_index()
    summary["n_stations"] = summary["n_stations"].fillna(0).astype(int)
    return summary


def lead_lag_table(
    frame: pd.DataFrame,
    variables: list[str] | tuple[str, ...] | None = None,
    max_lag: int = 7,
    min_pairs: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = lead_lag_detail(frame, variables, max_lag, min_pairs)
    summary = lead_lag_summary(detail, variables, max_lag)
    return detail, summary
