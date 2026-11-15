"""Descriptive predictor relationships and yearly drift.

All tables are computed on the cleaned panel. The yearly table is also written
for the development partition so the time-series notebook can reuse it without
touching the reserved holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

DEFAULT_VARIABLES = (
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "wind_speed",
    "no2",
    "co",
    "o3",
    "aod",
)
MET_CONTROLS = ("temperature_celsius", "pressure_mb", "wind_u", "wind_v")
SEASONS = ("DJF", "MAM", "JJA", "SON")


def add_season(frame: pd.DataFrame) -> pd.DataFrame:
    month = frame["timestamp"].dt.month
    season = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        list(SEASONS),
        default="NA",
    )
    return frame.assign(season=season)


def add_wind_speed(frame: pd.DataFrame) -> pd.DataFrame:
    if "wind_speed" in frame or not {"wind_u", "wind_v"}.issubset(frame):
        return frame
    return frame.assign(wind_speed=np.hypot(frame["wind_u"], frame["wind_v"]))


def season_correlations(frame: pd.DataFrame, variables=None, min_pairs=30) -> pd.DataFrame:
    """Spearman correlation between PM2.5 and each variable within each season."""
    working = add_season(add_wind_speed(frame))
    columns = [v for v in (variables or DEFAULT_VARIABLES) if v in working]
    rows = []
    for season, group in working.groupby("season", sort=True):
        for variable in columns:
            pairs = group[["pm25", variable]].dropna()
            if len(pairs) < min_pairs:
                continue
            rows.append(
                {
                    "season": season,
                    "variable": variable,
                    "n_pairs": int(len(pairs)),
                    "spearman_rho": float(spearmanr(pairs.pm25, pairs[variable]).statistic),
                }
            )
    return pd.DataFrame(rows, columns=["season", "variable", "n_pairs", "spearman_rho"])


def station_correlations(frame: pd.DataFrame, variables=None, min_pairs=30) -> pd.DataFrame:
    """Spearman correlation between PM2.5 and each variable within each station."""
    working = add_wind_speed(frame)
    columns = [v for v in (variables or DEFAULT_VARIABLES) if v in working]
    rows = []
    for station, group in working.groupby("station_id", sort=True):
        for variable in columns:
            pairs = group[["pm25", variable]].dropna()
            if len(pairs) < min_pairs:
                continue
            rows.append(
                {
                    "station_id": str(station),
                    "variable": variable,
                    "n_pairs": int(len(pairs)),
                    "spearman_rho": float(spearmanr(pairs.pm25, pairs[variable]).statistic),
                }
            )
    return pd.DataFrame(rows, columns=["station_id", "variable", "n_pairs", "spearman_rho"])


def _partial_spearman(group: pd.DataFrame, x: str, y: str, controls) -> tuple[int, float, float]:
    data = group[[x, y, *controls]].dropna()
    n = int(len(data))
    if n < 30:
        return n, np.nan, np.nan
    raw = float(spearmanr(data[x], data[y]).statistic)
    design = np.column_stack(
        [np.ones(n)] + [rankdata(data[c].to_numpy(dtype=float)) for c in controls]
    )
    target = rankdata(data[y].to_numpy(dtype=float))
    predictor = rankdata(data[x].to_numpy(dtype=float))
    residuals = []
    for values in (target, predictor):
        beta = np.linalg.lstsq(design, values, rcond=None)[0]
        residuals.append(values - design @ beta)
    partial = float(np.corrcoef(residuals[0], residuals[1])[0, 1])
    return n, raw, partial


def aod_meteorology_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Raw and meteorology-adjusted PM2.5–AOD rank correlations, pooled and by season."""
    columns = ["scope", "n_pairs", "raw_spearman", "partial_spearman", "controls"]
    if "aod" not in frame or not set(MET_CONTROLS).issubset(frame):
        return pd.DataFrame(columns=columns)
    working = add_season(add_wind_speed(frame))
    rows = []

    def record(scope, group):
        n, raw, partial = _partial_spearman(group, "aod", "pm25", MET_CONTROLS)
        rows.append(
            {
                "scope": scope,
                "n_pairs": n,
                "raw_spearman": raw,
                "partial_spearman": partial,
                "controls": "temperature_celsius, pressure_mb, wind_u, wind_v",
            }
        )

    record("pooled", working)
    for season, group in working.groupby("season", sort=True):
        record(season, group)
    return pd.DataFrame(rows)


def aod_wind_tertile_table(frame: pd.DataFrame) -> pd.DataFrame:
    """PM2.5–AOD rank correlation within low, middle and high wind-speed thirds."""
    columns = ["wind_tertile", "n_pairs", "mean_wind_speed", "spearman_rho"]
    if "aod" not in frame:
        return pd.DataFrame(columns=columns)
    working = add_wind_speed(frame)
    if "wind_speed" not in working:
        return pd.DataFrame(columns=columns)
    working = working.loc[working["wind_speed"].notna()].copy()
    working["wind_tertile"] = pd.qcut(working["wind_speed"], 3, labels=["low", "mid", "high"])
    rows = []
    for tertile, group in working.groupby("wind_tertile", observed=True, sort=True):
        pairs = group[["pm25", "aod", "wind_speed"]].dropna()
        if len(pairs) < 30:
            continue
        rows.append(
            {
                "wind_tertile": str(tertile),
                "n_pairs": int(len(pairs)),
                "mean_wind_speed": float(pairs.wind_speed.mean()),
                "spearman_rho": float(spearmanr(pairs.pm25, pairs.aod).statistic),
            }
        )
    return pd.DataFrame(rows)


def yearly_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Yearly PM2.5 level and high-concentration frequency."""
    working = frame.assign(year=frame["timestamp"].dt.year)
    rows = []
    for year, group in working.groupby("year", sort=True):
        values = group["pm25"].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "year": int(year),
                "n_records": int(len(group)),
                "n_stations": int(group["station_id"].nunique()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
                "pct_gt_25": float((values > 25).mean() * 100),
                "pct_gt_50": float((values > 50).mean() * 100),
                "pct_gt_100": float((values > 100).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def _plot_yearly(table: pd.DataFrame, destination: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(table.year, table["mean"], marker="o", color="#10243A", label="Mean")
    axes[0].plot(table.year, table["median"], marker="s", color="#087F80", label="Median")
    axes[0].set(xlabel="Year", ylabel="PM2.5 (µg/m³)", title="Yearly PM2.5 level")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(table.year, table.pct_gt_25, marker="o", color="#D9822B", label="> 25")
    axes[1].plot(table.year, table.pct_gt_50, marker="s", color="#C0392B", label="> 50")
    axes[1].set(xlabel="Year", ylabel="Share of records (%)", title="High-concentration frequency")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.suptitle(title, fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_relationships(frame: pd.DataFrame, cfg: dict, results: Path, figdir: Path) -> dict:
    """Write relationship tables, yearly drift tables and the yearly figure."""
    working = add_wind_speed(frame)
    results.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    season = season_correlations(working)
    station = station_correlations(working)
    aod_met = aod_meteorology_table(working)
    aod_wind = aod_wind_tertile_table(working)
    yearly = yearly_table(working)

    season.to_csv(results / "predictor_relationships_by_season.csv", index=False)
    station.to_csv(results / "predictor_relationships_by_station.csv", index=False)
    aod_met.to_csv(results / "aod_meteorology_relationships.csv", index=False)
    aod_wind.to_csv(results / "aod_by_wind_tertile.csv", index=False)
    yearly.to_csv(results / "yearly_trend.csv", index=False)

    development_yearly = None
    holdout_path = results / "future_holdout.json"
    if holdout_path.is_file():
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        development = working.iloc[holdout["development_indices"]]
        development_yearly = yearly_table(development)
        development_yearly.to_csv(results / "yearly_trend_development.csv", index=False)

    _plot_yearly(yearly, figdir / "pm25_yearly_trend.png", "Cleaned panel, 2016–2026 (unbalanced: 13–57 stations per year)")
    if development_yearly is not None:
        _plot_yearly(
            development_yearly,
            figdir / "pm25_yearly_trend_development.png",
            "Development partition, 2016–2024 (unbalanced: 13–54 stations per year)",
        )

    return {
        "season": season,
        "station": station,
        "aod_meteorology": aod_met,
        "aod_wind": aod_wind,
        "yearly": yearly,
        "yearly_development": development_yearly,
    }
