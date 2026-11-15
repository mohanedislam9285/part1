import numpy as np
import pandas as pd

from pm25_project.relationships import (
    aod_meteorology_table,
    aod_wind_tertile_table,
    run_relationships,
    season_correlations,
    station_correlations,
    yearly_table,
)


def _panel():
    rng = np.random.default_rng(11)
    rows = []
    for station in ("A", "B"):
        for i, date in enumerate(pd.date_range("2020-01-01", periods=400)):
            aod = rng.normal(0.3, 0.1)
            rows.append(
                {
                    "timestamp": date,
                    "station_id": station,
                    "pm25": 8 + 3 * np.sin(2 * np.pi * i / 365) + 4 * aod + rng.normal(0, 1),
                    "temperature_celsius": 15 + 8 * np.sin(2 * np.pi * i / 365),
                    "pressure_mb": 1010 + rng.normal(0, 3),
                    "wind_u": rng.normal(0, 2),
                    "wind_v": rng.normal(0, 2),
                    "no2": rng.normal(0.0001, 0.00002),
                    "co": rng.normal(0.03, 0.005),
                    "o3": rng.normal(0.12, 0.02),
                    "aod": aod,
                }
            )
    return pd.DataFrame(rows)


def test_seasonal_and_station_relationships_cover_every_split():
    panel = _panel()
    seasonal = season_correlations(panel)
    assert set(seasonal.season) == {"DJF", "MAM", "JJA", "SON"}
    assert set(seasonal.variable) == {
        "temperature_celsius",
        "pressure_mb",
        "wind_u",
        "wind_v",
        "wind_speed",
        "no2",
        "co",
        "o3",
        "aod",
    }
    assert seasonal.spearman_rho.notna().all()

    station = station_correlations(panel)
    assert station.station_id.nunique() == 2
    assert station.groupby("station_id").size().eq(9).all()


def test_aod_adjustment_and_wind_tertiles_are_computed():
    panel = _panel()
    table = aod_meteorology_table(panel)
    assert set(table.scope) == {"pooled", "DJF", "MAM", "JJA", "SON"}
    assert table.n_pairs.iloc[0] == len(panel)
    assert table.n_pairs.iloc[1:].sum() == len(panel)
    assert table.partial_spearman.between(-1, 1).all()

    tertiles = aod_wind_tertile_table(panel)
    assert list(tertiles.wind_tertile) == ["low", "mid", "high"]
    assert tertiles.n_pairs.sum() == len(panel)


def test_aod_tables_are_empty_when_columns_are_absent(panel):
    adjusted = aod_meteorology_table(panel)
    tertiles = aod_wind_tertile_table(panel)
    assert adjusted.empty and "partial_spearman" in adjusted.columns
    assert tertiles.empty and "wind_tertile" in tertiles.columns


def test_yearly_table_counts_and_shares():
    panel = _panel()
    yearly = yearly_table(panel)
    assert list(yearly.year) == [2020, 2021]
    assert yearly.n_records.sum() == len(panel)
    assert yearly.n_stations.eq(2).all()
    assert yearly.pct_gt_25.between(0, 100).all()
    assert yearly.pct_gt_50.between(0, 100).all()


def test_run_relationships_writes_tables(tmp_path):
    panel = _panel()
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    output = run_relationships(panel, {}, results, figures)
    assert output["yearly"].shape[0] == 2
    for name in (
        "predictor_relationships_by_season.csv",
        "predictor_relationships_by_station.csv",
        "aod_meteorology_relationships.csv",
        "aod_by_wind_tertile.csv",
        "yearly_trend.csv",
    ):
        assert (results / name).is_file()
    assert (figures / "pm25_yearly_trend.png").is_file()
