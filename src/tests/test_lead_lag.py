import numpy as np
import pandas as pd

from pm25_project.lead_lag import lead_lag_table


def test_lead_lag_uses_predictor_at_earlier_day():
    rng = np.random.default_rng(7)
    rows = []
    for station in ("A", "B"):
        values = rng.normal(10, 3, size=60)
        for i, date in enumerate(pd.date_range("2021-01-01", periods=60)):
            rows.append(
                {
                    "timestamp": date,
                    "station_id": station,
                    "pm25": float(values[i]),
                    "aod": float(values[i + 1]) if i + 1 < len(values) else np.nan,
                }
            )
    detail, summary = lead_lag_table(pd.DataFrame(rows), variables=["aod"], max_lag=2, min_pairs=30)

    assert set(detail.lag_days) == {-2, -1, 0, 1, 2}
    at_one = detail.loc[detail.lag_days.eq(1), "pearson_r"]
    assert at_one.notna().all()
    assert (at_one > 0.99).all()
    at_zero = detail.loc[detail.lag_days.eq(0), "pearson_r"]
    assert (at_zero.abs() < 0.9).all()

    estimable = summary[summary.n_stations > 0]
    assert estimable.n_stations.eq(2).all()
    assert estimable.median_r.loc[estimable.lag_days.eq(1)].iloc[0] > 0.99


def test_lead_lag_does_not_cross_date_gaps():
    rows = []
    for i, date in enumerate(pd.date_range("2021-01-01", periods=80)):
        if i == 40:
            continue
        rows.append({"timestamp": date, "station_id": "A", "pm25": float(i), "aod": float(i)})
    detail, _ = lead_lag_table(pd.DataFrame(rows), variables=["aod"], max_lag=2, min_pairs=30)
    assert len(detail) == 10
    assert detail.n_pairs.max() == 40
    assert detail.n_pairs.min() == 37
