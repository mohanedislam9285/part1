EXPLORATORY_DATA = [
    "variable", "family", "type", "unit", "n_total", "n_missing", "pct_missing",
    "n_unique", "modalities", "min", "q1", "median", "q3", "max", "mean", "std",
    "skewness", "kurtosis", "iqr", "n_out_iqr", "pct_out_iqr", "n_out_z",
    "pct_out_z", "n_sentinel_low", "n_sentinel_high", "n_removed_total",
    "target_assoc", "notes",
]

EXPLORATORY_TEMPORAL = [
    "variable", "node_id", "n_records", "t_min", "t_max",
    "sampling_interval_median_sec", "sampling_interval_std_sec",
    "pct_missing_intervals", "acf_lag1", "acf_lag7", "pacf_lag1", "ljung_box_p",
    "adf_p", "kpss_p", "trend_slope_per_day", "trend_pval", "target_drift_stat",
    "target_drift_pval", "n_days_above_25", "n_days_above_50",
    "pct_days_above_25", "pct_days_above_50", "n_cycles_observed",
    "no2_pct_missing_by_month", "co_pct_missing_by_month",
    "o3_pct_missing_by_month", "notes",
]
