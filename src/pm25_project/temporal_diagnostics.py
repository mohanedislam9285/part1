from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from .schemas import EXPLORATORY_TEMPORAL

def temporal_table(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for station,g in df.groupby("station_id",sort=True):
        g=g.sort_values("timestamp"); ts=g.timestamp.dropna(); diffs=ts.diff().dropna().dt.total_seconds(); y=g.pm25.dropna().to_numpy(dtype=float)
        row={k:np.nan for k in EXPLORATORY_TEMPORAL}; row.update(variable="pm25",node_id=str(station),n_records=len(g),t_min=str(ts.min()) if len(ts) else None,t_max=str(ts.max()) if len(ts) else None,sampling_interval_median_sec=float(diffs.median()) if len(diffs) else np.nan,sampling_interval_std_sec=float(diffs.std()) if len(diffs)>1 else np.nan,pct_missing_intervals=float(100*(diffs.ne(86400).mean())) if len(diffs) else np.nan,n_days_above_25=int((g.pm25>25).sum()),n_days_above_50=int((g.pm25>50).sum()),pct_days_above_25=float((g.pm25>25).mean()*100),pct_days_above_50=float((g.pm25>50).mean()*100),notes="Daily cadence diagnostics; no hourly statistics.")
        regular=bool(len(diffs)==0 or diffs.eq(86400).all())
        if regular and len(y)>2: row["acf_lag1"]=float(pd.Series(y).autocorr(1))
        if regular and len(y)>7: row["acf_lag7"]=float(pd.Series(y).autocorr(7))
        if not regular: row["notes"]="Irregular intervals: interval-sensitive ACF/stationarity tests omitted; lag unit would be ambiguous."
        if regular and len(y)>30:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                row["ljung_box_p"]=float(acorr_ljungbox(y,lags=[7],return_df=True)["lb_pvalue"].iloc[0])
                from statsmodels.tsa.stattools import adfuller,kpss,pacf
                from statsmodels.tools.sm_exceptions import InterpolationWarning
                row["adf_p"]=float(adfuller(y,autolag="AIC",result_object=True).pvalue)
                with warnings.catch_warnings(record=True) as diagnostic_warnings:
                    warnings.simplefilter("always",InterpolationWarning)
                    row["kpss_p"]=float(kpss(y,regression="c",nlags="auto",result_object=True).pvalue)
                if any(issubclass(item.category,InterpolationWarning) for item in diagnostic_warnings):
                    row["notes"]+=" KPSS p-value is at least 0.1; the upper lookup-table bound was reached."
                row["pacf_lag1"]=float(pacf(y,nlags=1,method="ywm")[1])
            except ImportError:
                row["notes"]="Install analysis extras for ADF/KPSS/Ljung-Box/PACF; ACF/trend computed from daily values."
            except (ValueError,ZeroDivisionError):
                row["notes"]="Some stationarity diagnostics were undefined for this series."
        if len(ts)>1:
            elapsed=(ts-ts.min()).dt.total_seconds().to_numpy()/86400
            trend=linregress(elapsed,g.pm25.to_numpy(dtype=float)); row["trend_slope_per_day"]=float(trend.slope); row["trend_pval"]=float(trend.pvalue)
            row["n_cycles_observed"]=float((ts.max()-ts.min()).days/365.25)
        if len(y)>=20:
            half=len(y)//2; before=float(np.median(y[:half])); after=float(np.median(y[half:])); row["target_drift_stat"]=after-before
            try:
                from scipy.stats import mannwhitneyu
                row["target_drift_pval"]=float(mannwhitneyu(y[:half],y[half:],alternative="two-sided").pvalue)
            except ValueError: pass
        for canonical in ("no2","co","o3"):
            if canonical in g:
                monthly=g.groupby(g.timestamp.dt.month)[canonical].apply(lambda s: float(s.isna().mean()*100))
                row[f"{canonical}_pct_missing_by_month"]=json.dumps({str(int(month)): value for month,value in monthly.items()},sort_keys=True)
        row["notes"]=(str(row["notes"])+" Daily lag units; ACF lag 7 is 7 days, not hourly.")
        rows.append(row)
    return pd.DataFrame(rows,columns=EXPLORATORY_TEMPORAL)
