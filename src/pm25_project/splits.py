from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

def development_holdout_split(df:pd.DataFrame,fraction:float=.2,min_holdout_days:int=30)->dict:
    """Reserve the latest contiguous dates before constructing any development folds."""
    if not 0<fraction<.5: raise ValueError("holdout_fraction must be in (0, 0.5)")
    date_series=pd.to_datetime(df.timestamp,utc=True).dt.tz_localize(None)
    dates=np.array(sorted(date_series.dropna().unique()),dtype="datetime64[ns]")
    n_holdout=max(min_holdout_days,int(np.ceil(len(dates)*fraction)))
    if len(dates)-n_holdout<8: raise ValueError("Insufficient dates for an independent holdout plus development folds")
    cut=len(dates)-n_holdout; dev_dates=set(dates[:cut]); holdout_dates=set(dates[cut:]); dt=date_series.to_numpy(dtype="datetime64[ns]")
    return {"scheme":"independent_future_holdout","status":"reserved","development_indices":np.flatnonzero(np.isin(dt,list(dev_dates))).tolist(),"holdout_indices":np.flatnonzero(np.isin(dt,list(holdout_dates))).tolist(),"development_end":str(dates[cut-1]),"holdout_start":str(dates[cut]),"holdout_end":str(dates[-1]),"n_development_dates":int(len(dev_dates)),"n_holdout_dates":int(len(holdout_dates)),"fraction_requested":fraction,"note":"Reserved as the latest chronological holdout; excluded from every development fold."}

def temporal_panel_folds(df: pd.DataFrame, n_splits: int=5, embargo_days: int=1) -> dict:
    date_series=pd.to_datetime(df.timestamp,utc=True).dt.tz_localize(None)
    dates=np.array(sorted(date_series.dropna().unique()),dtype="datetime64[ns]")
    if len(dates)<n_splits+embargo_days+2: raise ValueError(f"Need at least {n_splits+embargo_days+2} unique dates for temporal CV; found {len(dates)}")
    # Expanding chronological folds with a validation block; gap excludes embargo dates before validation.
    test_size=max(1,len(dates)//(n_splits+1)); folds=[]; first=len(dates)-n_splits*test_size
    for k in range(n_splits):
        va0=first+k*test_size; va1=min(len(dates),va0+test_size); tr1=max(0,va0-embargo_days)
        if tr1==0: continue
        trdates=set(dates[:tr1]); vadates=set(dates[va0:va1])
        # Date-based assignment keeps all stations on a date together.
        dt=date_series.to_numpy(dtype="datetime64[ns]"); tr=np.flatnonzero(np.isin(dt, list(trdates))); va=np.flatnonzero(np.isin(dt,list(vadates)))
        folds.append({"fold":k+1,"train_indices":tr.tolist(),"validation_indices":va.tolist(),"train_start":str(dates[0]),"train_end":str(dates[tr1-1]),"validation_start":str(dates[va0]),"validation_end":str(dates[va1-1]),"embargo_days":embargo_days})
    return {"scheme":"temporal_panel_expanding","n_splits":len(folds),"date_unit":"UTC day","folds":folds}

def spatial_folds(df: pd.DataFrame,n_splits:int=5)->dict:
    groups=df.station_id.astype(str).to_numpy(); unique=np.unique(groups)
    if len(unique)<n_splits: return {"scheme":"spatial_group_kfold","status":"not_applicable","reason":f"Need {n_splits} stations; found {len(unique)}"}
    folds=[]
    for i,(tr,va) in enumerate(GroupKFold(n_splits=n_splits).split(df,groups=groups),1): folds.append({"fold":i,"train_indices":tr.tolist(),"validation_indices":va.tolist(),"train_groups":sorted(np.unique(groups[tr]).tolist()),"validation_groups":sorted(np.unique(groups[va]).tolist())})
    return {"scheme":"spatial_group_kfold","n_splits":n_splits,"folds":folds}

def random_diagnostic_folds(df:pd.DataFrame,n_splits:int=5,seed:int=42)->dict:
    folds=[]
    for i,(tr,va) in enumerate(KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(df),1): folds.append({"fold":i,"train_indices":tr.tolist(),"validation_indices":va.tolist()})
    return {"scheme":"random_diagnostic","cv_purpose":"optimism diagnostic only; not selection evidence","n_splits":n_splits,"seed":seed,"folds":folds}

def save_folds(df,cfg):
    project_root=Path(cfg["_config_dir"])
    fold_root=project_root/"folds"; fold_root.mkdir(parents=True,exist_ok=True)
    result_root=project_root/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; result_root.mkdir(parents=True,exist_ok=True)
    holdout=development_holdout_split(df,cfg["project"].get("holdout_fraction",.2),cfg["project"].get("holdout_min_days",30))
    dev=df.iloc[holdout["development_indices"]].reset_index(drop=True)
    temporal=temporal_panel_folds(dev,cfg["project"]["temporal_folds"],cfg["project"]["embargo_days"])
    # Map fold positions back to the original cleaned table for traceable shared indices.
    original=np.asarray(holdout["development_indices"])
    for fold in temporal["folds"]:
        fold["train_indices"]=original[np.asarray(fold["train_indices"],dtype=int)].tolist()
        fold["validation_indices"]=original[np.asarray(fold["validation_indices"],dtype=int)].tolist()
    temporal["holdout_file"]="future_holdout.json"; temporal["development_end"]=holdout["development_end"]
    spatial=spatial_folds(dev,cfg["project"]["temporal_folds"])
    random=random_diagnostic_folds(dev,cfg["project"]["temporal_folds"],cfg["project"]["seed"])
    for scheme in (spatial,random):
        if "folds" in scheme:
            for fold in scheme["folds"]:
                fold["train_indices"]=original[np.asarray(fold["train_indices"],dtype=int)].tolist(); fold["validation_indices"]=original[np.asarray(fold["validation_indices"],dtype=int)].tolist()
    definitions={"folds_temporal.json":temporal,"folds_spatial.json":spatial,"folds_standard.json":random,"future_holdout.json":holdout}
    for name,data in definitions.items():
        destination=result_root/name if name=="future_holdout.json" else fold_root/name
        destination.write_text(json.dumps(data,indent=2),encoding="utf-8")
    return definitions
