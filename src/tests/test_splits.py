import pandas as pd

from pm25_project.splits import temporal_panel_folds,spatial_folds,random_diagnostic_folds,development_holdout_split,save_folds

def test_temporal_folds_are_chronological_and_date_grouped(panel):
    result=temporal_panel_folds(panel,n_splits=5,embargo_days=1)
    assert len(result["folds"])==5
    for f in result["folds"]:
        tr=panel.iloc[f["train_indices"]]; va=panel.iloc[f["validation_indices"]]
        assert tr.timestamp.max()<va.timestamp.min()
        assert tr.timestamp.max()<=va.timestamp.min()-pd.Timedelta(days=2)
        assert set(tr.timestamp).isdisjoint(va.timestamp)

def test_spatial_groups_are_disjoint(panel):
    result=spatial_folds(panel,2)
    for f in result["folds"]: assert set(f["train_groups"]).isdisjoint(f["validation_groups"])

def test_random_is_labelled_diagnostic_only(panel):
    assert "only" in random_diagnostic_folds(panel,5)["cv_purpose"]

def test_future_holdout_is_latest_and_disjoint(panel):
    hold=development_holdout_split(panel,.2,10)
    dev=panel.iloc[hold["development_indices"]]; future=panel.iloc[hold["holdout_indices"]]
    assert dev.timestamp.max()<future.timestamp.min()
    assert future.timestamp.nunique()==hold["n_holdout_dates"]

def test_saved_temporal_folds_exclude_holdout(panel,config):
    config["project"].update(temporal_folds=5,embargo_days=1,holdout_fraction=.2,holdout_min_days=10)
    defs=save_folds(panel,config); holdout=set(defs["future_holdout.json"]["holdout_indices"])
    for f in defs["folds_temporal.json"]["folds"]:
        assert not holdout.intersection(f["train_indices"])
        assert not holdout.intersection(f["validation_indices"])
