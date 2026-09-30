"""Smoke tests for all 8 layers. Run: python -m pytest tests/ -v"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import *  # noqa
from src import (load_data, prepare_features, split_and_scale,
                 train_and_select, StudentDigitalTwin, risk_profile,
                 recommend, compare_scenarios, DEFAULT_SCENARIOS)
from sklearn.preprocessing import LabelEncoder


def _trained():
    df, _ = load_data(n_synthetic=400)
    X, y = prepare_features(df)
    names = list(LabelEncoder().fit(df["Depression_Type"].astype(str)).classes_)
    Xtr, Xte, ytr, yte, scaler, cols = split_and_scale(X, y)
    best_name, best, fitted, results = train_and_select(Xtr, Xte, ytr, yte)
    return best, scaler, cols, names, X.iloc[0].to_dict()


def test_full_pipeline():
    best, scaler, cols, names, feats = _trained()
    twin = StudentDigitalTwin("test-1", feats, model=best,
                              scaler=scaler, feature_columns=cols,
                              label_names=names)
    pred = twin.predict()
    assert "label" in pred and "proba" in pred
    risk = risk_profile(pred["proba"], twin.state, names)
    assert 0 <= risk["score"] <= 30 and risk["level"] in RISK_LEVELS
    reco = recommend(risk, twin.state)
    assert len(reco["strategies"]) > 0
    outs = compare_scenarios(twin, DEFAULT_SCENARIOS)
    assert len(outs) == len(DEFAULT_SCENARIOS)
    twin.update(Sleep_Hours=7.5)
    assert len(twin.history) >= 2
