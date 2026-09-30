"""Streamlit dashboard — Layer 7 UI over the 8-layer system.
Run:  streamlit run app.py
"""
import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (load_data, prepare_features, split_and_scale,
                 train_and_select, StudentDigitalTwin, risk_profile,
                 risk_trend, recommend, compare_scenarios)
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Mental-Health Digital Twin", layout="wide")
st.title("🧠 Student Mental-Health Digital Twin")
st.caption("8-layer system: preprocess → predict → twin → what-if → risk → guidance → dashboard → reports")


@st.cache_resource(show_spinner="Training models (once, then cached)…")
def get_trained():
    df, used_real = load_data()
    X, y = prepare_features(df)
    label_names = list(LabelEncoder().fit(df["Depression_Type"].astype(str)).classes_)
    Xtr, Xte, ytr, yte, scaler, cols = split_and_scale(X, y)
    best_name, best, fitted, results = train_and_select(
        Xtr, Xte, ytr, yte, label_names=label_names)
    return df, used_real, X, label_names, scaler, cols, best_name, best, results


df, used_real, X, label_names, scaler, cols, best_name, best, results = get_trained()

st.sidebar.success(f"Best model: **{best_name}**  |  real data: {used_real}")
st.sidebar.subheader("Model F1 (macro)")
for k, v in results.items():
    st.sidebar.write(f"{k}: {v['f1_macro']:.3f}")

st.subheader("1️⃣ Live student state")
c1, c2, c3 = st.columns(3)
vals = {}
# Sliders auto-adapt to the REAL csv columns (all numeric) or synthetic schema.
for i, f in enumerate(cols):
    col = [c1, c2, c3][i % 3]
    lo, hi = float(df[f].min()), float(df[f].max())
    med = float(df[f].median())
    if hi - lo <= 2:  # binary flag
        vals[f] = col.selectbox(f, sorted(df[f].unique().tolist()),
                                index=0)
    else:
        vals[f] = col.slider(f, float(lo), float(hi), float(med))

twin = StudentDigitalTwin("demo-student", vals, model=best,
                          scaler=scaler, feature_columns=cols,
                          label_names=label_names)
pred = twin.predict()
risk = risk_profile(pred["proba"], twin.state, label_names)
reco = recommend(risk, twin.state)

st.subheader("2️⃣ Prediction + risk")
m1, m2, m3 = st.columns(3)
m1.metric("Predicted type", pred["label"])
m2.metric("Risk score", f"{risk['score']}/30")
m3.metric("Risk level", risk["level"])
st.write("Top-3:", ", ".join(f"{t['type']} ({t['prob']:.0%})" for t in risk["top3"]))

st.subheader("3️⃣ What-if interventions")
outs = compare_scenarios(twin)  # auto-picks scenarios for this schema
st.dataframe(pd.DataFrame([{"scenario": o["scenario"],
                            "before": o["before"]["score"],
                            "after": o["after"]["score"],
                            "delta": o["delta_score"]} for o in outs]))

st.subheader("4️⃣ Guidance")
for s in reco["strategies"]:
    st.write("•", s)
st.info(reco["encouragement"])
st.caption(reco["disclaimer"])
