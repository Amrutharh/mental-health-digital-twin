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
                 risk_trend, recommend, compare_scenarios, DEFAULT_SCENARIOS)
from src.config import FEATURES
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Mental-Health Digital Twin", layout="wide")
st.title("🧠 Student Mental-Health Digital Twin")
st.caption("8-layer system: preprocess → predict → twin → what-if → risk → guidance → dashboard → reports")

with st.spinner("Training models…"):
    df, used_real = load_data()
    X, y = prepare_features(df)
    label_names = list(LabelEncoder().fit(df["Depression_Type"].astype(str)).classes_)
    Xtr, Xte, ytr, yte, scaler, cols = split_and_scale(X, y)
    best_name, best, fitted, results = train_and_select(
        Xtr, Xte, ytr, yte, label_names=label_names)

st.sidebar.success(f"Best model: **{best_name}**  |  real data: {used_real}")
st.sidebar.subheader("Model F1 (macro)")
for k, v in results.items():
    st.sidebar.write(f"{k}: {v['f1_macro']:.3f}")

st.subheader("1️⃣ Live student state")
c1, c2, c3 = st.columns(3)
vals = {}
vals["Age"] = c1.slider("Age", 15, 30, 20)
vals["Gender"] = c1.selectbox("Gender", [0, 1, 2],
                              format_func=lambda x: ["Female", "Male", "Other"][x])
vals["Sleep_Hours"] = c1.slider("Sleep hours", 2.0, 11.0, 6.0, 0.5)
vals["Appetite_Score"] = c1.slider("Appetite (1-5)", 1, 5, 3)
for i, f in enumerate([f for f in FEATURES if f not in vals]):
    col = [c1, c2, c3][i % 3]
    if f in ("Self_Harm_Flag", "Suicidal_Thoughts", "Past_Depression_History"):
        vals[f] = col.selectbox(f, [0, 1])
    elif f == "Physical_Symptoms_Count":
        vals[f] = col.slider(f, 0, 8, 2)
    else:
        vals[f] = col.slider(f, 0, 10, 4)

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
scenarios = {k: v for k, v in DEFAULT_SCENARIOS.items()}
outs = compare_scenarios(twin, scenarios)
st.dataframe(pd.DataFrame([{"scenario": o["scenario"],
                            "before": o["before"]["score"],
                            "after": o["after"]["score"],
                            "delta": o["delta_score"]} for o in outs]))

st.subheader("4️⃣ Guidance")
for s in reco["strategies"]:
    st.write("•", s)
st.info(reco["encouragement"])
st.caption(reco["disclaimer"])
