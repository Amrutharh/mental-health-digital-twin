"""Streamlit dashboard - Layer 7 UI over the 8-layer system.
Run:  streamlit run app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.preprocessing import LabelEncoder

from src import (
    StudentDigitalTwin,
    compare_scenarios,
    load_data,
    prepare_features,
    recommend,
    risk_profile,
    split_and_scale,
    train_and_select,
)

st.set_page_config(
    page_title="Student Mental-Health Digital Twin",
    page_icon="🧠",
    layout="wide",
)

RISK_COLORS = {"Low": "green", "Medium": "orange", "High": "red", "Severe": "red"}


@st.cache_resource(show_spinner="Training models (once, then cached)...")
def get_trained():
    df, used_real = load_data()
    X, y = prepare_features(df)
    label_names = list(LabelEncoder().fit(df["Depression_Type"].astype(str)).classes_)
    Xtr, Xte, ytr, yte, scaler, cols = split_and_scale(X, y)
    best_name, best, _fitted, results = train_and_select(
        Xtr, Xte, ytr, yte, label_names=label_names
    )
    return df, used_real, X, label_names, scaler, cols, best_name, best, results


df, used_real, X, label_names, scaler, cols, best_name, best, results = get_trained()

# ---------------------------------------------------------------- header ---
st.title("Student Mental-Health Digital Twin")
st.caption(
    "8-layer system: preprocess → predict → twin → what-if → "
    "risk → guidance → dashboard → reports"
)

with st.sidebar:
    st.header("Model status")
    st.success(f"Best model: **{best_name}**")
    st.write(f"Real dataset: **{used_real}** (`Depression_Type` target)")
    st.write(f"Rows: **{len(df)}** · Features: **{len(cols)}** · Classes: **{len(label_names)}**")
    st.divider()
    st.subheader("Model F1 (macro)")
    st.dataframe(
        pd.DataFrame(
            [{"model": k, "macro-F1": v["f1_macro"]} for k, v in results.items()]
        ).set_index("model"),
        use_container_width=True,
    )
    st.caption("Best model picked automatically by macro F1.")

# ------------------------------------------------------- 1. live inputs ---
st.header("1. Live student state")
st.caption("Move any slider - prediction, risk and guidance update instantly.")
vals = {}
input_cols = st.columns(3)
for i, f in enumerate(cols):
    box = input_cols[i % 3]
    lo, hi = float(df[f].min()), float(df[f].max())
    med = float(df[f].median())
    with box:
        if hi - lo <= 2:  # binary / flag column
            opts = sorted(df[f].unique().tolist())
            vals[f] = st.selectbox(f.replace("_", " "), opts, index=0)
        else:
            vals[f] = st.slider(
                f.replace("_", " "), float(lo), float(hi), float(med)
            )

twin = StudentDigitalTwin(
    "demo-student",
    vals,
    model=best,
    scaler=scaler,
    feature_columns=cols,
    label_names=label_names,
)
pred = twin.predict()
risk = risk_profile(pred["proba"], twin.state, label_names)
reco = recommend(risk, twin.state)

# --------------------------------------------------- 2. prediction/risk ---
st.header("2. Prediction and risk")
m1, m2, m3 = st.columns(3)
m1.metric("Predicted type", pred["label"])
m2.metric("Risk score", f"{risk['score']} / 30")
m3.metric("Risk level", risk["level"])
st.markdown(f"Risk level: **:{RISK_COLORS.get(risk['level'], 'gray')}[{risk['level']}]**")

top3_df = pd.DataFrame(risk["top3"]).set_index("type") if risk["top3"] else pd.DataFrame()
c_left, c_right = st.columns([1, 1])
with c_left:
    st.subheader("Top-3 likely types")
    for t in risk["top3"]:
        st.write(f"- {t['type']}: {t['prob']:.0%}")
with c_right:
    st.subheader("Probability chart")
    if not top3_df.empty:
        st.bar_chart(top3_df["prob"], use_container_width=True)

# ------------------------------------------------------------- 3. what-if ---
st.header("3. What-if interventions")
st.caption("Simulated on the twin only - original state is restored after each test.")
outs = compare_scenarios(twin)  # auto-picks scenarios for this schema
if outs:
    scenario_df = pd.DataFrame(
        [
            {
                "Scenario": o["scenario"].replace("_", " "),
                "Before": o["before"]["score"],
                "After": o["after"]["score"],
                "Delta": o["delta_score"],
                "Improved?": "Yes" if o["improved"] else "No",
            }
            for o in outs
        ]
    )
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)
else:
    st.warning("No applicable scenarios for this feature schema.")

# ------------------------------------------------------------ 4. guidance ---
st.header("4. Guidance")
st.caption("Rule-based coping plan (paper's 'Generative AI Support Layer').")
if reco["flags"]:
    st.warning("Flags: " + ", ".join(reco["flags"]).replace("_", " "))
g1, g2 = st.columns([2, 1])
with g1:
    st.subheader("Strategies")
    for s in reco["strategies"]:
        st.write("- " + s)
    st.subheader("Weekly roadmap")
    for r in reco["weekly_roadmap"]:
        st.write("- " + r)
with g2:
    st.subheader("Encouragement")
    st.info(reco["encouragement"])

st.divider()
st.caption(reco["disclaimer"])
st.caption("Tip for reviewers: try Sleep_Hours 4 vs 8, or Nervous_Level 1 vs 9, "
           "and watch the risk score and top-3 change.")
