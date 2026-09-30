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
    page_title="MindMirror - Student Digital Twin",
    page_icon="🧠",
    layout="wide",
)

# ------------------------------------------------------------- styling ---
st.markdown(
    """
<style>
.hero {
  background: linear-gradient(120deg, #14B8A6 0%, #6366F1 60%, #A855F7 100%);
  border-radius: 18px; padding: 26px 28px; margin-bottom: 18px;
  color: white;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; }
.hero p { margin: 0; opacity: 0.92; }
.card {
  background: #151F35; border: 1px solid #263252;
  border-radius: 14px; padding: 16px 18px; margin-bottom: 14px;
}
.card h3 { margin-top: 0; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background: #151F35; border-radius: 10px 10px 0 0;
  padding: 8px 18px; border: 1px solid #263252;
}
.small { color: #94A3B8; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

RISK_BADGE = {
    "Low": "🟢 Low",
    "Medium": "🟡 Medium",
    "High": "🟠 High",
    "Severe": "🔴 Severe",
}

GROUPS = {
    "👤 Profile": ["Age", "Gender", "Education_Level", "Employment_Status"],
    "🌙 Lifestyle": [
        "Sleep_Hours", "SocialMedia_Hours", "SocialMedia_WhileEating",
        "Your overeating level", "How many times you eat ", "Coping_Methods",
    ],
    "💭 Mind signals": [
        "Symptoms", "Low_Energy", "Low_SelfEsteem", "Nervous_Level",
        "Depression_Score", "Search_Depression_Online",
        "Worsening_Depression", "Mental_Health_Support",
    ],
    "🚨 Risk flags": ["Self_Harm", "Suicide_Attempts"],
}


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
col_set = set(cols)

# ---------------------------------------------------------------- header ---
st.markdown(
    """<div class="hero">
<h1>🧠 MindMirror - Student Digital Twin</h1>
<p>Live mirror of student wellbeing: what-if simulations ·
risk score · personalised guidance. Move the sliders - everything updates.</p>
</div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write(
        "MindMirror is a supportive wellbeing check-in. "
        "Adjust the inputs and see your predicted state, risk and guidance."
    )
    st.divider()
    st.subheader("How to use")
    st.write("1. Fill the tabs under Live student state.")
    st.write("2. Read your prediction and risk below.")
    st.write("3. Try the what-if table to preview improvements.")
    st.caption("Supportive guidance only - not a medical diagnosis.")

# ------------------------------------------------------- 1. live inputs ---
st.subheader("1 · Live student state")
st.caption("Grouped like a real health app - edit values inside each tab.")

vals: dict = {}
tabs = st.tabs(list(GROUPS.keys()))
for tab, (gname, feats) in zip(tabs, GROUPS.items()):
    with tab:
        present = [f for f in feats if f in col_set]
        extra = [f for f in cols if f not in sum(GROUPS.values(), [])]
        show = present + (extra if gname == "💭 Mind signals" else [])
        if not show:
            st.write("No fields in this group for the current dataset.")
            continue
        tcols = st.columns(2)
        for i, f in enumerate(show):
            box = tcols[i % 2]
            lo, hi = float(df[f].min()), float(df[f].max())
            med = float(df[f].median())
            label = f.strip().replace("_", " ")
            with box:
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                if hi - lo <= 2:
                    opts = sorted(df[f].unique().tolist())
                    vals[f] = st.selectbox(label, opts, index=0, key=f"in_{f}")
                    st.markdown(
                        f"<span class='small'>Flag · 0/1 style</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    # All inputs are whole numbers (no decimals anywhere).
                    vals[f] = st.slider(label, int(round(lo)), int(round(hi)),
                                        int(round(med)), step=1, key=f"in_{f}")
                    st.markdown(
                        f"<span class='small'>Range {lo:.0f}–{hi:.0f} · "
                        f"typical {med:.0f}</span>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
# any column not covered (safety net)
missing = [f for f in cols if f not in vals]
if missing:
    with st.expander("Other fields"):
        for f in missing:
            vals[f] = st.slider(f, float(df[f].min()), float(df[f].max()),
                                float(df[f].median()))

twin = StudentDigitalTwin(
    "demo-student", vals, model=best,
    scaler=scaler, feature_columns=cols, label_names=label_names,
)
pred = twin.predict()
risk = risk_profile(pred["proba"], twin.state, label_names)
reco = recommend(risk, twin.state)

# --------------------------------------------------- 2. prediction/risk ---
st.subheader("2 · Prediction and risk")
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Predicted type", pred["label"])
    st.markdown("</div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Risk score", f"{risk['score']} / 30")
    st.markdown("</div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Risk level", RISK_BADGE.get(risk["level"], risk["level"]))
    st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns([1, 1])
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Top-3 likely types")
    for t in risk["top3"]:
        st.progress(float(t["prob"]), text=f"{t['type']}: {t['prob']:.0%}")
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Probability chart")
    top3_df = pd.DataFrame(risk["top3"]).set_index("type") if risk["top3"] else pd.DataFrame()
    if not top3_df.empty:
        st.bar_chart(top3_df["prob"], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------- 3. what-if ---
st.subheader("3 · What-if interventions")
st.caption("Tested on the twin only - the student's real state is never touched.")
outs = compare_scenarios(twin)
if outs:
    sdf = pd.DataFrame(
        [
            {
                "Scenario": o["scenario"].replace("_", " "),
                "Before": o["before"]["score"],
                "After": o["after"]["score"],
                "Delta": o["delta_score"],
                "Result": "Improved" if o["improved"] else ("Same" if o["delta_score"] == 0 else "Worse"),
            }
            for o in outs
        ]
    )
    st.dataframe(sdf, use_container_width=True, hide_index=True)
else:
    st.warning("No applicable scenarios for this feature schema.")

# ------------------------------------------------------------ 4. guidance ---
st.subheader("4 · Personal guidance")
if reco["flags"]:
    st.warning("⚠️ Flags: " + ", ".join(reco["flags"]).replace("_", " "))
gg1, gg2 = st.columns([2, 1])
with gg1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ✅ Strategies")
    for s in reco["strategies"]:
        st.write("- " + s)
    st.markdown("### 🗓️ Weekly roadmap")
    for r in reco["weekly_roadmap"]:
        st.write("- " + r)
    st.markdown("</div>", unsafe_allow_html=True)
with gg2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 💬 Encouragement")
    st.info(reco["encouragement"])
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption(reco["disclaimer"])
st.caption("Reviewer tip: set Sleep_Hours 4 vs 8, or Nervous_Level 1 vs 9, "
           "and watch the score, top-3 bars and what-if table move.")
