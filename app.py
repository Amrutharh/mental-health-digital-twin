"""Student Mental-Health Digital Twin - Layer 7 dashboard.
Run:  streamlit run app.py

7-indicator assessment over a 4-model backend (best selected by macro F1).
Remaining features are median-imputed so the twin stays fully specified.
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

st.markdown(
    """
<style>
.hero {
  background: linear-gradient(120deg, #0EA5E9 0%, #6366F1 55%, #8B5CF6 100%);
  border-radius: 16px; padding: 24px 28px; margin-bottom: 16px;
  color: white;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2rem; }
.hero p { margin: 2px 0; opacity: 0.92; }
.qcard {
  background: #151F35; border: 1px solid #263252;
  border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
}
.qcard p { margin: 0 0 2px 0; font-size: 1rem; font-weight: 600; }
.badge {
  display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-weight: 700; font-size: 0.95rem;
}
.badge-low { background: #14532D; color: #BBF7D0; }
.badge-medium { background: #713F12; color: #FDE68A; }
.badge-high { background: #7C2D12; color: #FED7AA; }
.badge-severe { background: #7F1D1D; color: #FECACA; }
</style>
""",
    unsafe_allow_html=True,
)

BADGE = {
    "Low": "<span class='badge badge-low'>LOW RISK</span>",
    "Medium": "<span class='badge badge-medium'>MODERATE RISK</span>",
    "High": "<span class='badge badge-high'>HIGH RISK</span>",
    "Severe": "<span class='badge badge-severe'>SEVERE RISK</span>",
}

# 7 sentinel indicators (top model drivers, answerable in <60s).
# kind: "number" -> integer slider, "yesno" -> No/Yes mapped to low/high quartiles.
QUESTIONS = [
    ("Sleep_Hours", "Sleep duration (hrs/night)",
     "Self-reported nightly sleep. Reference: 7-9 hrs.", "number"),
    ("Nervous_Level", "Nervousness (0-10)",
     "Current anxiety/arousal level.", "number"),
    ("Symptoms", "Somatic symptom score (0-14)",
     "Count of reported psycho-somatic symptoms.", "number"),
    ("Low_Energy", "Persistent low energy?",
     "Fatigue most days over the last 2 weeks.", "yesno"),
    ("Low_SelfEsteem", "Low self-esteem?",
     "Feelings of worthlessness most days.", "yesno"),
    ("Your overeating level", "Disordered overeating?",
     "Eating beyond satiety regularly.", "yesno"),
    ("SocialMedia_WhileEating", "Device use while eating?",
     "Screen use during meals.", "yesno"),
]

SCENARIO_NAMES = {
    "better_sleep": "Sleep hygiene - 8 hrs/night",
    "calm_nervousness": "Anxiety reduction - nervousness to 1",
    "less_social_media": "Digital detox - 2 hrs/day social media",
    "stop_self_harm": "Safety plan - self-harm and attempts to 0",
}


@st.cache_resource(show_spinner="Training ensemble (one-time)...")
def get_trained():
    from src.config import DEPRESSION_TYPES

    df, used_real = load_data()
    X, y = prepare_features(df)
    raw_names = list(LabelEncoder().fit(df["Depression_Type"].astype(str)).classes_)
    label_names = []
    for c in raw_names:
        try:
            label_names.append(DEPRESSION_TYPES[int(float(c))])
        except (ValueError, IndexError):
            label_names.append(str(c))
    Xtr, Xte, ytr, yte, scaler, cols = split_and_scale(X, y)
    best_name, best, _fitted, results = train_and_select(Xtr, Xte, ytr, yte)
    medians = {c: float(X[c].median()) for c in cols}
    q25 = {c: float(X[c].quantile(0.25)) for c in cols}
    q75 = {c: float(X[c].quantile(0.75)) for c in cols}
    return df, used_real, X, label_names, scaler, cols, best_name, best, results, medians, q25, q75


(df, used_real, X, label_names, scaler, cols, best_name, best,
 results, medians, q25, q75) = get_trained()
col_set = set(cols)

# ---------------------------------------------------------------- header ---
st.markdown(
    """<div class="hero">
<h1>🧠 Student Mental-Health Digital Twin</h1>
<p>8-layer system: preprocessing → 4-model ensemble (best-F1 selection) →
live digital twin → counterfactual simulation → 0-30 risk stratification →
tiered intervention plan.</p>
<p>Dataset: 1,998 records · 20 features · 12 depression subtypes ·
target <b>Depression_Type</b>.</p>
</div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment status")
    st.success(f"Selected model: **{best_name}**")
    c_a, c_b = st.columns(2)
    c_a.metric("Records", len(df))
    c_b.metric("Features", len(cols))
    st.write(f"Real dataset: **{used_real}** · Classes: **{len(label_names)}**")
    st.divider()
    st.subheader("Macro-F1 by candidate")
    st.dataframe(
        pd.DataFrame(
            [{"Model": k, "Macro-F1": v["f1_macro"]} for k, v in results.items()]
        ).set_index("Model"),
        use_container_width=True,
    )
    st.caption("Selection criterion: macro F1 (robust to class imbalance).")
    st.divider()
    st.subheader("Protocol")
    st.write("1. Complete the 7-indicator assessment.")
    st.write("2. Review prediction and risk stratum.")
    st.write("3. Inspect counterfactual interventions.")
    st.write("4. Follow the tiered action plan.")
    st.caption("Decision support only - not a clinical diagnosis.")

# ------------------------------------------------------- 1. assessment ---
st.subheader("1 · Clinical assessment (7 sentinel indicators)")
st.caption("Remaining features are median-imputed so the twin stays fully specified.")

answers: dict = {}
left, right = st.columns(2)
for i, (col, question, hint, kind) in enumerate(QUESTIONS):
    box = left if i % 2 == 0 else right
    with box:
        st.markdown("<div class='qcard'>", unsafe_allow_html=True)
        st.markdown(f"<p>{question}</p>", unsafe_allow_html=True)
        if col not in col_set:
            st.caption("Unavailable in current schema.")
        elif kind == "yesno":
            choice = st.radio(question, ["No", "Yes"], horizontal=True,
                              key=f"q_{col}", label_visibility="collapsed")
            answers[col] = q25[col] if choice == "No" else q75[col]
        else:
            lo, hi = int(round(float(df[col].min()))), int(round(float(df[col].max())))
            med = int(round(float(df[col].median())))
            answers[col] = st.slider(question, lo, hi, med, step=1,
                                     key=f"q_{col}", label_visibility="collapsed")
        st.caption(hint)
        st.markdown("</div>", unsafe_allow_html=True)

state = dict(medians)
state.update(answers)

twin = StudentDigitalTwin(
    "demo-student", state, model=best,
    scaler=scaler, feature_columns=cols, label_names=label_names,
)
pred = twin.predict()
risk = risk_profile(pred["proba"], twin.state, label_names)
reco = recommend(risk, twin.state)

# ------------------------------------------------- 2. prediction / risk ---
st.subheader("2 · Prediction and risk stratification")
k1, k2, k3 = st.columns(3)
k1.metric("Predicted subtype", pred["label"])
k2.metric("Risk score", f"{risk['score']} / 30")
k3.metric("Risk stratum", risk["level"])
st.markdown(BADGE.get(risk["level"], risk["level"]), unsafe_allow_html=True)
st.caption("Strata: Low 0-9 · Moderate 10-15 · High 16-22 · Severe 23-30.")

left, right = st.columns([1, 1])
with left:
    st.markdown("### Posterior top-3")
    for t in risk["top3"]:
        st.progress(float(t["prob"]), text=f"{t['type']} - {t['prob']:.0%}")
with right:
    st.markdown("### Posterior distribution")
    top3_df = pd.DataFrame(risk["top3"]).set_index("type") if risk["top3"] else pd.DataFrame()
    if not top3_df.empty:
        st.bar_chart(top3_df["prob"], use_container_width=True)

# ------------------------------------------------------ 3. counterfactual ---
st.subheader("3 · Counterfactual intervention simulation")
st.caption("Each row: perturb the twin, re-run inference, report the delta, "
           "then restore baseline. The physical student is never touched.")
outs = compare_scenarios(twin)
if outs:
    best = min(outs, key=lambda o: o["after"]["score"])
    if best["improved"]:
        st.success(
            f"Highest-leverage intervention: "
            f"**{SCENARIO_NAMES.get(best['scenario'], best['scenario'].replace('_', ' '))}** - "
            f"risk {best['before']['score']} → {best['after']['score']}."
        )
    sdf = pd.DataFrame(
        [
            {
                "Intervention": SCENARIO_NAMES.get(o["scenario"], o["scenario"].replace("_", " ")),
                "Risk (pre)": o["before"]["score"],
                "Risk (post)": o["after"]["score"],
                "P(top) pre": f"{o['before']['top3'][0]['prob']:.0%}" if o["before"]["top3"] else "-",
                "P(top) post": f"{o['after']['top3'][0]['prob']:.0%}" if o["after"]["top3"] else "-",
                "Outcome": ("Improved" if o["improved"]
                            else ("No measurable delta" if o["delta_score"] == 0 else "Deteriorated")),
            }
            for o in outs
        ]
    )
    st.dataframe(sdf, use_container_width=True, hide_index=True)
    st.caption("Risk is quantised to integers, so sub-threshold shifts surface only "
               "in the posterior columns. Flat rows indicate the ensemble's decision "
               "boundary is insensitive to that perturbation for this state - itself "
               "an informative twin readout.")
else:
    st.warning("No applicable scenarios for this feature schema.")

# ------------------------------------------------------------ 4. action ---
st.subheader("4 · Tiered action plan")
st.caption("Rule-based intervention layer keyed to risk stratum and flagged features.")
if reco["flags"]:
    st.warning("Clinical flags: " + ", ".join(reco["flags"]).replace("_", " "))
improved = [o for o in outs if o["improved"]] if outs else []
same = [o for o in outs if not o["improved"]] if outs else []

st.markdown("### Required behaviour changes")
QUESTION_OF = {col: q for col, q, _h, _k in QUESTIONS}
KIND_OF = {col: k for col, _q, _h, k in QUESTIONS}


def _word(col, value):
    if KIND_OF.get(col) == "yesno":
        return "Yes" if value >= (q25[col] + q75[col]) / 2 else "No"
    return f"{round(value)}"


if improved:
    for n, o in enumerate(improved, 1):
        st.write(
            f"**{n}. {SCENARIO_NAMES.get(o['scenario'], o['scenario'].replace('_', ' '))}** "
            f"- risk {o['before']['score']} → {o['after']['score']}"
        )
        for col, new_v in o["changes"].items():
            q = QUESTION_OF.get(col, col)
            st.caption(f"↳ {q}: current {_word(col, twin.state[col])} → target {_word(col, new_v)}")
else:
    st.write("- No escalation indicated - maintain sleep, hydration and activity baselines.")

if same:
    with st.expander("Low-yield perturbations (informative negatives)"):
        for o in same:
            st.write(
                f"- {SCENARIO_NAMES.get(o['scenario'], o['scenario'].replace('_', ' '))}: "
                f"risk unchanged at {o['before']['score']} - insufficient as monotherapy."
            )

st.markdown("### Immediate strategies")
for s in reco["strategies"]:
    st.write("- " + s)
st.markdown("### 7-day structured plan")
for i, r in enumerate(reco["weekly_roadmap"], 1):
    st.write(f"**Day {i}:** {r}")
st.info(reco["encouragement"])

st.divider()
st.subheader("Run summary")
st.write("**Inputs:** 7 sentinel indicators (remainder median-imputed over 20 features).")
st.write("**Outputs:** predicted subtype, 0-30 risk stratum, counterfactual ranking, tiered 7-day plan.")
st.caption(reco["disclaimer"])
st.caption("Stack: Python · scikit-learn · XGBoost · Streamlit · 8-layer digital-twin architecture.")
