"""Student Mental-Health Digital Twin - Layer 7 dashboard.
Run:  streamlit run app.py

Paged flow: Assessment -> Results -> Simulations -> Action Plan -> Summary.
7 sentinel indicators; remainder median-imputed.
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
  background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #0EA5E9 100%);
  border-radius: 16px; padding: 24px 28px; margin-bottom: 16px;
  color: white;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2rem; color: white; }
.hero p { margin: 2px 0; opacity: 0.93; color: white; }
.qcard {
  background: #F8FAFC; border: 1px solid #E2E8F0;
  border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
}
.qcard p { margin: 0 0 2px 0; font-size: 1rem; font-weight: 600; color: #0F172A; }
.badge {
  display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-weight: 700; font-size: 0.95rem;
}
.badge-low { background: #DCFCE7; color: #166534; }
.badge-medium { background: #FEF9C3; color: #854D0E; }
.badge-high { background: #FFEDD5; color: #9A3412; }
.badge-severe { background: #FEE2E2; color: #991B1B; }
.win {
  background: linear-gradient(120deg, #ECFDF5 0%, #EFF6FF 100%);
  border: 2px solid #6EE7B7; border-radius: 16px;
  padding: 22px 24px; margin: 14px 0;
}
.navbtn button { font-weight: 700; }
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

PAGES = ["1 · Assessment", "2 · Results", "3 · Simulations",
         "4 · Action Plan", "5 · Summary"]


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

if "page" not in st.session_state:
    st.session_state.page = 0


def goto(i: int):
    st.session_state.page = max(0, min(len(PAGES) - 1, i))


def current_answers() -> dict:
    """Answers from widgets if rendered, else sensible defaults."""
    out = {}
    for col, _q, _h, kind in QUESTIONS:
        if col not in col_set:
            continue
        key = f"q_{col}"
        if kind == "yesno":
            if key in st.session_state:
                out[col] = q25[col] if st.session_state[key] == "No" else q75[col]
            else:
                out[col] = q25[col]
        else:
            if key in st.session_state:
                out[col] = float(st.session_state[key])
            else:
                out[col] = float(round(float(df[col].median())))
    return out


def build_twin(state: dict):
    twin = StudentDigitalTwin(
        "demo-student", state, model=best,
        scaler=scaler, feature_columns=cols, label_names=label_names,
    )
    pred = twin.predict()
    risk = risk_profile(pred["proba"], twin.state, label_names)
    reco = recommend(risk, twin.state)
    outs = compare_scenarios(twin)
    return twin, pred, risk, reco, outs


# ---------------------------------------------------------------- header ---
st.markdown(
    """<div class="hero">
<h1>🧠 Student Mental-Health Digital Twin</h1>
<p>8-layer system: preprocessing → 4-model ensemble (best-F1) →
live digital twin → counterfactual simulation → 0-30 risk stratification →
tiered intervention plan.</p>
<p>Dataset: 1,998 records · 20 features · 12 subtypes · target
<b>Depression_Type</b>.</p>
</div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment status")
    st.success(f"Selected model: **{best_name}**")
    c_a, c_b = st.columns(2)
    c_a.metric("Records", len(df))
    c_b.metric("Features", len(cols))
    st.divider()
    st.subheader("Navigate")
    choice = st.radio("Go to page", PAGES, index=st.session_state.page,
                      label_visibility="collapsed")
    goto(PAGES.index(choice))
    st.divider()
    st.subheader("Macro-F1 by candidate")
    st.dataframe(
        pd.DataFrame(
            [{"Model": k, "Macro-F1": v["f1_macro"]} for k, v in results.items()]
        ).set_index("Model"),
        use_container_width=True,
    )
    st.caption("Decision support only - not a clinical diagnosis.")

page = st.session_state.page
state = dict(medians)
state.update(current_answers())
twin, pred, risk, reco, outs = build_twin(state)

# ------------------------------------------------------------- Page 1 ---
if page == 0:
    st.subheader("Step 1 · Clinical assessment (7 sentinel indicators)")
    st.caption("Remaining features are median-imputed so the twin stays fully specified.")
    left, right = st.columns(2)
    for i, (col, question, hint, kind) in enumerate(QUESTIONS):
        box = left if i % 2 == 0 else right
        with box:
            st.markdown("<div class='qcard'>", unsafe_allow_html=True)
            st.markdown(f"<p>{question}</p>", unsafe_allow_html=True)
            if col not in col_set:
                st.caption("Unavailable in current schema.")
            elif kind == "yesno":
                st.radio(question, ["No", "Yes"], horizontal=True,
                         key=f"q_{col}", label_visibility="collapsed")
            else:
                lo = int(round(float(df[col].min())))
                hi = int(round(float(df[col].max())))
                med = int(round(float(df[col].median())))
                st.slider(question, lo, hi, med, step=1,
                          key=f"q_{col}", label_visibility="collapsed")
            st.caption(hint)
            st.markdown("</div>", unsafe_allow_html=True)
    st.button("Next: see results →", on_click=goto, args=(1,), type="primary")

# ------------------------------------------------------------- Page 2 ---
elif page == 1:
    st.subheader("Step 2 · Prediction and risk stratification")
    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted subtype", pred["label"])
    k2.metric("Risk score", f"{risk['score']} / 30")
    k3.metric("Risk stratum", risk["level"])
    st.markdown(BADGE.get(risk["level"], risk["level"]), unsafe_allow_html=True)
    st.caption("Strata: Low 0-9 · Moderate 10-15 · High 16-22 · Severe 23-30.")
    left, right = st.columns(2)
    with left:
        st.markdown("### Posterior top-3")
        for t in risk["top3"]:
            st.progress(float(t["prob"]), text=f"{t['type']} - {t['prob']:.0%}")
    with right:
        st.markdown("### Posterior distribution")
        top3_df = pd.DataFrame(risk["top3"]).set_index("type") if risk["top3"] else pd.DataFrame()
        if not top3_df.empty:
            st.bar_chart(top3_df["prob"], use_container_width=True)
    c1, c2 = st.columns(2)
    c1.button("← Back to assessment", on_click=goto, args=(0,))
    c2.button("Next: try simulations →", on_click=goto, args=(2,), type="primary")

# ------------------------------------------------------------- Page 3 ---
elif page == 2:
    st.subheader("Step 3 · Counterfactual intervention simulation")
    st.caption("Perturb the twin, re-run inference, report the delta, restore baseline.")
    if outs:
        top = min(outs, key=lambda o: o["after"]["score"])
        if top["improved"]:
            st.success(
                f"Highest-leverage intervention: "
                f"**{SCENARIO_NAMES.get(top['scenario'], top['scenario'].replace('_', ' '))}** - "
                f"risk {top['before']['score']} → {top['after']['score']}."
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
        st.caption("Quantised integer risk hides sub-threshold shifts - see posterior columns.")
    else:
        st.warning("No applicable scenarios for this feature schema.")
    c1, c2 = st.columns(2)
    c1.button("← Back to results", on_click=goto, args=(1,))
    c2.button("Next: action plan →", on_click=goto, args=(3,), type="primary")

# ------------------------------------------------------------- Page 4 ---
elif page == 3:
    st.subheader("Step 4 · Tiered action plan")
    st.caption("Rule-based intervention layer keyed to stratum and flagged features.")
    if reco["flags"]:
        st.warning("Clinical flags: " + ", ".join(reco["flags"]).replace("_", " "))
    improved = [o for o in outs if o["improved"]] if outs else []
    same = [o for o in outs if not o["improved"]] if outs else []
    QUESTION_OF = {col: q for col, q, _h, _k in QUESTIONS}
    KIND_OF = {col: k for col, _q, _h, k in QUESTIONS}

    def _word(col, value):
        if KIND_OF.get(col) == "yesno":
            return "Yes" if value >= (q25[col] + q75[col]) / 2 else "No"
        return f"{round(value)}"

    st.markdown("### Required behaviour changes")
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
        st.write("- No escalation indicated - maintain baselines.")
    if same:
        with st.expander("Low-yield perturbations (informative negatives)"):
            for o in same:
                st.write(
                    f"- {SCENARIO_NAMES.get(o['scenario'], o['scenario'].replace('_', ' '))}: "
                    f"unchanged at {o['before']['score']}."
                )
    st.markdown("### Immediate strategies")
    for s in reco["strategies"]:
        st.write("- " + s)
    st.markdown("### 7-day structured plan (tick each day as you finish)")
    for i, r in enumerate(reco["weekly_roadmap"], 1):
        done = st.checkbox(f"Day {i}: {r}", key=f"day_{i}_{risk['level']}")
        if done:
            st.caption(f"✓ Day {i} complete - momentum builds recovery.")
    st.info(reco["encouragement"])
    c1, c2 = st.columns(2)
    c1.button("← Back to simulations", on_click=goto, args=(2,))
    c2.button("Finish: see summary →", on_click=goto, args=(4,), type="primary")

# ------------------------------------------------------------- Page 5 ---
else:
    st.subheader("Your journey, celebrated 🎉")
    if risk["level"] == "Low":
        st.balloons()
    st.markdown(
        """<div class="win">
<h3>You did something strong today 💪</h3>
<p>You answered honestly, faced your result, tested what you can change,
and walked out with a plan. That is exactly how change starts -
<b>one small habit, one day at a time</b>.</p>
</div>""",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("You told us", "7 answers")
    m2.metric("Your score", f"{risk['score']} / 30")
    m3.metric("Your level", risk["level"])
    st.markdown(BADGE.get(risk["level"], risk["level"]), unsafe_allow_html=True)
    best_line = ""
    if outs:
        top = min(outs, key=lambda o: o["after"]["score"])
        if top["improved"]:
            best_line = (f"Your most powerful move: "
                         f"**{SCENARIO_NAMES.get(top['scenario'], top['scenario'].replace('_', ' '))}** "
                         f"(score {top['before']['score']} → {top['after']['score']}).")
    if best_line:
        st.success(best_line)
    st.write("**Remember:** " + reco["encouragement"])
    st.caption("Week plan progress and all results stay on the previous pages - "
               "revisit any step from the sidebar.")
    st.caption(reco["disclaimer"])
    st.caption("Stack: Python · scikit-learn · XGBoost · Streamlit · 8-layer digital-twin architecture.")
    st.button("↺ Start over", on_click=goto, args=(0,))
