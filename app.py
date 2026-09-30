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
    "About you": ["Age", "Gender", "Education_Level", "Employment_Status"],
    "Daily life": [
        "Sleep_Hours", "SocialMedia_Hours", "SocialMedia_WhileEating",
        "Your overeating level", "How many times you eat ", "Coping_Methods",
    ],
    "Feelings": [
        "Symptoms", "Low_Energy", "Low_SelfEsteem", "Nervous_Level",
        "Depression_Score", "Search_Depression_Online",
        "Worsening_Depression", "Mental_Health_Support",
    ],
    "Need help now": ["Self_Harm", "Suicide_Attempts"],
}

# Simple words a 10-year-old understands: (question shown, extra help line)
FRIENDLY = {
    "Age": ("How old are you?", "Pick your age in years."),
    "Gender": ("Are you a girl or a boy?", "Pick 0 for girl, 1 for boy."),
    "Education_Level": ("Which class do you study in?", "Bigger number = higher class."),
    "Employment_Status": ("Do you also work?", "Pick the number that matches you."),
    "Sleep_Hours": ("How many hours do you sleep at night?", "Most children need 8 to 10 hours."),
    "SocialMedia_Hours": ("How many hours do you watch phone or TV for fun?", "Count play-time on phone or TV."),
    "SocialMedia_WhileEating": ("Do you watch phone while eating?", "Bigger number = more often."),
    "Your overeating level": ("Do you eat too much food?", "Bigger number = eats too much."),
    "How many times you eat ": ("How many times do you eat in a day?", "Count breakfast, lunch, dinner and snacks."),
    "Coping_Methods": ("What do you do when you feel sad?", "Pick the number that matches you."),
    "Symptoms": ("How many problems do you feel?", "Counts things like sadness and tiredness."),
    "Low_Energy": ("Do you feel tired all the time?", "No or Yes."),
    "Low_SelfEsteem": ("Do you feel you are not good enough?", "No or Yes."),
    "Nervous_Level": ("How nervous or scared do you feel?", "Bigger number = more scared."),
    "Depression_Score": ("How sad is your heart?", "Bigger number = more sad."),
    "Search_Depression_Online": ("Did you search the internet for help?", "No or Yes."),
    "Worsening_Depression": ("Is your sadness getting bigger?", "No or Yes."),
    "Mental_Health_Support": ("Is somebody helping you?", "No or Yes."),
    "Self_Harm": ("Did you hurt yourself?", "No or Yes. Tell a teacher now if Yes."),
    "Suicide_Attempts": ("Did you try to end your life?", "No or Yes. Tell an adult now if Yes."),
}


@st.cache_resource(show_spinner="Training models (once, then cached)...")
def get_trained():
    from src.config import DEPRESSION_TYPES

    df, used_real = load_data()
    X, y = prepare_features(df)
    raw_names = list(LabelEncoder().fit(df["Depression_Type"].astype(str)).classes_)
    # Real CSV uses numeric codes 0-11: show human-readable names instead.
    label_names = []
    for c in raw_names:
        try:
            label_names.append(DEPRESSION_TYPES[int(float(c))])
        except (ValueError, IndexError):
            label_names.append(str(c))
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
    st.header("About this app")
    st.write(
        "This app asks you simple questions about your day and your feelings. "
        "At the end it tells you how you are doing and what you can do to feel better."
    )
    st.divider()
    st.subheader("How to use (3 steps)")
    st.write("1. Answer the questions in Step 1.")
    st.write("2. See your result in Step 2.")
    st.write("3. See what to do next in Steps 3 and 4.")
    st.caption("This app only gives friendly advice. It is not a doctor.")

# ------------------------------------------------------- 1. live inputs ---
st.subheader("Step 1 · Answer these questions about you")
st.caption("Tap each tab and answer. These are the questions the app is asking you.")

YES_NO = {"No": 0, "Yes": 1}

# Fixed answer type per question (data ranges alone guess wrong).
# "yesno" -> No/Yes dropdown · "number" -> whole-number slider ·
# "options" -> dropdown of that column's real choices.
WIDGETS = {
    "Age": "number",
    "Education_Level": "options",
    "Employment_Status": "options",
    "Sleep_Hours": "number",
    "SocialMedia_Hours": "number",
    "SocialMedia_WhileEating": "yesno",
    "Your overeating level": "yesno",
    "How many times you eat ": "number",
    "Coping_Methods": "options",
    "Symptoms": "number",
    "Low_Energy": "yesno",
    "Low_SelfEsteem": "yesno",
    "Nervous_Level": "number",
    "Depression_Score": "number",
    "Search_Depression_Online": "yesno",
    "Worsening_Depression": "yesno",
    "Mental_Health_Support": "yesno",
    "Self_Harm": "yesno",
    "Suicide_Attempts": "yesno",
}

vals: dict = {}
tabs = st.tabs(list(GROUPS.keys()))
for tab, (gname, feats) in zip(tabs, GROUPS.items()):
    with tab:
        present = [f for f in feats if f in col_set]
        extra = [f for f in cols if f not in sum(GROUPS.values(), [])]
        show = present + (extra if gname == "Feelings" else [])
        if not show:
            st.write("No questions here.")
            continue
        tcols = st.columns(2)
        for i, f in enumerate(show):
            box = tcols[i % 2]
            lo, hi = float(df[f].min()), float(df[f].max())
            med = float(df[f].median())
            label, hint = FRIENDLY.get(f, (f.strip().replace("_", " "), ""))
            kind = WIDGETS.get(f)
            if kind is None:  # fallback guess for unseen columns
                kind = "yesno" if (hi - lo <= 2) else "number"
            with box:
                if f == "Gender":
                    choice = st.selectbox(label, ["Girl", "Boy"], index=0,
                                          key=f"in_{f}", help=hint)
                    vals[f] = 0 if choice == "Girl" else 1
                elif kind == "yesno":
                    choice = st.selectbox(label, ["No", "Yes"], index=0,
                                          key=f"in_{f}", help=hint)
                    vals[f] = YES_NO[choice]
                elif kind == "options":
                    opts = sorted(df[f].unique().tolist())
                    pick = st.selectbox(label, opts, index=0,
                                        key=f"in_{f}", help=hint)
                    vals[f] = pick
                else:
                    # All inputs are whole numbers (no decimals anywhere).
                    vals[f] = st.slider(label, int(round(lo)), int(round(hi)),
                                        int(round(med)), step=1,
                                        key=f"in_{f}", help=hint)
                if hint:
                    st.caption(hint)
# any column not covered (safety net)
missing = [f for f in cols if f not in vals]
if missing:
    with st.expander("More questions"):
        for f in missing:
            label, hint = FRIENDLY.get(f, (f.strip().replace("_", " "), ""))
            vals[f] = st.slider(label, int(round(float(df[f].min()))),
                                int(round(float(df[f].max()))),
                                int(round(float(df[f].median()))), step=1)

twin = StudentDigitalTwin(
    "demo-student", vals, model=best,
    scaler=scaler, feature_columns=cols, label_names=label_names,
)
pred = twin.predict()
risk = risk_profile(pred["proba"], twin.state, label_names)
reco = recommend(risk, twin.state)

# --------------------------------------------------- 2. prediction/risk ---
st.subheader("Step 2 · Your result (what the app gives you)")
st.caption("This is what the app tells you after reading your answers.")
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("How are you?", pred["label"])
    st.caption("This is what the computer thinks about your answers.")
with k2:
    st.metric("Care score", f"{risk['score']} / 30")
    st.caption("0 means fine. 30 means you need help right now.")
with k3:
    st.metric("Care level", RISK_BADGE.get(risk["level"], risk["level"]))
    st.caption("Green is good. Red means talk to an adult today.")

left, right = st.columns([1, 1])
with left:
    st.markdown("### What may be happening?")
    st.caption("The top guesses, with how sure the computer is.")
    for t in risk["top3"]:
        st.progress(float(t["prob"]), text=f"{t['type']} - {t['prob']:.0%}")
with right:
    st.markdown("### How sure is the computer?")
    top3_df = pd.DataFrame(risk["top3"]).set_index("type") if risk["top3"] else pd.DataFrame()
    if not top3_df.empty:
        st.bar_chart(top3_df["prob"], use_container_width=True)

# ------------------------------------------------------------- 3. what-if ---
st.subheader("Step 3 · What happens if you change something?")
st.caption("Example: what if you sleep 8 hours? The app tries it safely - nothing about you really changes.")
outs = compare_scenarios(twin)
if outs:
    sdf = pd.DataFrame(
        [
            {
                "Try this": o["scenario"].replace("_", " "),
                "Score before": o["before"]["score"],
                "Score after": o["after"]["score"],
                "Better?": "Yes" if o["improved"] else ("Same" if o["delta_score"] == 0 else "No"),
            }
            for o in outs
        ]
    )
    st.dataframe(sdf, use_container_width=True, hide_index=True)
else:
    st.warning("No examples to show right now.")

# ------------------------------------------------------------ 4. guidance ---
st.subheader("Step 4 · What should you do next?")
st.caption("These are the tips the app gives you at the end.")
if reco["flags"]:
    st.warning("⚠️ The app noticed: " + ", ".join(reco["flags"]).replace("_", " "))
gg1, gg2 = st.columns([2, 1])
with gg1:
    st.markdown("### ✅ Strategies")
    for s in reco["strategies"]:
        st.write("- " + s)
    st.markdown("### 🗓️ Weekly roadmap")
    for r in reco["weekly_roadmap"]:
        st.write("- " + r)
with gg2:
    st.markdown("### 💬 Encouragement")
    st.info(reco["encouragement"])

st.divider()
st.subheader("All done! What just happened?")
st.write(
    "**What the app asked you:** simple questions about your day, "
    "your sleep, your phone time and your feelings (Step 1)."
)
st.write(
    "**What the app gave you:** how you seem right now (Step 2), "
    "what would happen if you change one habit (Step 3), "
    "and what to do next (Step 4)."
)
st.caption(reco["disclaimer"])
st.caption("Try this: change your sleep to 8 hours and watch your score get better.")
