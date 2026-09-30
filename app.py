"""MindMirror - a wellbeing check-in even a 10-year-old can use.
Run:  streamlit run app.py

Only 7 questions. Everything else is filled in automatically.
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
    page_title="MindMirror - How are you feeling?",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
.hero {
  background: linear-gradient(120deg, #14B8A6 0%, #6366F1 60%, #A855F7 100%);
  border-radius: 18px; padding: 26px 28px; margin-bottom: 18px;
  color: white;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; }
.hero p { margin: 0; opacity: 0.93; }
.qcard {
  background: #151F35; border: 1px solid #263252;
  border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;
}
.qcard p { margin: 0 0 4px 0; font-size: 1.05rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

RISK_BADGE = {
    "Low": "🟢 Good",
    "Medium": "🟡 Okay - take care",
    "High": "🟠 Needs care",
    "Severe": "🔴 Needs help now",
}

# The only 7 questions the app asks (chosen: top drivers a child can answer).
# kind: "number" -> whole-number slider, "yesno" -> No/Yes buttons.
QUESTIONS = [
    ("Sleep_Hours", "😴 How many hours do you sleep at night?",
     "Most children need 8 to 10 hours.", "number"),
    ("Nervous_Level", "😟 How nervous or scared do you feel?",
     "0 = not scared at all, 10 = very scared.", "number"),
    ("Symptoms", "🤒 How many problems do you feel in your body and mind?",
     "For example sadness, tiredness, headache. Pick a number.", "number"),
    ("Low_Energy", "🔋 Do you feel tired all the time?",
     "Even after sleeping?", "yesno"),
    ("Low_SelfEsteem", "🪞 Do you feel you are not good enough?",
     "Everybody is good enough.", "yesno"),
    ("Your overeating level", "🍔 Do you eat too much food?",
     "More than your tummy needs?", "yesno"),
    ("SocialMedia_WhileEating", "📱 Do you watch the phone while eating?",
     "At breakfast, lunch or dinner?", "yesno"),
]

# Friendly names for the what-if table.
SCENARIO_NAMES = {
    "better_sleep": "💤 Sleep 8 hours every night",
    "calm_nervousness": "😌 Feel calm, not scared",
    "less_social_media": "📵 Less phone time",
    "stop_self_harm": "🤝 Get help and stay safe",
}


@st.cache_resource(show_spinner="Getting ready... (one time only)")
def get_trained():
    from src.config import DEPRESSION_TYPES

    df, _used_real = load_data()
    X, y = prepare_features(df)
    raw_names = list(LabelEncoder().fit(df["Depression_Type"].astype(str)).classes_)
    label_names = []
    for c in raw_names:
        try:
            label_names.append(DEPRESSION_TYPES[int(float(c))])
        except (ValueError, IndexError):
            label_names.append(str(c))
    Xtr, Xte, ytr, yte, scaler, cols = split_and_scale(X, y)
    _name, best, _fitted, _results = train_and_select(Xtr, Xte, ytr, yte)
    medians = {c: float(X[c].median()) for c in cols}
    q25 = {c: float(X[c].quantile(0.25)) for c in cols}
    q75 = {c: float(X[c].quantile(0.75)) for c in cols}
    return df, X, label_names, scaler, cols, best, medians, q25, q75


df, X, label_names, scaler, cols, best, medians, q25, q75 = get_trained()
col_set = set(cols)

# ---------------------------------------------------------------- header ---
st.markdown(
    """<div class="hero">
<h1>🧠 MindMirror</h1>
<p>Answer <b>7 small questions</b>. Then this app tells you
<b>how you are doing</b> and <b>what to do next</b>. Easy!</p>
</div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About this app")
    st.write(
        "You answer 7 questions. The app tells you how you are doing "
        "and gives tips to feel better."
    )
    st.divider()
    st.subheader("Steps")
    st.write("1️⃣ Answer 7 questions.")
    st.write("2️⃣ See your result.")
    st.write("3️⃣ See tips to feel better.")
    st.caption("Friendly advice only. Not a doctor.")

# ------------------------------------------------------------- Step 1 ---
st.subheader("Step 1 · Answer 7 questions")
answers: dict = {}
left, right = st.columns(2)
for i, (col, question, hint, kind) in enumerate(QUESTIONS):
    box = left if i % 2 == 0 else right
    with box:
        st.markdown("<div class='qcard'>", unsafe_allow_html=True)
        st.markdown(f"<p>{question}</p>", unsafe_allow_html=True)
        if col not in col_set:
            st.caption("Skipped for this data.")
            st.markdown("</div>", unsafe_allow_html=True)
            continue
        if kind == "yesno":
            choice = st.radio(question, ["No", "Yes"], horizontal=True,
                              key=f"q_{col}", label_visibility="collapsed")
            # No -> low value, Yes -> high value (keeps the AI sensitive).
            answers[col] = q25[col] if choice == "No" else q75[col]
        else:
            lo, hi = int(round(float(df[col].min()))), int(round(float(df[col].max())))
            med = int(round(float(df[col].median())))
            answers[col] = st.slider(question, lo, hi, med, step=1,
                                     key=f"q_{col}", label_visibility="collapsed")
        st.caption(hint)
        st.markdown("</div>", unsafe_allow_html=True)

# Fill the rest quietly with typical values so the AI still works.
state = dict(medians)
state.update(answers)

twin = StudentDigitalTwin(
    "demo-student", state, model=best,
    scaler=scaler, feature_columns=cols, label_names=label_names,
)
pred = twin.predict()
risk = risk_profile(pred["proba"], twin.state, label_names)
reco = recommend(risk, twin.state)

# ------------------------------------------------------------- Step 2 ---
st.subheader("Step 2 · Your result")
st.caption("This is what the app tells you after reading your answers.")
k1, k2, k3 = st.columns(3)
k1.metric("How are you?", pred["label"])
k2.metric("Care score", f"{risk['score']} / 30")
k3.metric("Care level", RISK_BADGE.get(risk["level"], risk["level"]))
st.caption("Score 0 = happy and fine. Score 30 = talk to a teacher or parent today.")
st.progress(risk["score"] / 30, text=f"Care score: {risk['score']} out of 30")

st.markdown("### What may be happening?")
st.caption("The top guess, and how sure the computer is.")
top = risk["top3"][0] if risk["top3"] else {"type": pred["label"], "prob": 0}
st.progress(float(top["prob"]), text=f"{top['type']} - {top['prob']:.0%} sure")

# ------------------------------------------------------------- Step 3 ---
st.subheader("Step 3 · If you change something, what happens?")
st.caption("Each row says: if you do THIS, your score changes from THIS to THAT.")
outs = compare_scenarios(twin)
if outs:
    best = min(outs, key=lambda o: o["after"]["score"])
    if best["improved"]:
        st.success(
            f"⭐ Best thing to try: **{SCENARIO_NAMES.get(best['scenario'], best['scenario'].replace('_', ' '))}** - "
            f"your score goes from **{best['before']['score']} to {best['after']['score']}**."
        )
    sdf = pd.DataFrame(
        [
            {
                "If you do this": SCENARIO_NAMES.get(o["scenario"], o["scenario"].replace("_", " ")),
                "Your score now": o["before"]["score"],
                "Score after": o["after"]["score"],
                "Sure now": f"{o['before']['top3'][0]['prob']:.0%}" if o["before"]["top3"] else "-",
                "Sure after": f"{o['after']['top3'][0]['prob']:.0%}" if o["after"]["top3"] else "-",
                "What happens": ("✅ Gets better" if o["improved"]
                                 else ("Same, no change" if o["delta_score"] == 0 else "❌ Gets worse")),
            }
            for o in outs
        ]
    )
    st.dataframe(sdf, use_container_width=True, hide_index=True)
    st.caption("Score is a whole number, so tiny wins hide. The 'Sure' columns show "
               "even small changes. If sleep and phone move little for you, the AI is saying "
               "this student needs the bigger help in Step 4. Try healthier answers in Step 1 "
               "and watch sleep move the score.")
else:
    st.warning("No examples to show right now.")

# ------------------------------------------------------------- Step 4 ---
st.subheader("Step 4 · How to change your level + your 7-day plan")
st.caption("First see WHAT to change, then your tips, then your day-by-day plan.")
if reco["flags"]:
    st.warning("⚠️ The app noticed: " + ", ".join(reco["flags"]).replace("_", " "))
improved = [o for o in outs if o["improved"]] if outs else []
same = [o for o in outs if not o["improved"]] if outs else []
st.markdown("### 🔧 To change your level, change these")
QUESTION_OF = {col: q for col, q, _h, _k in QUESTIONS}
KIND_OF = {col: k for col, _q, _h, k in QUESTIONS}


def _word(col, value):
    """Show values as words: Yes/No for yes-no questions, whole numbers else."""
    if KIND_OF.get(col) == "yesno":
        return "Yes" if value >= (q25[col] + q75[col]) / 2 else "No"
    return f"{round(value)}"


if improved:
    for n, o in enumerate(improved, 1):
        parts = []
        for col, new_v in o["changes"].items():
            q = QUESTION_OF.get(col, col)
            parts.append(f"{q} (now: {_word(col, twin.state[col])} → do: {_word(col, new_v)})")
        st.write(
            f"**{n}. {SCENARIO_NAMES.get(o['scenario'], o['scenario'].replace('_', ' '))}**"
        )
        for p in parts:
            st.write(f"   - {p}")
        st.caption(f"   Score {o['before']['score']} → {o['after']['score']} ✅")
else:
    st.write("- You are doing well - keep your sleep, water and play habits! 🎉")
if same:
    with st.expander("Changes that help only a little (see why)"):
        for o in same:
            st.write(
                f"- {SCENARIO_NAMES.get(o['scenario'], o['scenario'].replace('_', ' '))}: "
                f"score stays {o['before']['score']} - this habit alone is not enough, "
                f"do the big changes above first."
            )
st.markdown("### ✅ Tips for you")
for s in reco["strategies"]:
    st.write("- " + s)
st.markdown("### 🗓️ Your 7-day plan (do one line each day)")
for i, r in enumerate(reco["weekly_roadmap"], 1):
    st.write(f"**Day {i}:** {r}")
st.info(reco["encouragement"])

st.divider()
st.subheader("All done! What just happened?")
st.write("**What the app asked you:** 7 small questions about sleep, phone, food and feelings.")
st.write("**What the app gave you:** how you seem now, what would help, and what to do next.")
st.caption(reco["disclaimer"])
