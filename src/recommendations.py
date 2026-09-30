"""Layer 6 — Recommendation Engine (rule-based, not LLM).
Picks coping strategies, a weekly roadmap and encouragement from risk level +
specific feature values. Called 'Generative AI Support Layer' in the paper.
"""
from .config import RISK_LEVELS  # noqa: F401  (documents the 4 levels)


def recommend(risk: dict, state: dict) -> dict:
    level, score = risk["level"], risk["score"]
    strategies, roadmap, flags = [], [], []

    # --- universal by level ------------------------------------------------
    # Simple words + concrete daily habits (drink water, sleep, play...).
    if level == "Low":
        strategies += ["Sleep 8 hours every night at the same time.",
                       "Drink 6-8 glasses of water every day.",
                       "Play outside for 30 minutes every day."]
        roadmap = ["Mon: drink 6 glasses of water + play outside 30 min",
                   "Tue: sleep 8 hours + eat breakfast",
                   "Wed: play with a friend + drink water",
                   "Thu: sleep 8 hours + no phone while eating",
                   "Fri: play outside 30 min + tell family one happy thing",
                   "Sat: drink water + help at home + early bed",
                   "Sun: rest, drink water, get ready for school"]
        encouragement = "You are doing well! Keep these small habits every day."
    elif level == "Medium":
        strategies += ["Sleep 8 hours every night - no phone after 9pm.",
                       "Drink water every time you feel tired.",
                       "Tell one person you trust how you feel."]
        roadmap = ["Mon: sleep 8 hours + drink 6 glasses of water",
                   "Tue: tell a parent or teacher how you feel",
                   "Wed: play outside 30 min + no phone while eating",
                   "Thu: sleep 8 hours + drink water + deep breaths 5 times",
                   "Fri: play with a friend + eat dinner with family",
                   "Sat: morning walk + drink water + early bed",
                   "Sun: rest + tell family one good thing of the week"]
        encouragement = "You noticed early signs - doing these small things now helps a lot."
    elif level == "High":
        strategies += ["Tell a parent or teacher TODAY how you feel.",
                       "Sleep 8 hours - give your phone to a parent at 9pm.",
                       "Drink water and eat 3 meals every day."]
        roadmap = ["Mon: tell a parent TODAY + sleep 8 hours",
                   "Tue: meet the school counsellor + drink 6 glasses of water",
                   "Wed: no phone while eating + play outside with someone",
                   "Thu: sleep 8 hours + eat breakfast, lunch, dinner",
                   "Fri: tell the counsellor how the week went + drink water",
                   "Sat: family time + morning walk + early bed",
                   "Sun: rest + parent checks how you feel"]
        encouragement = "Things feel heavy now, but people want to help you - talk to them today."
    else:  # Severe
        strategies += ["URGENT: tell a parent, teacher or counsellor NOW.",
                       "Do not stay alone - sit with someone you trust.",
                       "If you want to hurt yourself, call emergency services now."]
        roadmap = ["Mon: tell an adult NOW + stay with family today",
                   "Tue: visit the doctor or counsellor + drink water + eat meals",
                   "Wed: stay with family + sleep 8 hours + no phone alone",
                   "Thu: counsellor visit + eat 3 meals + drink water",
                   "Fri: family checks on you many times + early bed",
                   "Sat: stay with loved ones + short walk with someone",
                   "Sun: rest with family + plan next week with the counsellor"]
        encouragement = "You matter the most. Please stay with someone and get help right now."

    # --- feature-specific flags (handles both synthetic + real schemas) -----
    def _get(*names, default=0):
        for n in names:
            if n in state:
                return state[n]
        return default
    if _get("Sleep_Hours", default=7) < 6:
        flags.append("low_sleep")
        strategies.append(f"Sleep is low ({_get('Sleep_Hours', default=7)}h) — "
                          "target 7-8h, fixed wake time.")
    if _get("Self_Harm_Flag", "Self_Harm", default=0) == 1:
        flags.append("self_harm")
        strategies.append("Self-harm flagged — please involve a professional immediately.")
    if _get("Suicidal_Thoughts", "Suicide_Attempts", default=0) >= 1:
        flags.append("suicidal_thoughts")
        strategies.append("Suicidal thoughts flagged — crisis support now (not later).")
    if _get("Social_Withdrawal", default=0) >= 7:
        flags.append("withdrawal")
        strategies.append("High withdrawal — schedule one small social contact this week.")
    if _get("Academic_Pressure", default=0) >= 7:
        flags.append("academic_pressure")
        strategies.append("High academic pressure — break work into 25-min blocks + ask for help.")
    if _get("Nervous_Level", "Nervousness_Level", default=0) >= 7:
        flags.append("high_nervousness")
        strategies.append("High nervousness — try daily breathing/mindfulness + cut caffeine.")
    if _get("SocialMedia_Hours", default=0) >= 7:
        flags.append("heavy_social_media")
        strategies.append("Heavy social-media use — set a 2h/day limit + no phone after 10pm.")

    return {"risk_level": level, "risk_score": score, "flags": flags,
            "strategies": strategies, "weekly_roadmap": roadmap,
            "encouragement": encouragement,
            "disclaimer": "Supportive guidance only — not a medical diagnosis. "
                          "Consult a qualified professional."}
