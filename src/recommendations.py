"""Layer 6 — Recommendation Engine (rule-based, not LLM).
Picks coping strategies, a weekly roadmap and encouragement from risk level +
specific feature values. Called 'Generative AI Support Layer' in the paper.
"""
from .config import RISK_LEVELS  # noqa: F401  (documents the 4 levels)


def recommend(risk: dict, state: dict) -> dict:
    level, score = risk["level"], risk["score"]
    strategies, roadmap, flags = [], [], []

    # --- universal by level ------------------------------------------------
    if level == "Low":
        strategies += ["Maintain a consistent sleep schedule (7-8h).",
                       "Keep a brief daily mood journal.",
                       "Stay physically active 30 min/day."]
        roadmap = ["Mon: 30-min walk", "Wed: journal + friend catch-up",
                   "Fri: review week", "Sun: plan next week"]
        encouragement = "You're doing well — small steady habits keep you resilient."
    elif level == "Medium":
        strategies += ["Practice 10-min breathing / mindfulness daily.",
                       "Limit late-night screens; protect sleep.",
                       "Talk to a trusted friend or mentor this week."]
        roadmap = ["Daily: 10-min mindfulness", "Tue: counselling-centre intro visit",
                   "Thu: exercise session", "Sat: social activity"]
        encouragement = "You've noticed early signs — acting now makes a big difference."
    elif level == "High":
        strategies += ["Book a counsellor appointment within 48 hours.",
                       "Share how you feel with someone you trust today.",
                       "Use grounding (5-4-3-2-1) when overwhelmed."]
        roadmap = ["Day 1-2: counsellor booking + trusted-person chat",
                   "Day 3-4: sleep reset (no screens after 10pm)",
                   "Day 5-7: daily check-ins + light exercise"]
        encouragement = "Things feel heavy right now, but support helps — reach out today."
    else:  # Severe
        strategies += ["URGENT: contact campus counselling / crisis helpline now.",
                       "Do not stay alone — reach a trusted person immediately.",
                       "If you may act on self-harm thoughts, call emergency services."]
        roadmap = ["NOW: helpline + trusted person",
                   "Today: professional assessment",
                   "This week: daily supervised check-ins"]
        encouragement = "Your safety matters most. Please reach out for help right now."

    # --- feature-specific flags -------------------------------------------
    if state.get("Sleep_Hours", 7) < 6:
        flags.append("low_sleep")
        strategies.append(f"Sleep is low ({state['Sleep_Hours']}h) — "
                          "target 7-8h, fixed wake time.")
    if state.get("Self_Harm_Flag", 0) == 1:
        flags.append("self_harm")
        strategies.append("Self-harm flagged — please involve a professional immediately.")
    if state.get("Suicidal_Thoughts", 0) == 1:
        flags.append("suicidal_thoughts")
        strategies.append("Suicidal thoughts flagged — crisis support now (not later).")
    if state.get("Social_Withdrawal", 0) >= 7:
        flags.append("withdrawal")
        strategies.append("High withdrawal — schedule one small social contact this week.")
    if state.get("Academic_Pressure", 0) >= 7:
        flags.append("academic_pressure")
        strategies.append("High academic pressure — break work into 25-min blocks + ask for help.")

    return {"risk_level": level, "risk_score": score, "flags": flags,
            "strategies": strategies, "weekly_roadmap": roadmap,
            "encouragement": encouragement,
            "disclaimer": "Supportive guidance only — not a medical diagnosis. "
                          "Consult a qualified professional."}
