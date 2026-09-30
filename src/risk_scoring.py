"""Layer 5 — Risk Scoring: probabilities -> 0-30 score, level, top-3, trend."""
import numpy as np
from .config import risk_level_from_score

SEVERE_FLAGS = {"Severe Depression", "Psychotic Depression", "Bipolar Depression"}


def risk_profile(proba, state: dict, label_names=None) -> dict:
    """Convert model probabilities + symptom state into a readable profile."""
    proba = np.array(proba, dtype=float) if proba is not None else np.array([1.0])
    names = label_names or [f"class_{i}" for i in range(len(proba))]
    p_no = float(proba[0]) if len(proba) > 0 else 0.0  # class 0 ~ No Depression
    # Base: how far from 'No Depression', scaled to ~0-20
    base = (1.0 - p_no) * 20.0
    # Symptom load 0-10 -> adds 0-5
    symptom_keys = ["Sadness_Level", "Interest_Loss", "Fatigue_Level",
                    "Worthlessness_Feeling", "Nervousness_Level"]
    sym = float(np.mean([state.get(k, 0) for k in symptom_keys])) if state else 0
    score = base + sym * 0.5
    # Red flags push the score up hard
    if state:
        if state.get("Suicidal_Thoughts", 0) == 1:
            score += 5
        if state.get("Self_Harm_Flag", 0) == 1:
            score += 3
        if state.get("Sleep_Hours", 7) < 4:
            score += 2
    # Severe-type probability mass adds up to +3
    for i, n in enumerate(names):
        if n in SEVERE_FLAGS and i < len(proba):
            score += float(proba[i]) * 3.0
    score = int(np.clip(round(score), 0, 30))
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [{"type": names[i], "prob": round(float(proba[i]), 4)}
            for i in top3_idx if i < len(names)]
    return {"score": score, "level": risk_level_from_score(score),
            "top3": top3, "p_no_depression": round(p_no, 4)}


def risk_trend(scores: list[int]) -> str:
    """worsening / improving / stable from a score series."""
    if len(scores) < 2:
        return "stable"
    d = scores[-1] - scores[0]
    if d >= 3:
        return "worsening"
    if d <= -3:
        return "improving"
    return "stable"
