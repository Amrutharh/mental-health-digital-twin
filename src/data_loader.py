"""Layer 0b — Data loading.

Loads the REAL csv `Mental Health Classification.csv` (target col `Depression_Type`).
If the file is absent (e.g. fresh clone without the dataset), falls back to a
realistic synthetic generator with the same 19-feature / 12-class schema so the
whole pipeline still runs end-to-end. Place your real file at `data/`.
"""
import os
import numpy as np
import pandas as pd

from .config import FEATURES, DEPRESSION_TYPES, TARGET_COL, DATA_CANDIDATES


def find_real_csv(explicit: str | None = None) -> str | None:
    if explicit and os.path.exists(explicit):
        return explicit
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def load_data(csv_path: str | None = None, n_synthetic: int = 1500,
              random_state: int = 42) -> tuple[pd.DataFrame, bool]:
    """Returns (df, used_real_data). df always has FEATURES + TARGET_COL."""
    found = find_real_csv(csv_path)
    if found:
        df = pd.read_csv(found)
        if TARGET_COL not in df.columns:
            raise ValueError(
                f"'{TARGET_COL}' column not found in {found}. "
                f"Columns: {list(df.columns)}")
        print(f"[data] Loaded REAL dataset: {found} shape={df.shape}")
        return df, True
    print("[data] Real CSV not found — generating synthetic fallback "
          f"(n={n_synthetic}). Drop your file at data/'Mental Health Classification.csv'.")
    return generate_synthetic(n=n_synthetic, seed=random_state), False


def generate_synthetic(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = {}
    d["Age"] = rng.integers(15, 30, n)
    d["Gender"] = rng.integers(0, 3, n)
    d["Sleep_Hours"] = np.clip(rng.normal(6.5, 1.6, n), 2, 11).round(1)
    d["Appetite_Score"] = rng.integers(1, 6, n)
    for col in ["Interest_Loss", "Fatigue_Level", "Concentration_Difficulty",
                "Nervousness_Level", "Restlessness", "Worthlessness_Feeling",
                "Sadness_Level", "Irritability", "Social_Withdrawal",
                "Academic_Pressure", "Family_Conflict"]:
        d[col] = rng.integers(0, 11, n)
    d["Physical_Symptoms_Count"] = rng.integers(0, 9, n)
    # Correlated distress -> severity signal
    distress = (d["Sadness_Level"] + d["Interest_Loss"] + d["Fatigue_Level"]
                + d["Worthlessness_Feeling"] + d["Nervousness_Level"]) / 5.0
    poor_sleep = np.clip(7.5 - d["Sleep_Hours"], 0, 5)
    p_self = 1 / (1 + np.exp(-(distress - 5.5) * 1.1))
    p_suic = 1 / (1 + np.exp(-(distress - 6.8) * 1.2))
    d["Self_Harm_Flag"] = (rng.random(n) < p_self * 0.5).astype(int)
    d["Suicidal_Thoughts"] = (rng.random(n) < p_suic * 0.4).astype(int)
    d["Past_Depression_History"] = (rng.random(n) < 0.25).astype(int)

    score = distress + poor_sleep * 0.8 + d["Self_Harm_Flag"] * 2.0 \
        + d["Suicidal_Thoughts"] * 2.5 + d["Past_Depression_History"] * 0.7
    labels = []
    for s in score:
        if s < 3.0:
            labels.append("No Depression")
        elif s < 4.5:
            labels.append(rng.choice(["Mild Depression", "Situational Depression"]))
        elif s < 6.0:
            labels.append(rng.choice(["Moderate Depression", "Seasonal Affective Disorder",
                                       "Atypical Depression"]))
        elif s < 7.5:
            labels.append(rng.choice(["Moderately Severe Depression",
                                       "Persistent Depressive Disorder",
                                       "Postpartum Depression"]))
        elif s < 9.0:
            labels.append(rng.choice(["Severe Depression", "Bipolar Depression"]))
        else:
            labels.append(rng.choice(["Severe Depression", "Psychotic Depression"]))
    d[TARGET_COL] = labels
    # Safety: ensure all 12 classes appear at least a few times
    for i, t in enumerate(DEPRESSION_TYPES):
        if t not in labels:
            d[TARGET_COL][i % n] = t
    return pd.DataFrame(d)[FEATURES + [TARGET_COL]]
