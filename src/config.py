"""Layer 0 — Setup & Config.
Central constants: 19 input features, 12 depression-type labels, 4 risk levels.
Single source of truth imported by every other layer.
"""

FEATURES = [
    "Age",
    "Gender",                    # 0=Female, 1=Male, 2=Other
    "Sleep_Hours",               # hours per night
    "Appetite_Score",            # 1-5 (1=very poor, 5=healthy)
    "Interest_Loss",             # 0-10
    "Fatigue_Level",             # 0-10
    "Concentration_Difficulty",  # 0-10
    "Nervousness_Level",         # 0-10
    "Restlessness",              # 0-10
    "Worthlessness_Feeling",     # 0-10
    "Sadness_Level",             # 0-10
    "Irritability",              # 0-10
    "Social_Withdrawal",         # 0-10
    "Academic_Pressure",         # 0-10
    "Family_Conflict",           # 0-10
    "Physical_Symptoms_Count",   # 0-8
    "Self_Harm_Flag",            # 0/1
    "Suicidal_Thoughts",         # 0/1
    "Past_Depression_History",   # 0/1
]

# 12-class target — must match `Depression_Type` values in the real CSV.
DEPRESSION_TYPES = [
    "No Depression",
    "Mild Depression",
    "Moderate Depression",
    "Moderately Severe Depression",
    "Severe Depression",
    "Persistent Depressive Disorder",
    "Postpartum Depression",
    "Seasonal Affective Disorder",
    "Atypical Depression",
    "Bipolar Depression",
    "Situational Depression",
    "Psychotic Depression",
]

TYPE_TO_ID = {t: i for i, t in enumerate(DEPRESSION_TYPES)}
ID_TO_TYPE = {i: t for t, i in TYPE_TO_ID.items()}

# Risk levels mapped to a 0-30 score.
RISK_LEVELS = ["Low", "Medium", "High", "Severe"]
RISK_THRESHOLDS = [(9, "Low"), (15, "Medium"), (22, "High"), (30, "Severe")]

TARGET_COL = "Depression_Type"
RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_CANDIDATES = [
    "data/Mental Health Classification.csv",
    "data/mental_health_classification.csv",
    "../data/Mental Health Classification.csv",
    "Mental Health Classification.csv",
]


def risk_level_from_score(score: int) -> str:
    """Map 0-30 score -> Low/Medium/High/Severe."""
    for thresh, level in RISK_THRESHOLDS:
        if score <= thresh:
            return level
    return "Severe"
