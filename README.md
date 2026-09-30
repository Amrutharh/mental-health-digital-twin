# 🎓 Student Depression Prediction using Digital Twin

**🚀 Live Demo:** https://mental-health-digital-twin-gff6akkdz5nn9m8x9vnjyr.streamlit.app/

An **8-layer ML system** that mirrors a student's mental-health state live, predicts
depression type (12 classes), simulates **what-if interventions** without touching
the real student, and outputs risk scores + rule-based guidance.

Built from scratch in Python (scikit-learn + XGBoost + Streamlit).

## Architecture (8 layers)

| # | Layer | File | What it does |
|---|-------|------|--------------|
| 0 | Setup & Config | `src/config.py` | 19 features, 12 depression types, 4 risk levels (0-30) |
| 0b | Load Data | `src/data_loader.py` | Loads real `Mental Health Classification.csv` (`Depression_Type` target); synthetic fallback |
| 1 | Preprocessing | `src/preprocessing.py` | `StandardScaler` + stratified train/test split |
| 2 | Predictive Intelligence | `src/predictive.py` | RF / GB / LogReg / XGBoost → best by macro **F1** |
| 3 | Digital Twin Engine | `src/digital_twin.py` | `StudentDigitalTwin`: live state + full history |
| 4 | Scenario Analysis | `src/scenario.py` | What-if (e.g. sleep → 7h), measure delta, restore |
| 5 | Risk Scoring | `src/risk_scoring.py` | Probabilities → score 0-30, level, top-3, trend |
| 6 | Recommendation Engine | `src/recommendations.py` | Rule-based coping, roadmap, encouragement (paper's "Generative AI Support Layer") |
| 7 | Dashboard | `app.py` + `src/visualization.py` | Streamlit live UI + trajectory charts |
| 8 | Persistence & Reports | `src/persistence.py` | `bundle.joblib`, history CSV, text report |

```
CSV → scale/split → train 4 models → best-F1 → twin.predict()
      → risk_profile() → recommend() → what-if simulate() → dashboard/report
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. (optional) drop your real dataset here:
#    data/Mental Health Classification.csv   # needs Depression_Type column

# 2. train all 4 models, pick best by F1
python train.py

# 3. launch dashboard
streamlit run app.py

# 4. run tests
python -m pytest tests/ -v
```

Without the real CSV the pipeline auto-generates synthetic data with the
identical 19-feature / 12-class schema, so everything runs end-to-end.

## Notebook mirror

`notebooks/Mental_Health_Digital_Twin.ipynb` walks Cells 1-11 through every
layer (config → load → preprocess → train → twin → what-if → risk →
recommendations → charts → report) — ideal for viva / interview walkthrough.

## Features (19) / Labels (12) / Risk (4)

- **Features:** Age, Gender, Sleep_Hours, Appetite_Score, Interest_Loss,
  Fatigue_Level, Concentration_Difficulty, Nervousness_Level, Restlessness,
  Worthlessness_Feeling, Sadness_Level, Irritability, Social_Withdrawal,
  Academic_Pressure, Family_Conflict, Physical_Symptoms_Count,
  Self_Harm_Flag, Suicidal_Thoughts, Past_Depression_History
- **Labels:** No / Mild / Moderate / Moderately Severe / Severe / Persistent /
  Postpartum / Seasonal Affective / Atypical / Bipolar / Situational / Psychotic Depression
- **Risk:** Low 0-9 · Medium 10-15 · High 16-22 · Severe 23-30

## ⚠️ Disclaimer

Supportive guidance only — **not a medical diagnosis**. Consult a qualified professional.
