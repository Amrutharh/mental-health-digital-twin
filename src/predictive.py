"""Layer 2 — Predictive Intelligence.
Trains RF / GB / LogReg / XGBoost (if installed) and picks best by macro F1.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from .config import RANDOM_STATE


def build_models(random_state: int = RANDOM_STATE) -> dict:
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, multi_class="auto", n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            random_state=random_state, n_jobs=-1)
    return models


def train_and_select(X_train, X_test, y_train, y_test, label_names=None):
    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        results[name] = {"f1_macro": round(float(f1), 4), "model": model}
        fitted[name] = model
        print(f"[models] {name:20s} macro-F1 = {f1:.4f}")
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best = results[best_name]["model"]
    print(f"[models] Best = {best_name} "
          f"(F1={results[best_name]['f1_macro']:.4f})"
          + ("" if HAS_XGB else "  (XGBoost not installed)"))
    try:
        print(classification_report(y_test, best.predict(X_test),
                                    target_names=label_names, zero_division=0))
    except Exception:
        pass
    return best_name, best, fitted, results
