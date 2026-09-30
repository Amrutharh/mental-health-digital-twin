"""Layer 8 — Persistence & Reports: save/load bundle, CSV history, text report."""
import joblib
import os
import pandas as pd


def save_bundle(path: str, model, scaler, feature_columns, label_names,
                results: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler,
                 "feature_columns": feature_columns,
                 "label_names": label_names,
                 "results": {k: v["f1_macro"] for k, v in (results or {}).items()}},
                path)
    print(f"[persist] bundle saved -> {path}")


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def export_history_csv(twin, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    twin.history_df().to_csv(path, index=False)
    return path


def write_report(path: str, student_id: str, risk: dict, reco: dict,
                 scenarios: list[dict] | None = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [f"Student Digital Twin — Report for {student_id}", "=" * 50, "",
             f"Risk score: {risk['score']}/30  Level: {risk['level']}",
             "Top-3 predicted types:"]
    lines += [f"  - {t['type']}: {t['prob']:.2%}" for t in risk.get("top3", [])]
    lines += ["", "Recommended strategies:"]
    lines += [f"  {i+1}. {s}" for i, s in enumerate(reco.get("strategies", []))]
    lines += ["", "Weekly roadmap:"]
    lines += [f"  - {r}" for r in reco.get("weekly_roadmap", [])]
    lines += ["", f"Encouragement: {reco.get('encouragement', '')}"]
    if scenarios:
        lines += ["", "What-if scenarios (delta = after - before):"]
        for s in scenarios:
            lines += [f"  - {s['scenario']}: {s['before']['score']} -> "
                      f"{s['after']['score']} (delta {s['delta_score']:+d})"]
    lines += ["", reco.get("disclaimer", "")]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
