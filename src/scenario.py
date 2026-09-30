"""Layer 4 — Scenario Analysis ('what-if' interventions).
Temporarily mutate the twin, re-run the model, measure risk delta, restore.
"""
from .risk_scoring import risk_profile


def simulate(twin, event_name: str, restore: bool = True, **changes) -> dict:
    """Run one what-if scenario on a twin. Original state is restored."""
    original = dict(twin.state)
    before = risk_profile(twin.predict()["proba"], twin.state,
                          label_names=twin.label_names)
    twin.update(event=f"scenario:{event_name}", **changes)
    after = risk_profile(twin.predict()["proba"], twin.state,
                         label_names=twin.label_names)
    if restore:
        twin.state = original
        twin.snapshot(event=f"restore:{event_name}")
    return {"scenario": event_name, "changes": changes,
            "before": before, "after": after,
            "delta_score": after["score"] - before["score"],
            "improved": after["score"] < before["score"]}


def compare_scenarios(twin, scenarios: dict | None = None) -> list[dict]:
    """scenarios: {name: {feature: value}}. Returns sorted (best first).
    Auto-filters to features the twin actually tracks (real vs synthetic schema)."""
    if scenarios is None:
        scenarios = default_scenarios_for(twin)
    valid = {n: {k: v for k, v in chg.items() if k in twin.state}
             for n, chg in scenarios.items()}
    valid = {n: c for n, c in valid.items() if c}
    outs = [simulate(twin, name, **chg) for name, chg in valid.items()]
    return sorted(outs, key=lambda r: r["delta_score"])


def default_scenarios_for(twin) -> dict:
    """Pick what-if scenarios matching the twin's real feature set."""
    s = set(twin.state.keys())
    if "Academic_Pressure" in s:  # synthetic 19-feature schema
        return dict(DEFAULT_SCENARIOS)
    # Real CSV schema (Sleep_Hours, Nervous_Level, SocialMedia_Hours, ...)
    cand = {
        "better_sleep": {"Sleep_Hours": 8},
        "calm_nervousness": {"Nervous_Level": 1},
        "less_social_media": {"SocialMedia_Hours": 2},
        "stop_self_harm": {"Self_Harm": 0, "Suicide_Attempts": 0},
    }
    return {n: c for n, c in cand.items()
            if all(k in s for k in c)}


DEFAULT_SCENARIOS = {
    "better_sleep_7h": {"Sleep_Hours": 7.5},
    "lower_academic_pressure": {"Academic_Pressure": 2},
    "reconnect_socially": {"Social_Withdrawal": 2},
    "calm_nervousness": {"Nervousness_Level": 2, "Restlessness": 2},
}
