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


def compare_scenarios(twin, scenarios: dict) -> list[dict]:
    """scenarios: {name: {feature: value}}. Returns sorted (best first)."""
    outs = [simulate(twin, name, **chg) for name, chg in scenarios.items()]
    return sorted(outs, key=lambda r: r["delta_score"])


DEFAULT_SCENARIOS = {
    "better_sleep_7h": {"Sleep_Hours": 7.5},
    "lower_academic_pressure": {"Academic_Pressure": 2},
    "reconnect_socially": {"Social_Withdrawal": 2},
    "calm_nervousness": {"Nervousness_Level": 2, "Restlessness": 2},
}
