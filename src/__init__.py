"""Convenience re-exports so `from src import X` works everywhere."""
from .config import FEATURES, DEPRESSION_TYPES, RISK_LEVELS, TARGET_COL
from .data_loader import load_data, generate_synthetic
from .preprocessing import prepare_features, split_and_scale
from .predictive import train_and_select
from .digital_twin import StudentDigitalTwin
from .scenario import simulate, compare_scenarios, DEFAULT_SCENARIOS
from .risk_scoring import risk_profile, risk_trend
from .recommendations import recommend
from .persistence import save_bundle, load_bundle, export_history_csv, write_report

__all__ = ["FEATURES", "DEPRESSION_TYPES", "RISK_LEVELS", "TARGET_COL",
           "load_data", "generate_synthetic", "prepare_features",
           "split_and_scale", "train_and_select", "StudentDigitalTwin",
           "simulate", "compare_scenarios", "DEFAULT_SCENARIOS",
           "risk_profile", "risk_trend", "recommend",
           "save_bundle", "load_bundle", "export_history_csv", "write_report"]
