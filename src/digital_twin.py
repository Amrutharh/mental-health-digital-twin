"""Layer 3 — Digital Twin Engine.
A StudentDigitalTwin holds LIVE feature values, predicts through the trained
model, and appends a history row on every change (the 'live mirror').
"""
from copy import deepcopy
from datetime import datetime
import pandas as pd

from .config import FEATURES, ID_TO_TYPE


class StudentDigitalTwin:
    def __init__(self, student_id: str, features: dict,
                 model=None, scaler=None, feature_columns=None,
                 label_names=None):
        missing = [f for f in FEATURES if f not in features]
        if missing:
            raise ValueError(f"Missing features for twin: {missing}")
        self.student_id = student_id
        self.state = {f: features[f] for f in FEATURES}
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns or list(FEATURES)
        self.label_names = label_names
        self.history: list[dict] = []
        self.snapshot(event="init")

    # -- internals ---------------------------------------------------------
    def _vector(self):
        row = pd.DataFrame([{c: self.state.get(c, 0)
                             for c in self.feature_columns}])
        if self.scaler is not None:
            return self.scaler.transform(row)
        return row.values

    # -- public API --------------------------------------------------------
    def predict(self) -> dict:
        if self.model is None:
            raise RuntimeError("Twin has no trained model attached.")
        X = self._vector()
        pred_id = int(self.model.predict(X)[0])
        proba = (self.model.predict_proba(X)[0]
                 if hasattr(self.model, "predict_proba") else None)
        names = self.label_names or [ID_TO_TYPE.get(i, str(i))
                                     for i in range(len(proba) if proba is not None else 0)]
        return {"label_id": pred_id,
                "label": names[pred_id] if pred_id < len(names) else str(pred_id),
                "proba": proba}

    def snapshot(self, event: str = "update") -> dict:
        out = self.predict() if self.model is not None else {}
        row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "student_id": self.student_id, "event": event,
               **deepcopy(self.state),
               "pred_label": out.get("label"),
               "pred_id": out.get("label_id")}
        self.history.append(row)
        return row

    def update(self, event: str = "update", **changes) -> dict:
        for k, v in changes.items():
            if k not in FEATURES:
                raise KeyError(f"Unknown feature '{k}'. Expected one of {FEATURES}")
            self.state[k] = v
        return self.snapshot(event=event)

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def __repr__(self):
        last = self.history[-1] if self.history else {}
        return (f"StudentDigitalTwin(id={self.student_id}, "
                f"pred={last.get('pred_label')})")
