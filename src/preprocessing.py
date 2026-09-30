"""Layer 1 — Preprocessing: scaling + train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .config import FEATURES, TARGET_COL, TEST_SIZE, RANDOM_STATE


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select numeric features, one-hot any extra categoricals, encode target."""
    df = df.copy()
    # If the CSV matches the documented 19-feature schema use it;
    # otherwise (e.g. the real 20-col CSV) use ALL non-target columns.
    overlap = [c for c in FEATURES if c in df.columns]
    if len(overlap) >= 5:
        feat_cols = overlap
    else:
        feat_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feat_cols]
    # One-hot non-numeric extras (e.g. Gender='M'/'F' in the real CSV).
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat:
        X = pd.get_dummies(X, columns=cat, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))
    y_raw = df[TARGET_COL].astype(str)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=df.index)
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series,
                    test_size: float = TEST_SIZE,
                    random_state: int = RANDOM_STATE):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_te_s, y_tr, y_te, scaler, list(X.columns)
