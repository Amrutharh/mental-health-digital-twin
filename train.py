"""End-to-end training script: python train.py [--csv PATH] [--n 1500]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (load_data, prepare_features, split_and_scale,
                 train_and_select, save_bundle)
from sklearn.preprocessing import LabelEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="models/bundle.joblib")
    args = ap.parse_args()

    df, used_real = load_data(args.csv, n_synthetic=args.n)
    X, y_ids = prepare_features(df)
    # Recover label names from encoder order: prepare_features encodes sorted labels;
    # rebuild names so indices line up.
    le = LabelEncoder().fit(df["Depression_Type"].astype(str))
    label_names = list(le.classes_)
    Xtr, Xte, ytr, yte, scaler, cols = split_and_scale(X, y_ids)
    best_name, best, fitted, results = train_and_select(
        Xtr, Xte, ytr, yte, label_names=label_names)
    save_bundle(args.out, best, scaler, cols, label_names, results)
    print(f"[done] best={best_name} real_data={used_real} -> {args.out}")


if __name__ == "__main__":
    main()
