"""
Build the modeling dataset WITH engineered features — leak-safe end to end.

This supersedes the bare Phase 2 splits by inserting Phase 4 feature engineering
in the correct place. The ordering is what keeps it honest:

    1. load + clean (drop duplicates)
    2. engineer row-wise features (Hour, Amount_log)   <- no leakage, per-row
    3. SPLIT (stratified)                               <- before anything learned
    4. Is_high_amount flag using TRAIN's 99th pctile    <- threshold from train only
    5. scale continuous cols, fitting on TRAIN only
    6. save train/test parquet + scaler + threshold

Run:  python src/build_dataset.py
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

from data_loader import load_data
from preprocessing import clean_data, split_data
from features import engineer_features, add_high_amount_flag

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TARGET = "Class"

# Continuous columns to scale (Is_high_amount is binary; V1..V28 already PCA-scaled)
SCALE_COLS = ["Amount", "Time", "Amount_log", "Hour"]


def build(save: bool = True):
    # 1. load + clean
    df = clean_data(load_data())

    # 2. row-wise feature engineering (safe before split)
    df = engineer_features(df)        # adds Hour, Amount_log

    # 3. split BEFORE learning anything from the data
    X_train, X_test, y_train, y_test = split_data(df, TARGET)

    # 4. high-amount flag — threshold computed on TRAIN, reused on TEST
    X_train, threshold = add_high_amount_flag(X_train)            # learns threshold
    X_test, _ = add_high_amount_flag(X_test, threshold=threshold) # reuses it
    print(f"High-amount threshold (train 99th pct): ${threshold:,.2f}")

    # 5. scale continuous cols — fit on TRAIN only
    scaler = RobustScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])

    print(f"Final feature count: {X_train.shape[1]} "
          f"(was 30; +3 engineered)")

    if save:
        PROC.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{TARGET: y_train.values}).to_parquet(PROC / "train.parquet")
        X_test.assign(**{TARGET: y_test.values}).to_parquet(PROC / "test.parquet")
        joblib.dump(scaler, PROC / "scaler.joblib")
        (PROC / "feature_meta.json").write_text(json.dumps({
            "high_amount_threshold": float(threshold),
            "scale_cols": SCALE_COLS,
            "n_features": int(X_train.shape[1]),
            "features": list(X_train.columns),
        }, indent=2))
        print(f"Saved feature-rich splits + scaler + meta to {PROC}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    build(save=True)
