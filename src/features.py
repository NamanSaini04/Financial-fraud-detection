"""
Phase 4 — Feature Engineering.

Principle: add a feature only if it MIGHT add signal, then check whether it
actually does. We don't add features for decoration — every extra column is
more noise, more overfitting risk, and more to explain. We engineer, then we
audit (see evaluate_features below).

For ULB we are limited (V1..V28 are opaque PCA), so we work with the two raw
columns we *can* reason about — Time and Amount:
    - Hour          : fraud rate spikes at 2-4am (found in Phase 3)
    - Amount_log    : Amount is heavily right-skewed; log tames it
    - Is_high_amount: simple rule-based flag for >99th percentile amounts

NOTE: build features on RAW Amount/Time, BEFORE scaling. In the real pipeline,
engineer first, then split, then scale (so the scaler also scales Amount_log).
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable, leakage-safe features derived from Time and Amount."""
    df = df.copy()

    # Hour of day (0-23) from seconds-since-first-transaction
    df["Hour"] = (df["Time"] // 3600) % 24

    # Log-transform Amount to compress the long right tail. log1p handles 0.
    df["Amount_log"] = np.log1p(df["Amount"])

    # Rule-based high-amount flag. Threshold from the data's 99th percentile.
    # IMPORTANT: in the full pipeline compute this threshold on TRAIN only to
    # avoid leakage. Exposed as a param so callers can pass the train threshold.
    return df


def add_high_amount_flag(df: pd.DataFrame, threshold: float | None = None) -> tuple[pd.DataFrame, float]:
    """
    Add Is_high_amount using a threshold. If none given, use the 99th percentile
    of THIS dataframe — callers should pass the TRAIN threshold for test data.
    Returns (df, threshold) so the threshold can be reused on the test set.
    """
    df = df.copy()
    if threshold is None:
        threshold = df["Amount"].quantile(0.99)
    df["Is_high_amount"] = (df["Amount"] > threshold).astype(int)
    return df, threshold


def evaluate_features(df: pd.DataFrame, features: list[str], target: str = "Class") -> pd.DataFrame:
    """
    Audit whether engineered features actually separate the classes.

    We report, per feature:
      - |corr| with target (linear signal)
      - mean value for legit vs fraud (does it differ?)
    A feature that looks identical across classes is not earning its place.
    """
    rows = []
    for f in features:
        corr = df[f].corr(df[target])
        legit_mean = df.loc[df[target] == 0, f].mean()
        fraud_mean = df.loc[df[target] == 1, f].mean()
        rows.append({
            "feature": f,
            "abs_corr": abs(corr),
            "legit_mean": legit_mean,
            "fraud_mean": fraud_mean,
            "verdict": "keep" if abs(corr) >= 0.01 else "weak signal",
        })
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data

    df = clean_data(load_data())
    df = engineer_features(df)
    df, thr = add_high_amount_flag(df)
    print(f"High-amount threshold (99th pct): ${thr:,.2f}\n")

    new_feats = ["Hour", "Amount_log", "Is_high_amount"]
    print("Do the engineered features separate the classes?")
    print(evaluate_features(df, new_feats).to_string(index=False))
