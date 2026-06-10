"""
Phase 5 — Handling Class Imbalance (the technical heart of the project).

With ~0.17% fraud, a model can score 99.8% accuracy by predicting "never fraud"
and catch zero criminals. We fix the *training signal* so the model actually
learns the minority class. Three strategies, simplest first:

    1. class weights  -> tell the model minority errors cost more (no new data)
    2. SMOTE          -> synthesize new minority examples until balanced
    3. SMOTE + Tomek  -> oversample minority, then clean borderline overlap

⚠️ THE CARDINAL RULE (enforced here): resample TRAIN ONLY.
    Never apply SMOTE to the test set. The test set must stay a realistic
    0.17%-fraud sample so evaluation reflects production. Resampling test data
    is a leakage bug that produces beautiful, meaningless scores.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


def class_weight_for(y: pd.Series) -> dict:
    """
    Compute balanced class weights = n_samples / (n_classes * class_count).
    Pass this to models via class_weight=... (LogReg, RF, SVM). No data is
    created or destroyed — we just reweight the loss. Safest first option.
    """
    counts = y.value_counts()
    n = len(y)
    return {cls: n / (len(counts) * cnt) for cls, cnt in counts.items()}


def scale_pos_weight_for(y: pd.Series) -> float:
    """
    XGBoost's native imbalance knob = (#negative / #positive).
    Often removes the need for SMOTE entirely for gradient-boosted trees.
    """
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return float(neg / pos)


def apply_smote(X_train, y_train, random_state: int = 42, sampling_strategy="auto"):
    """
    SMOTE on TRAIN ONLY. Synthesizes minority points by interpolating between
    real fraud cases and their nearest fraud neighbours.

    sampling_strategy='auto' balances to 50/50. You can pass a float (e.g. 0.1)
    to only bring fraud up to 10% — often better than full balance, since 50/50
    is wildly unlike reality and can cause many false positives.
    """
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    _report("SMOTE", y_train, y_res)
    return X_res, y_res


def apply_smote_tomek(X_train, y_train, random_state: int = 42):
    """
    SMOTE to oversample minority, then Tomek links to remove borderline pairs
    where a fraud and legit point sit right on top of each other. Cleaner
    decision boundary, usually a small precision gain over plain SMOTE.
    """
    st = SMOTETomek(random_state=random_state)
    X_res, y_res = st.fit_resample(X_train, y_train)
    _report("SMOTE+Tomek", y_train, y_res)
    return X_res, y_res


def _report(name: str, y_before: pd.Series, y_after) -> None:
    """Print before/after class counts so the effect is visible, not magic."""
    b = pd.Series(y_before).value_counts().to_dict()
    a = pd.Series(y_after).value_counts().to_dict()
    print(f"[{name}] before: {b}  ->  after: {a}")


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    # Load the leak-proof TRAIN split saved in Phase 2
    proc = Path(__file__).resolve().parents[1] / "data" / "processed"
    train = pd.read_parquet(proc / "train.parquet")
    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]

    print(f"Train fraud rate (untouched): {y_train.mean():.4%}\n")

    print("balanced class weights:", {k: round(v, 1) for k, v in class_weight_for(y_train).items()})
    print("xgboost scale_pos_weight:", round(scale_pos_weight_for(y_train), 1), "\n")

    # Demonstrate both resamplers (TRAIN ONLY)
    apply_smote(X_train, y_train)                       # full 50/50
    apply_smote(X_train, y_train, sampling_strategy=0.1)  # fraud up to 10%
    apply_smote_tomek(X_train, y_train)
    print("\nNote: the TEST set is never resampled — it stays at ~0.17% fraud.")
