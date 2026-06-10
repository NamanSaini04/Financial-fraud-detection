"""
Phase 6 — Model Training & Comparison.

We train four classifiers and compare them on the SAME untouched test set:
    Logistic Regression  (baseline — everything must beat this)
    Random Forest
    XGBoost
    SVM                  (trained on a subsample — see note)

Imbalance handling per model (from Phase 5):
    LogReg / RF / SVM -> class_weight='balanced'  (reweight loss, no fake data)
    XGBoost           -> scale_pos_weight = neg/pos (native knob)

Why class weights over SMOTE here: for a first honest comparison we avoid
synthesizing data, so results reflect the real signal. SMOTE is offered in
imbalance.py and can be swapped in later to see if it helps.

Evaluation philosophy: we DO NOT rank on accuracy (useless at 0.17% fraud).
We report recall, precision, F1, ROC-AUC, and — most importantly — PR-AUC
(average precision), the right summary metric for rare-event detection.

SVM NOTE: SVC scales poorly (~O(n^2)) on 226k rows. We train it on a stratified
subsample for tractability and label it honestly in the results. This is a real
engineering trade-off, stated openly rather than hidden.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
TARGET = "Class"
SEED = 42


def load_splits():
    """Load the leak-proof, pre-scaled splits saved in Phase 2."""
    train = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    return X_train, y_train, X_test, y_test


def build_models(y_train):
    """Instantiate the four models with appropriate imbalance handling."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="aucpr",
            n_jobs=-1, random_state=SEED),
        "SVM (subsample)": SVC(
            class_weight="balanced", probability=True, random_state=SEED),
    }


def _metrics(name, y_test, y_pred, y_proba, fit_secs):
    return {
        "model": name,
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "fit_secs": round(fit_secs, 1),
    }


def train_and_compare(save_best: bool = True) -> pd.DataFrame:
    X_train, y_train, X_test, y_test = load_splits()
    print(f"Train {X_train.shape} | Test {X_test.shape} | "
          f"test fraud: {y_test.sum()} ({y_test.mean():.3%})\n")

    models = build_models(y_train)
    results, fitted = [], {}

    for name, model in models.items():
        # SVM: train on a stratified subsample for tractability
        if name.startswith("SVM"):
            Xs, _, ys, _ = train_test_split(
                X_train, y_train, train_size=30_000,
                stratify=y_train, random_state=SEED)
            print(f"  {name}: training on {len(Xs):,}-row subsample (speed)...")
            Xfit, yfit = Xs, ys
        else:
            Xfit, yfit = X_train, y_train

        t0 = time.time()
        model.fit(Xfit, yfit)
        secs = time.time() - t0

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        m = _metrics(name, y_test, y_pred, y_proba, secs)
        results.append(m)
        fitted[name] = model
        print(f"  {name:<20} recall={m['recall']:.3f}  prec={m['precision']:.3f}  "
              f"PR-AUC={m['pr_auc']:.3f}  ({m['fit_secs']}s)")

    df = (pd.DataFrame(results)
            .sort_values("pr_auc", ascending=False)
            .reset_index(drop=True))

    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "model_comparison.csv", index=False)

    if save_best:
        best_name = df.iloc[0]["model"]
        MODELS_DIR.mkdir(exist_ok=True)
        # filename-safe
        fname = best_name.split()[0].lower()
        joblib.dump(fitted[best_name], MODELS_DIR / f"best_{fname}.joblib")
        print(f"\nSaved best model ({best_name}) -> models/best_{fname}.joblib")

    return df


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.4f}".format)
    table = train_and_compare()
    print("\n=== MODEL COMPARISON (ranked by PR-AUC) ===")
    print(table.to_string(index=False))
    print("\nBaseline to beat: LogisticRegression. Accuracy intentionally omitted "
          "— it is ~99.8% for everything and tells us nothing here.")
