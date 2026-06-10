"""
Phase 7 — Evaluation & Threshold Tuning (the payoff phase).

A trained model outputs a PROBABILITY. Turning it into a decision needs a
THRESHOLD. The default 0.5 is almost never the right business choice for fraud.
Here we:

    1. Plot PR and ROC curves (the full trade-off, not one point)
    2. Show how precision/recall move as we slide the threshold
    3. Pick a threshold to hit a RECALL TARGET (e.g. catch 90% of fraud)
    4. Pick a COST-OPTIMAL threshold: assign $ to a missed fraud (FN) vs a
       false alarm (FP), then minimise total expected cost
    5. Confusion matrix at the chosen threshold, in business terms

All evaluation is on the UNTOUCHED test set (~0.17% fraud) so it reflects
production reality.

Cost assumptions (EDIT THESE to your business — they drive the recommendation):
    COST_FN = a missed fraud costs us the transaction + chargeback/ops overhead
    COST_FP = a false alarm costs analyst review time + customer friction
We use illustrative defaults and are explicit that they are assumptions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve,
                             confusion_matrix, average_precision_score,
                             roc_auc_score, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FIG.mkdir(parents=True, exist_ok=True)

# --- Illustrative business cost assumptions (EDIT to your context) ---
COST_FN = 150.0   # $ cost of a fraud we FAIL to catch (loss + overhead)
COST_FP = 5.0     # $ cost of a false alarm (review time + customer friction)


def load_test_and_model(model_path=MODELS / "best_xgboost.joblib"):
    test = pd.read_parquet(PROC / "test.parquet")
    X_test, y_test = test.drop(columns=["Class"]), test["Class"]
    model = joblib.load(model_path)
    y_proba = model.predict_proba(X_test)[:, 1]
    return X_test, y_test, y_proba


def plot_curves(y_test, y_proba):
    """PR curve (lead) + ROC curve (context), side by side."""
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    baseline = y_test.mean()  # a no-skill classifier's precision = prevalence

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(rec, prec, color="#c0392b", label=f"XGBoost (AP={ap:.3f})")
    ax[0].axhline(baseline, ls="--", color="gray",
                  label=f"no-skill ({baseline:.3%})")
    ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall Curve (lead metric)"); ax[0].legend()

    ax[1].plot(fpr, tpr, color="#2980b9", label=f"XGBoost (AUC={auc:.3f})")
    ax[1].plot([0, 1], [0, 1], ls="--", color="gray", label="no-skill")
    ax[1].set_xlabel("False Positive Rate"); ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("ROC Curve (optimistic under imbalance)"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "06_pr_roc_curves.png", dpi=110)
    plt.close()


def threshold_for_recall(y_test, y_proba, target_recall=0.90):
    """Smallest threshold that achieves at least the target recall."""
    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    # precision_recall_curve returns thr of len-1; align by dropping last point
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= target_recall
    if not ok.any():
        return None
    # among thresholds meeting recall, pick the one with best precision
    idx = np.argmax(np.where(ok, prec, -1))
    return {"threshold": float(thr[idx]), "recall": float(rec[idx]),
            "precision": float(prec[idx])}


def cost_optimal_threshold(y_test, y_proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """
    Sweep thresholds, compute total $ cost = FN*cost_fn + FP*cost_fp at each,
    return the minimiser. This converts a model into a business decision.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cost = fn * cost_fn + fp * cost_fp
        recall = tp / (tp + fn) if (tp + fn) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        rows.append((t, fn, fp, cost, recall, precision))
    df = pd.DataFrame(rows, columns=["threshold", "FN", "FP", "cost",
                                     "recall", "precision"])
    best = df.loc[df["cost"].idxmin()]

    # Plot cost vs threshold
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["threshold"], df["cost"], color="#8e44ad")
    ax.axvline(best["threshold"], ls="--", color="red",
               label=f"min cost @ {best['threshold']:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Total cost ($)")
    ax.set_title(f"Expected Cost vs Threshold (FN=${cost_fn:.0f}, FP=${cost_fp:.0f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "07_cost_vs_threshold.png", dpi=110)
    plt.close()
    return best, df


def confusion_at(y_test, y_proba, threshold, tag):
    """Confusion matrix at a chosen threshold, saved + returned in plain terms."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix @ threshold={threshold:.2f} ({tag})")
    plt.tight_layout()
    plt.savefig(FIG / f"08_confusion_{tag}.png", dpi=110)
    plt.close()

    return {"threshold": threshold, "TP": int(tp), "FN": int(fn),
            "FP": int(fp), "TN": int(tn),
            "recall": tp / (tp + fn), "precision": tp / (tp + fp) if (tp+fp) else 0}


if __name__ == "__main__":
    X_test, y_test, y_proba = load_test_and_model()
    print(f"Test set: {len(y_test):,} txns, {int(y_test.sum())} fraud "
          f"({y_test.mean():.3%})\n")

    plot_curves(y_test, y_proba)

    print("=== Default threshold 0.50 ===")
    print(confusion_at(y_test, y_proba, 0.50, "default"))

    print("\n=== Tuned for 90% recall (catch 90% of fraud) ===")
    rec90 = threshold_for_recall(y_test, y_proba, 0.90)
    print(rec90)
    if rec90:
        print(confusion_at(y_test, y_proba, rec90["threshold"], "recall90"))

    print(f"\n=== Cost-optimal threshold (FN=${COST_FN:.0f}, FP=${COST_FP:.0f}) ===")
    best, _ = cost_optimal_threshold(y_test, y_proba)
    print(best.to_dict())
    print(confusion_at(y_test, y_proba, float(best["threshold"]), "costopt"))

    print("\nFigures saved: 06_pr_roc_curves, 07_cost_vs_threshold, 08_confusion_*")
