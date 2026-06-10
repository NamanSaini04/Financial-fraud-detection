"""
Phase 9 - Streamlit fraud-detection dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Four tabs:
    1. Overview   - KPI cards (volume, fraud rate, model recall/precision)
    2. Explore    - the saved EDA figures
    3. Threshold  - LIVE slider: drag it, watch recall/precision/cost update
    4. Score      - sample a real transaction and see its risk score + band

The dashboard reuses the exact artifacts from the pipeline (the saved XGBoost
model, the leak-safe test split, the fitted scaler), so what you see here matches
the analysis - nothing is recomputed differently.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score

# --- paths (relative to project root so it runs from anywhere) ---
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
MODEL_PATH = ROOT / "models" / "best_xgboost.joblib"

# --- business cost assumptions (mirror src/evaluate.py) ---
COST_FN = 150.0   # $ cost of a missed fraud
COST_FP = 5.0     # $ cost of a false alarm
RECOMMENDED_THRESHOLD = 0.12

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")


# ----------------------------------------------------------------------------
# Cached loaders - run once, not on every slider move
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test():
    test = pd.read_parquet(PROC / "test.parquet")
    return test.drop(columns=["Class"]), test["Class"]


@st.cache_data
def get_scores():
    """Predicted fraud probabilities on the test set (computed once)."""
    model = load_model()
    X_test, y_test = load_test()
    return model.predict_proba(X_test)[:, 1], y_test.values


def metrics_at(threshold, y_proba, y_true):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    cost = fn * COST_FN + fp * COST_FP
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, recall=recall, precision=precision, cost=cost)


# ----------------------------------------------------------------------------
# Guard: artifacts must exist
# ----------------------------------------------------------------------------
if not MODEL_PATH.exists() or not (PROC / "test.parquet").exists():
    st.error(
        "Missing artifacts. Run the pipeline first:\n\n"
        "`python src/preprocessing.py && python src/models.py`"
    )
    st.stop()

y_proba, y_true = get_scores()
n_total = 283_726  # full de-duplicated dataset size (from the analysis)
fraud_rate = 0.0017

st.title("💳 Financial Fraud Detection Dashboard")
st.caption(
    "XGBoost on the ULB credit-card dataset (0.17% fraud). "
    "Metrics shown on the held-out test set. Cost model: "
    f"missed fraud = ${COST_FN:.0f}, false alarm = ${COST_FP:.0f}."
)

tab_overview, tab_explore, tab_threshold, tab_score = st.tabs(
    ["📊 Overview", "🔍 Explore", "🎚️ Threshold simulator", "🧾 Score a transaction"]
)

# ----------------------------------------------------------------------------
# Tab 1 - Overview
# ----------------------------------------------------------------------------
with tab_overview:
    rec = metrics_at(RECOMMENDED_THRESHOLD, y_proba, y_true)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analysed", f"{n_total:,}")
    c2.metric("Fraud rate", f"{fraud_rate:.2%}")
    c3.metric("Recall @ 0.12", f"{rec['recall']:.0%}", help="Share of fraud caught")
    c4.metric("Precision @ 0.12", f"{rec['precision']:.0%}", help="Share of flags that are real")

    st.divider()
    st.subheader("Model quality (threshold-independent)")
    d1, d2 = st.columns(2)
    d1.metric("PR-AUC", f"{average_precision_score(y_true, y_proba):.3f}",
              help="Lead metric for rare-event detection")
    d2.metric("ROC-AUC", f"{roc_auc_score(y_true, y_proba):.3f}",
              help="Optimistic under heavy imbalance - see report")

    st.info(
        "**Recommendation:** operate at the **0.12** threshold. It catches "
        f"~{rec['recall']:.0%} of fraud while flagging only {rec['tp'] + rec['fp']} "
        f"of {len(y_true):,} test transactions for review - cutting estimated cost "
        "~20% vs the naive 0.50 default. Explore the trade-off in the "
        "**Threshold simulator** tab."
    )

# ----------------------------------------------------------------------------
# Tab 2 - Explore (saved EDA figures)
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Exploratory analysis")
    figs = [
        ("01_class_balance.png", "Class balance - fraud is 0.17% of transactions."),
        ("03_fraud_rate_by_hour.png", "Fraud rate peaks at 2-4 AM (~8x baseline)."),
        ("02_amount_by_class.png", "Fraud skews to small amounts (median $9 vs $22)."),
        ("04_top_feature_boxplots.png", "A few PCA features separate the classes cleanly."),
    ]
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, (fname, cap) in zip(cols, figs[i:i + 2]):
            path = FIG / fname
            if path.exists():
                col.image(str(path), caption=cap, use_container_width=True)
            else:
                col.warning(f"Missing figure: {fname} (run `python src/eda.py`)")

# ----------------------------------------------------------------------------
# Tab 3 - Threshold simulator (the centerpiece)
# ----------------------------------------------------------------------------
with tab_threshold:
    st.subheader("Drag the threshold - watch the trade-off move")
    threshold = st.slider("Decision threshold", 0.01, 0.99, RECOMMENDED_THRESHOLD, 0.01)
    m = metrics_at(threshold, y_proba, y_true)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recall (fraud caught)", f"{m['recall']:.0%}")
    k2.metric("Precision (flags real)", f"{m['precision']:.0%}")
    k3.metric("Missed fraud (FN)", f"{m['fn']}")
    k4.metric("False alarms (FP)", f"{m['fp']}")

    base = metrics_at(0.50, y_proba, y_true)["cost"]
    st.metric("Estimated cost at this threshold", f"${m['cost']:,.0f}",
              delta=f"{m['cost'] - base:,.0f} vs default 0.50",
              delta_color="inverse")

    st.write("**Confusion matrix at this threshold**")
    cm = pd.DataFrame(
        [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        index=["Actual Legit", "Actual Fraud"],
        columns=["Pred Legit", "Pred Fraud"],
    )
    st.table(cm)

    if threshold < 0.05:
        st.warning("Very low threshold: you catch more fraud but false alarms explode "
                   "- analysts get buried and cost rises.")
    elif threshold > 0.6:
        st.warning("High threshold: few false alarms, but you miss more fraud.")
    else:
        st.success("Balanced zone. The cost-minimising threshold for these "
                   "assumptions is ~0.12.")

# ----------------------------------------------------------------------------
# Tab 4 - Score a transaction
# ----------------------------------------------------------------------------
with tab_score:
    st.subheader("Score a real transaction")
    st.caption("Features are PCA-anonymised, so we sample an actual test transaction "
               "rather than typing 28 opaque values by hand.")

    X_test, y_test = load_test()
    colA, colB = st.columns(2)
    pick_fraud = colB.checkbox("Sample only known-fraud cases", value=False)

    if colA.button("🎲 Pick a random transaction", type="primary") or "row_idx" not in st.session_state:
        pool = y_test[y_test == 1].index if pick_fraud else y_test.index
        st.session_state.row_idx = int(np.random.choice(pool))

    idx = st.session_state.row_idx
    pos = X_test.index.get_loc(idx)
    proba = float(y_proba[pos])
    actual = int(y_test.loc[idx])

    band = ("🟢 LOW", "green") if proba < RECOMMENDED_THRESHOLD else \
           ("🟠 MEDIUM", "orange") if proba < 0.5 else ("🔴 HIGH", "red")

    s1, s2, s3 = st.columns(3)
    s1.metric("Fraud probability", f"{proba:.1%}")
    s2.metric("Risk band", band[0])
    s3.metric("Actual label", "FRAUD" if actual else "Legit")

    decision = "FLAG for review" if proba >= RECOMMENDED_THRESHOLD else "Allow"
    st.markdown(f"**Decision at 0.12 threshold:** :{band[1]}[{decision}]")

    with st.expander("Show this transaction's (scaled) feature values"):
        st.dataframe(X_test.loc[[idx]].T, use_container_width=True)
