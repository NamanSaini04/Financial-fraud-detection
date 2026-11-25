"""
Financial Fraud Detection - interactive Streamlit dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Tabs:
    1. Overview   - KPI cards + the cost-optimal operating point (driven by the
                    sidebar cost model)
    2. Explore    - the saved EDA figures
    3. Threshold  - LIVE slider with interactive cost + PR curves
    4. Score      - sample a real transaction, see a risk gauge + decision

The sidebar "business cost model" is the key interactive idea: the recommended
threshold is NOT hard-coded - it is recomputed from the cost you assign to a
missed fraud vs a false alarm. Change those and watch the recommendation move.

All metrics are on the untouched test set (~0.17% fraud), reusing the exact
pipeline artifacts (saved XGBoost model, leak-safe split, fitted scaler).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, average_precision_score,
                             roc_auc_score, precision_recall_curve)

# --- paths (relative to project root so it runs from anywhere) ---
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
MODEL_PATH = ROOT / "models" / "best_xgboost.joblib"
REPO_URL = "https://github.com/NamanSaini04/Financial-fraud-detection"

# palette
ACCENT = "#e15759"      # red (fraud / primary)
GOOD = "#59a14f"        # green
MUTED = "#8a8d98"

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")


# ----------------------------------------------------------------------------
# Cached loaders - run once, not on every widget change
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
    model = load_model()
    X_test, y_test = load_test()
    return model.predict_proba(X_test)[:, 1], y_test.values


@st.cache_data
def pr_curve():
    return precision_recall_curve(y_true, y_proba)


def counts_at(threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, recall=recall, precision=precision)


def cost_at(threshold, cost_fn, cost_fp):
    c = counts_at(threshold)
    return c["fn"] * cost_fn + c["fp"] * cost_fp


@st.cache_data
def cost_sweep(cost_fn, cost_fp):
    """Cost at each threshold + the cost-minimising threshold (cached per cost pair)."""
    ts = np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = np.array([cost_at(t, cost_fn, cost_fp) for t in ts])
    opt_t = float(ts[int(np.argmin(costs))])
    return ts, costs, opt_t


# ----------------------------------------------------------------------------
# Guard: artifacts must exist
# ----------------------------------------------------------------------------
if not MODEL_PATH.exists() or not (PROC / "test.parquet").exists():
    st.error("Missing artifacts. Run the pipeline first:\n\n"
             "`python src/preprocessing.py && python src/models.py`")
    st.stop()

y_proba, y_true = get_scores()
N_TOTAL = 283_726
FRAUD_RATE = 0.0017
PR_AUC = average_precision_score(y_true, y_proba)
ROC_AUC = roc_auc_score(y_true, y_proba)

# ----------------------------------------------------------------------------
# Sidebar - the interactive business cost model
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Business cost model")
    st.caption("The recommended threshold is derived from these - not hard-coded. "
               "Change them to match your business and watch it move.")
    cost_fn = st.number_input("Cost of a **missed fraud** (FN), $", 10, 2000, 150, step=10)
    cost_fp = st.number_input("Cost of a **false alarm** (FP), $", 1, 200, 5, step=1)
    ratio = cost_fn / cost_fp
    st.caption(f"A missed fraud costs **{ratio:.0f}×** a false alarm.")

    ts, costs, OPT_T = cost_sweep(cost_fn, cost_fp)
    st.metric("💡 Cost-optimal threshold", f"{OPT_T:.2f}")

    st.divider()
    st.markdown(f"**Model:** XGBoost · PR-AUC {PR_AUC:.2f}")
    st.markdown("**Data:** ULB credit-card (0.17% fraud)")
    st.markdown(f"[📂 GitHub repo]({REPO_URL})")
    st.caption("Metrics on held-out test set.")

opt = counts_at(OPT_T)
base_cost = cost_at(0.50, cost_fn, cost_fp)
opt_cost = cost_at(OPT_T, cost_fn, cost_fp)
savings_pct = (base_cost - opt_cost) / base_cost * 100 if base_cost else 0

# ----------------------------------------------------------------------------
st.title("💳 Financial Fraud Detection Dashboard")
st.caption(f"XGBoost on the ULB credit-card dataset. Cost model (editable in the "
           f"sidebar): missed fraud = ${cost_fn}, false alarm = ${cost_fp}.")

tab_overview, tab_explore, tab_threshold, tab_score = st.tabs(
    ["📊 Overview", "🔍 Explore", "🎚️ Threshold simulator", "🧾 Score a transaction"])

# ----------------------------------------------------------------------------
# Tab 1 - Overview
# ----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Model quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analysed", f"{N_TOTAL:,}")
    c2.metric("Fraud rate", f"{FRAUD_RATE:.2%}")
    c3.metric("PR-AUC", f"{PR_AUC:.3f}", help="Lead metric for rare-event detection")
    c4.metric("ROC-AUC", f"{ROC_AUC:.3f}", help="Optimistic under heavy imbalance")

    st.divider()
    st.subheader(f"Recommended operating point — threshold {OPT_T:.2f}")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Fraud caught (recall)", f"{opt['recall']:.0%}")
    o2.metric("Flags that are real (precision)", f"{opt['precision']:.0%}")
    o3.metric("Review queue", f"{opt['tp'] + opt['fp']}", help="Transactions flagged on the test set")
    o4.metric("Est. cost", f"${opt_cost:,.0f}", delta=f"-{savings_pct:.0f}% vs default 0.50",
              delta_color="inverse")

    st.success(
        f"**Recommendation:** operate at the **{OPT_T:.2f}** threshold. It catches "
        f"~{opt['recall']:.0%} of fraud while sending only {opt['tp'] + opt['fp']} of "
        f"{len(y_true):,} test transactions to review — about **{savings_pct:.0f}% lower "
        f"cost** than the naive 0.50 default, under the current cost assumptions. "
        "Adjust the costs in the sidebar to see how the recommendation shifts."
    )

# ----------------------------------------------------------------------------
# Tab 2 - Explore
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Exploratory analysis")
    st.caption("Key patterns found before modelling. Full write-up in the repo's reports/.")
    figs = [
        ("01_class_balance.png", "Class balance — fraud is 0.17% of transactions."),
        ("03_fraud_rate_by_hour.png", "Fraud rate peaks at 2–4 AM (~8× baseline)."),
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
# Tab 3 - Threshold simulator
# ----------------------------------------------------------------------------
with tab_threshold:
    st.subheader("Drag the threshold — watch the trade-off move")
    threshold = st.slider("Decision threshold", 0.01, 0.99, OPT_T, 0.01,
                          help="Probabilities at or above this are flagged as fraud.")
    m = counts_at(threshold)
    cur_cost = cost_at(threshold, cost_fn, cost_fp)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recall (fraud caught)", f"{m['recall']:.0%}")
    k2.metric("Precision (flags real)", f"{m['precision']:.0%}")
    k3.metric("Missed fraud (FN)", f"{m['fn']}")
    k4.metric("False alarms (FP)", f"{m['fp']}")
    st.metric("Estimated cost at this threshold", f"${cur_cost:,.0f}",
              delta=f"{cur_cost - base_cost:,.0f} vs default 0.50", delta_color="inverse")

    left, right = st.columns(2)

    # --- cost curve ---
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=ts, y=costs, mode="lines", line=dict(color=ACCENT, width=3),
                                  name="cost", hovertemplate="thr %{x:.2f}<br>$%{y:,.0f}<extra></extra>"))
    fig_cost.add_vline(x=OPT_T, line=dict(color=GOOD, dash="dot"),
                       annotation_text=f"optimal {OPT_T:.2f}", annotation_position="top")
    fig_cost.add_vline(x=threshold, line=dict(color="#ffffff", dash="dash"))
    fig_cost.add_trace(go.Scatter(x=[threshold], y=[cur_cost], mode="markers",
                                  marker=dict(color="#ffffff", size=11), name="current",
                                  hovertemplate="you<br>$%{y:,.0f}<extra></extra>"))
    fig_cost.update_layout(title="Expected cost vs threshold", height=330,
                           margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                           template="plotly_dark", xaxis_title="threshold", yaxis_title="cost ($)")
    left.plotly_chart(fig_cost, use_container_width=True)

    # --- PR curve with current operating point ---
    prec, rec, _ = pr_curve()
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color=ACCENT, width=3),
                                name="PR", hovertemplate="recall %{x:.2f}<br>prec %{y:.2f}<extra></extra>"))
    fig_pr.add_hline(y=FRAUD_RATE, line=dict(color=MUTED, dash="dot"),
                     annotation_text="no-skill", annotation_position="bottom right")
    fig_pr.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers",
                                marker=dict(color="#ffffff", size=13, line=dict(color=ACCENT, width=2)),
                                name="current", hovertemplate="you<br>recall %{x:.2f}<br>prec %{y:.2f}<extra></extra>"))
    fig_pr.update_layout(title=f"PR curve (AP={PR_AUC:.2f}) — ⚪ = your threshold", height=330,
                         margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                         template="plotly_dark", xaxis_title="recall", yaxis_title="precision",
                         xaxis_range=[0, 1], yaxis_range=[0, 1.02])
    right.plotly_chart(fig_pr, use_container_width=True)

    if threshold < 0.05:
        st.warning("Very low threshold: you catch more fraud but false alarms explode — "
                   "analysts get buried and cost rises.")
    elif threshold > 0.6:
        st.warning("High threshold: few false alarms, but you miss more fraud.")
    else:
        st.success(f"Balanced zone. The cost-minimising threshold for these assumptions "
                   f"is **{OPT_T:.2f}** (green line).")

# ----------------------------------------------------------------------------
# Tab 4 - Score a transaction
# ----------------------------------------------------------------------------
with tab_score:
    st.subheader("Score a real transaction")
    st.caption("Features are PCA-anonymised, so we sample an actual test transaction "
               "rather than typing 28 opaque values by hand.")

    X_test, y_test = load_test()
    colA, colB = st.columns([1, 1])
    pick_fraud = colB.checkbox("Sample only known-fraud cases", value=False)
    if colA.button("🎲 Pick a random transaction", type="primary") or "row_idx" not in st.session_state:
        pool = y_test[y_test == 1].index if pick_fraud else y_test.index
        st.session_state.row_idx = int(np.random.choice(pool))

    idx = st.session_state.row_idx
    pos = X_test.index.get_loc(idx)
    proba = float(y_proba[pos])
    actual = int(y_test.loc[idx])
    flagged = proba >= OPT_T

    gcol, icol = st.columns([1.1, 1])

    # risk gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Fraud probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ffffff", "thickness": 0.25},
            "steps": [
                {"range": [0, OPT_T * 100], "color": GOOD},
                {"range": [OPT_T * 100, 50], "color": "#e8a33d"},
                {"range": [50, 100], "color": ACCENT},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": OPT_T * 100},
        },
    ))
    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10), template="plotly_dark")
    gcol.plotly_chart(gauge, use_container_width=True)

    band = "🟢 LOW" if proba < OPT_T else "🟠 MEDIUM" if proba < 0.5 else "🔴 HIGH"
    icol.metric("Risk band", band)
    icol.metric("Actual label", "FRAUD" if actual else "Legit")
    icol.metric("Decision @ "f"{OPT_T:.2f}", "🚩 FLAG for review" if flagged else "✅ Allow")
    if flagged == bool(actual):
        icol.success("Model decision matches the true label.")
    else:
        icol.error("Model decision does **not** match the true label.")

    with st.expander("Show this transaction's (scaled) feature values"):
        st.dataframe(X_test.loc[[idx]].T, use_container_width=True)
