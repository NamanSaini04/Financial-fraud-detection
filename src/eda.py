"""
Phase 3 — Deep EDA: what actually separates fraud from legit?

These helpers return BOTH the numbers and a saved figure, so the notebook can
tell a story and the README can embed the charts. Every function works on the
RAW dataframe (pre-scaling) so the numbers stay human-readable (real dollars,
real hours).

Honesty note: V1..V28 are PCA-anonymized. We can say "V14 separates the classes
strongly" but we CANNOT say what V14 means in business terms. We are explicit
about that limitation rather than inventing a narrative.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # file output, no GUI needed
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def add_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day from Time (seconds since first transaction)."""
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    return df


def fraud_rate_by_hour(df: pd.DataFrame, target: str = "Class") -> pd.DataFrame:
    """
    Fraud RATE (not count) per hour-of-day. Rate matters because raw counts just
    track overall traffic; rate tells you when a transaction is *riskier*.
    """
    df = add_hour(df)
    by_hour = (df.groupby("Hour")[target]
                 .agg(n="size", fraud="sum"))
    by_hour["fraud_rate"] = by_hour["fraud"] / by_hour["n"]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=by_hour.index, y=by_hour["fraud_rate"], color="#c0392b", ax=ax)
    ax.axhline(df[target].mean(), ls="--", color="gray",
               label=f"overall avg ({df[target].mean():.3%})")
    ax.set_title("Fraud Rate by Hour of Day")
    ax.set_xlabel("Hour"); ax.set_ylabel("Fraud rate"); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_fraud_rate_by_hour.png", dpi=110)
    plt.close()
    return by_hour


def feature_separation(df: pd.DataFrame, target: str = "Class", top: int = 8) -> pd.Series:
    """
    Rank features by absolute correlation with the target. This points us at the
    columns that most separate fraud from legit, so we don't boxplot all 30.
    """
    corr = df.corr(numeric_only=True)[target].drop(target)
    ranked = corr.abs().sort_values(ascending=False)
    return corr[ranked.index][:top]   # signed corr, ordered by strength


def boxplots_top_features(df: pd.DataFrame, target: str = "Class", top: int = 4):
    """Boxplots of the most-separating features, fraud vs legit."""
    top_feats = feature_separation(df, target, top=top).index.tolist()
    fig, axes = plt.subplots(1, len(top_feats), figsize=(4 * len(top_feats), 4))
    for ax, feat in zip(axes, top_feats):
        sns.boxplot(data=df, x=target, y=feat, ax=ax, showfliers=False)
        ax.set_title(f"{feat} (|corr|={df[feat].corr(df[target]):.2f})")
        ax.set_xlabel("0=Legit  1=Fraud")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_top_feature_boxplots.png", dpi=110)
    plt.close()
    return top_feats


def correlation_heatmap(df: pd.DataFrame, target: str = "Class"):
    """Correlation of each feature with the target, as a single readable strip."""
    corr = df.corr(numeric_only=True)[[target]].drop(target).sort_values(target)
    fig, ax = plt.subplots(figsize=(4, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": f"corr with {target}"})
    ax.set_title("Feature Correlation with Fraud")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_corr_with_target.png", dpi=110)
    plt.close()
    return corr


def class_balance_plot(df: pd.DataFrame, target: str = "Class"):
    """Bar chart of class counts — the imbalance, made visible (figure 01)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target, ax=ax)
    ax.set_title("Class Balance (0=Legit, 1=Fraud) — ULB real data")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_class_balance.png", dpi=110)
    plt.close()


def amount_by_class_plot(df: pd.DataFrame, target: str = "Class"):
    """KDE of Amount for fraud vs legit, clipped for readability (figure 02)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=df[df[target] == 0], x="Amount", label="Legit", ax=ax)
    sns.kdeplot(data=df[df[target] == 1], x="Amount", label="Fraud", ax=ax)
    ax.set_xlim(0, 500)
    ax.legend()
    ax.set_title("Amount Distribution by Class (clipped to $500) — ULB real data")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_amount_by_class.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data
    df = clean_data(load_data())          # use cleaned (deduped) data for honest EDA

    class_balance_plot(df)
    amount_by_class_plot(df)

    print("\n=== Fraud rate by hour ===")
    print(fraud_rate_by_hour(df)[["n", "fraud", "fraud_rate"]])

    print("\n=== Top features by |correlation with Class| ===")
    print(feature_separation(df).round(3))

    boxplots_top_features(df)
    correlation_heatmap(df)
    print(f"\nFigures saved to {FIG_DIR}")
