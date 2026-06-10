"""
Phase 1 — Data loading & first-look exploration.

Why this file exists:
    We keep loading logic OUT of the notebook so it is reusable and testable.
    The notebook should *call* these functions, not redefine them. That is the
    difference between a script and a portfolio-quality project.

Dataset (start here): ULB Credit Card Fraud
    Kaggle: mlg-ulb/creditcardfraud
    Columns: Time, V1..V28 (PCA-anonymized), Amount, Class (1 = fraud)
"""

from pathlib import Path
import pandas as pd

# Project-root-relative default path so the code works from notebook OR script.
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"


def load_data(path: str | Path = DEFAULT_DATA_PATH, target: str = "Class") -> pd.DataFrame:
    """
    Load the transaction dataset and run a quick sanity check.

    Why the checks: before any modeling we confirm the file loaded, the target
    exists, and we *see* how rare fraud is. That fraud rate is the single most
    important number in the whole project — it dictates every later decision.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Download the dataset from Kaggle (mlg-ulb/creditcardfraud) and place "
            "creditcard.csv in data/raw/. See README for instructions."
        )

    df = pd.read_csv(path)

    if target not in df.columns:
        raise KeyError(f"Expected target column '{target}' not found. Columns: {list(df.columns)}")

    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Fraud rate: {df[target].mean():.4%}  ({df[target].sum():,} fraud cases)")
    return df


def summarize(df: pd.DataFrame, target: str = "Class") -> None:
    """
    Print a compact first-look summary: dtypes, missing values, duplicates,
    and class balance. Run this once at the top of EDA.

    Why: you should never start modeling without knowing your data's shape,
    quality, and imbalance. This is the 'know your data' step.
    """
    print("=" * 60)
    print("SHAPE:", df.shape)

    print("\nMISSING VALUES (non-zero only):")
    missing = df.isna().sum()
    print(missing[missing > 0] if (missing > 0).any() else "  None ✓")

    print(f"\nDUPLICATE ROWS: {df.duplicated().sum():,}")

    print("\nCLASS BALANCE:")
    counts = df[target].value_counts()
    pct = df[target].value_counts(normalize=True)
    for cls in counts.index:
        label = "FRAUD" if cls == 1 else "LEGIT"
        print(f"  {cls} ({label}): {counts[cls]:>8,}  ({pct[cls]:.4%})")
    print("=" * 60)


def amount_by_class(df: pd.DataFrame, target: str = "Class") -> pd.DataFrame:
    """
    Compare the Amount distribution for fraud vs legit transactions.

    Why: 'do frauds look different in size?' is the first business question.
    This single groupby often reveals the most quotable insight in the project.
    """
    return df.groupby(target)["Amount"].describe()


if __name__ == "__main__":
    # Allows: python src/data_loader.py  (quick smoke test once data is present)
    data = load_data()
    summarize(data)
    print("\nAmount by class:\n", amount_by_class(data))
