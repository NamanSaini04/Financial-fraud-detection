"""
Phase 2 — Clean & Preprocess.

The golden rule of this file: NO DATA LEAKAGE.
    Everything that "learns" from the data (scaler statistics, and later SMOTE)
    must see the TRAIN set only. The test set must stay untouched until the very
    end, so it honestly simulates unseen future transactions.

Correct order (do not reorder):
    1. clean_data        -> drop duplicates, sanity-check
    2. split_data        -> stratified train/test split  (split FIRST)
    3. scale_features    -> fit RobustScaler on TRAIN, transform both

Why RobustScaler (not StandardScaler): it centers/scales using the median and
IQR, so extreme transaction amounts don't distort the scaling. V1..V28 are
already PCA-scaled, so we only scale the two raw columns: Amount and Time.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RAW_COLS_TO_SCALE = ["Amount", "Time"]
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def clean_data(df: pd.DataFrame, target: str = "Class") -> pd.DataFrame:
    """
    Drop exact duplicate rows and run cheap integrity checks.

    Why drop duplicates: identical rows can land in BOTH train and test, leaking
    information and inflating scores. The ULB set has ~1,081 of them.

    Why we DON'T drop outliers: in fraud, the outlier often *is* the fraud. A
    huge or oddly-timed transaction is signal, not noise — we keep it.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df):,} duplicate rows ({len(df):,} remain)")

    # Integrity checks — fail loud rather than train on corrupt data
    assert df.isna().sum().sum() == 0, "Unexpected missing values present"
    assert (df["Amount"] >= 0).all(), "Negative transaction amounts found"
    df[target] = df[target].astype(int)
    return df


def split_data(df: pd.DataFrame, target: str = "Class",
               test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split. SPLIT BEFORE SCALING OR RESAMPLING.

    Why stratify=y: with only ~0.17% fraud, a random split could put very few
    (or zero) fraud cases in the test set. Stratifying keeps the fraud ratio
    identical in train and test so evaluation is meaningful.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]:,} rows ({y_train.sum()} fraud) | "
          f"Test: {X_test.shape[0]:,} rows ({y_test.sum()} fraud)")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   cols=RAW_COLS_TO_SCALE):
    """
    Fit RobustScaler on TRAIN ONLY, then transform train and test with it.

    Returning the fitted scaler matters: at prediction time (dashboard / live
    scoring) you must apply the SAME transformation, not refit. Refitting on new
    data is a classic leakage bug.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = RobustScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])   # fit on train
    X_test[cols] = scaler.transform(X_test[cols])         # transform only
    return X_train, X_test, scaler


def preprocess(df: pd.DataFrame, target: str = "Class", save: bool = False):
    """
    Convenience wrapper running the full Phase 2 pipeline in the correct order.
    Set save=True to persist processed splits + scaler for later phases.
    """
    df = clean_data(df, target)
    X_train, X_test, y_train, y_test = split_data(df, target)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    if save:
        import joblib
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{target: y_train.values}).to_parquet(PROCESSED_DIR / "train.parquet")
        X_test.assign(**{target: y_test.values}).to_parquet(PROCESSED_DIR / "test.parquet")
        joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")
        print(f"Saved processed splits + scaler to {PROCESSED_DIR}")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    from data_loader import load_data
    data = load_data()
    Xtr, Xte, ytr, yte, scaler = preprocess(data, save=True)
    print("\nScaled column preview (train head):")
    print(Xtr[RAW_COLS_TO_SCALE].head())
