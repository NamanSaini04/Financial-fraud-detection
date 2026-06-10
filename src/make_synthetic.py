"""
Generate a SYNTHETIC stand-in dataset that mirrors the ULB Credit Card schema.

⚠️  THIS IS FAKE DATA — for pipeline testing only, NOT analysis.
    Replace data/raw/creditcard.csv with the real Kaggle download
    (mlg-ulb/creditcardfraud) before drawing any real conclusions.

Schema matched: Time, V1..V28 (PCA-like), Amount, Class (1 = fraud, ~0.17%).
We deliberately make fraud slightly different in a couple of V-columns and in
Amount so the EDA charts show *something* — real fraud separation will differ.
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"
N = 50_000
FRAUD_RATE = 0.0017  # ~ same as the real dataset
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_fraud = int(N * FRAUD_RATE)
    n_legit = N - n_fraud

    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # V1..V28 — standard normal, like PCA components
    V = rng.standard_normal((N, 28))
    # Nudge a couple of columns for the fraud class so EDA has signal
    V[y == 1, 13] += 2.5   # mimic the real V14 separation
    V[y == 1, 9] -= 2.0

    # Time — seconds over ~2 days
    time = rng.integers(0, 172_792, size=N)

    # Amount — legit log-normal small; fraud skews to small rapid amounts + a few large
    amount = np.empty(N)
    amount[y == 0] = rng.lognormal(mean=3.0, sigma=1.2, size=n_legit)
    amount[y == 1] = rng.lognormal(mean=2.0, sigma=1.5, size=n_fraud)
    amount = np.round(np.clip(amount, 0, 25000), 2)

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = y

    # Shuffle so fraud isn't all at the bottom
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"⚠️  Wrote SYNTHETIC data to {OUT}")
    print(f"    {len(df):,} rows, fraud rate {df['Class'].mean():.4%}")


if __name__ == "__main__":
    main()
