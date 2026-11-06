"""
Generates a synthetic customer behaviour dataset for the churn project.

Why synthetic? It keeps the repo self-contained (no Kaggle login needed) and
lets anyone reproduce the exact same results. If you want to use a real
dataset (e.g. the Telco Customer Churn CSV from Kaggle), skip this script and
see the "Using your own data" section in the README.

Run:  python src/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 7000

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "churn_data.csv"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    rng = np.random.default_rng(SEED)
    n = N_CUSTOMERS

    # --- account / billing info -------------------------------------------
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    plan = rng.choice(["Basic", "Standard", "Premium"], size=n, p=[0.40, 0.40, 0.20])
    payment = rng.choice(
        ["Card", "Bank transfer", "Mobile wallet", "Cash"], size=n, p=[0.35, 0.25, 0.25, 0.15]
    )

    tenure = rng.exponential(scale=22, size=n) + 1
    tenure += np.where(contract == "One year", 10, 0) + np.where(contract == "Two year", 20, 0)
    tenure = np.clip(tenure, 1, 72).round().astype(int)

    base_price = pd.Series(plan).map({"Basic": 35, "Standard": 60, "Premium": 95}).to_numpy()
    monthly_charges = np.clip(base_price + rng.normal(0, 8, size=n), 15, 140).round(2)

    autopay = ((payment != "Cash") & (rng.random(n) < 0.7)).astype(int)

    # --- behavioural signals ------------------------------------------------
    # one hidden "engagement" factor drives most of the usage columns
    engagement = rng.normal(0, 1, size=n)

    logins_last_30d = np.clip((14 + 5 * engagement + rng.normal(0, 3, n)).round(), 0, 60).astype(int)
    avg_session_minutes = np.clip(18 + 6 * engagement + rng.normal(0, 4, n), 1, 90).round(1)
    feature_usage_score = np.clip(50 + 15 * engagement + rng.normal(0, 8, n), 0, 100).round(1)
    days_since_last_login = np.clip(
        (rng.exponential(6, n) - 2.5 * engagement).round(), 0, 90
    ).astype(int)
    support_tickets_90d = rng.poisson(np.where(engagement < -0.5, 1.8, 0.9))
    late_payments_6m = rng.poisson(np.where(payment == "Cash", 0.9, 0.35))

    # --- churn label ------------------------------------------------------------
    logit = (
        -1.55
        + 1.2 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        - 0.03 * tenure
        + 0.012 * (monthly_charges - 60)
        - 0.55 * engagement
        + 0.04 * days_since_last_login
        + 0.25 * support_tickets_90d
        + 0.35 * late_payments_6m
        - 0.45 * autopay
        + rng.normal(0, 0.6, n)  # noise so the problem is not trivially separable
    )
    churn = (rng.random(n) < sigmoid(logit)).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "contract_type": contract,
            "plan_type": plan,
            "payment_method": payment,
            "has_autopay": autopay,
            "monthly_charges": monthly_charges,
            "logins_last_30d": logins_last_30d,
            "avg_session_minutes": avg_session_minutes,
            "feature_usage_score": feature_usage_score,
            "days_since_last_login": days_since_last_login,
            "support_tickets_90d": support_tickets_90d,
            "late_payments_6m": late_payments_6m,
            "churn": churn,
        }
    )

    # a few missing values, like real data has
    for col in ["avg_session_minutes", "monthly_charges", "feature_usage_score"]:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(f"Churn rate: {df['churn'].mean():.1%}")


if __name__ == "__main__":
    main()
