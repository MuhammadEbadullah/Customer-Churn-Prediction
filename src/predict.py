"""
Score customers with the saved churn model.

Usage:
    python src/predict.py                      # scores 10 random rows from data/churn_data.csv
    python src/predict.py path/to/new_file.csv # scores your own file (same columns as training data)

Output: outputs/predictions.csv with churn probability and a contact / don't contact flag.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
OUT_PATH = ROOT / "outputs" / "predictions.csv"


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run  python src/train.py  first.")

    bundle = joblib.load(MODEL_PATH)
    pipe, threshold, features = bundle["pipeline"], bundle["threshold"], bundle["features"]

    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
    else:
        df = pd.read_csv(ROOT / "data" / "churn_data.csv").sample(10, random_state=1)

    proba = pipe.predict_proba(df[features])[:, 1]
    out = df.copy()
    out["churn_probability"] = proba.round(3)
    out["action"] = ["Contact with offer" if p >= threshold else "No action" for p in proba]
    out = out.sort_values("churn_probability", ascending=False)

    OUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    show = [c for c in ["customer_id", "contract_type", "tenure_months", "churn_probability", "action"] if c in out.columns]
    print(f"Decision threshold used: {threshold:.2f}")
    print(out[show].to_string(index=False))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
