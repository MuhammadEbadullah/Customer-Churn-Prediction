"""
Customer Churn Classifier - training, evaluation and threshold optimisation.

Author : Muhammad Ebadullah

Steps:
    1. load data
    2. preprocessing pipeline (impute + scale + one-hot)
    3. compare 3 models with 5-fold cross validation
    4. pick the best one, evaluate on a held-out test set
       (precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix)
    5. choose the decision threshold by business cost instead of the default 0.5
    6. save plots, metrics and the trained model

Run:  python src/train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no popup windows, just save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Settings you can play with
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Business assumptions (change these to match your own scenario)
CUSTOMER_VALUE = 300      # money we keep if a churner is saved (USD)
OFFER_COST = 30           # cost of one retention offer / call (USD)
SAVE_RATE = 0.50          # share of contacted churners the offer actually saves

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "churn_data.csv"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

TARGET = "churn"
ID_COL = "customer_id"

sns.set_theme(style="whitegrid", context="notebook")
COLORS = {"blue": "#2b6cb0", "orange": "#dd6b20", "green": "#2f855a", "red": "#c53030", "grey": "#718096"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def net_benefit(y_true, y_pred):
    """Money gained by running the retention campaign vs doing nothing."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    gain_per_tp = SAVE_RATE * CUSTOMER_VALUE - OFFER_COST
    loss_per_fp = OFFER_COST
    return tp * gain_per_tp - fp * loss_per_fp


def scores_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "net_benefit": round(float(net_benefit(y_true, pred)), 2),
    }


def plot_confusion(y_true, proba, threshold, title, path):
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Stays", "Churns"], yticklabels=["Stays", "Churns"],
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_preprocessor(num_cols, cat_cols):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    if not DATA_PATH.exists():
        raise SystemExit(f"Data file not found: {DATA_PATH}\nRun  python src/generate_data.py  first.")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df[TARGET].mean():.1%}")

    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # class balance plot
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.map({0: "Stayed", 1: "Churned"}).value_counts()
    ax.bar(counts.index, counts.values, color=[COLORS["blue"], COLORS["orange"]])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v} ({v / len(y):.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_balance.png", dpi=150)
    plt.close(fig)

    # 2. candidate models -----------------------------------------------------
    pre = build_preprocessor(num_cols, cat_cols)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=250, random_state=RANDOM_STATE
        ),
    }
    pipes = {name: Pipeline([("prep", pre), ("model", m)]) for name, m in candidates.items()}

    # 3. cross validation on the training set ----------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in pipes.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=1)
        rows.append(
            {
                "model": name,
                "cv_roc_auc": res["test_roc_auc"].mean(),
                "cv_roc_auc_std": res["test_roc_auc"].std(),
                "cv_pr_auc": res["test_average_precision"].mean(),
            }
        )
        print(f"  {name:<20} CV ROC-AUC = {rows[-1]['cv_roc_auc']:.4f} (+/- {rows[-1]['cv_roc_auc_std']:.4f})")

    cv_table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    best_name = cv_table.loc[0, "model"]
    print(f"Best model by CV ROC-AUC: {best_name}")

    # 4. fit all on train, evaluate on test (for the comparison plots) ----------
    test_proba = {}
    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        test_proba[name] = pipe.predict_proba(X_test)[:, 1]

    cv_table["test_roc_auc"] = cv_table["model"].map(lambda n: roc_auc_score(y_test, test_proba[n]))
    cv_table["test_pr_auc"] = cv_table["model"].map(lambda n: average_precision_score(y_test, test_proba[n]))
    cv_table.round(4).to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # ROC curves
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for (name, p), c in zip(test_proba.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["grey"], label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_roc_curves.png", dpi=150)
    plt.close(fig)

    # Precision-recall curves
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for (name, p), c in zip(test_proba.items(), palette):
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, color=c, lw=2, label=f"{name} (AP = {average_precision_score(y_test, p):.3f})")
    ax.axhline(y_test.mean(), ls="--", color=COLORS["grey"], label="Baseline (churn rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_precision_recall_curves.png", dpi=150)
    plt.close(fig)

    # 5. threshold selection (done on out-of-fold train predictions, so the
    #    test set stays untouched until the final report) ------------------------
    best_pipe = pipes[best_name]
    oof_proba = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    benefit_curve = np.array([net_benefit(y_train, (oof_proba >= t).astype(int)) for t in grid])
    f1_curve = np.array([f1_score(y_train, (oof_proba >= t).astype(int)) for t in grid])
    best_t = float(grid[benefit_curve.argmax()])
    f1_t = float(grid[f1_curve.argmax()])
    print(f"Business-optimal threshold: {best_t:.2f}   (F1-optimal: {f1_t:.2f}, default: 0.50)")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(grid, benefit_curve, color=COLORS["green"], lw=2.2, label="Net benefit (train, out-of-fold)")
    ax1.axvline(best_t, color=COLORS["green"], ls="--", lw=1.2)
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Net benefit (USD)", color=COLORS["green"])
    ax2 = ax1.twinx()
    ax2.grid(False)
    prec_c = [precision_score(y_train, (oof_proba >= t).astype(int), zero_division=0) for t in grid]
    rec_c = [recall_score(y_train, (oof_proba >= t).astype(int)) for t in grid]
    ax2.plot(grid, prec_c, color=COLORS["blue"], lw=1.6, label="Precision")
    ax2.plot(grid, rec_c, color=COLORS["orange"], lw=1.6, label="Recall")
    ax2.set_ylabel("Precision / Recall")
    ax2.set_ylim(0, 1.02)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=9)
    ax1.set_title(f"Choosing the threshold by business value (best = {best_t:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_threshold_optimisation.png", dpi=150)
    plt.close(fig)

    # 6. final evaluation on the test set ---------------------------------------
    final_proba = test_proba[best_name]
    at_default = scores_at(y_test, final_proba, 0.5)
    at_best = scores_at(y_test, final_proba, best_t)
    plot_confusion(y_test, final_proba, 0.5, "Confusion matrix @ threshold 0.50", FIG_DIR / "05_confusion_default.png")
    plot_confusion(
        y_test, final_proba, best_t, f"Confusion matrix @ threshold {best_t:.2f}", FIG_DIR / "06_confusion_optimised.png"
    )

    # permutation importance (on raw columns, so it is easy to read)
    imp = permutation_importance(
        best_pipe, X_test, y_test, scoring="roc_auc", n_repeats=8, random_state=RANDOM_STATE, n_jobs=1
    )
    imp_df = (
        pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["feature"], imp_df["importance"], xerr=imp_df["std"], color=COLORS["blue"])
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("What drives churn? (permutation importance)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_feature_importance.png", dpi=150)
    plt.close(fig)
    imp_df.sort_values("importance", ascending=False).round(4).to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # how many customers would we contact, and what is the total money effect?
    n_test = len(y_test)
    summary = {
        "best_model": best_name,
        "n_train": int(len(y_train)),
        "n_test": int(n_test),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, final_proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, final_proba)), 4),
        "chosen_threshold": best_t,
        "f1_optimal_threshold": f1_t,
        "at_threshold_0.50": at_default,
        "at_chosen_threshold": at_best,
        "business_assumptions": {
            "customer_value": CUSTOMER_VALUE,
            "offer_cost": OFFER_COST,
            "save_rate": SAVE_RATE,
        },
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    report = [
        f"Best model: {best_name}",
        f"Test ROC-AUC: {summary['test_roc_auc']}   Test PR-AUC: {summary['test_pr_auc']}",
        "",
        "Threshold 0.50 (default)",
        json.dumps(at_default, indent=2),
        "",
        f"Threshold {best_t:.2f} (chosen by net benefit)",
        json.dumps(at_best, indent=2),
    ]
    (OUT_DIR / "classification_report.txt").write_text("\n".join(report), encoding="utf-8")

    # save model + threshold together so predict.py can use both
    joblib.dump({"pipeline": best_pipe, "threshold": best_t, "features": X.columns.tolist()}, MODEL_DIR / "churn_model.joblib")

    print("\n=== Test set results ===")
    print(f"ROC-AUC {summary['test_roc_auc']}   PR-AUC {summary['test_pr_auc']}")
    for label, s in [("threshold 0.50", at_default), (f"threshold {best_t:.2f}", at_best)]:
        print(
            f"{label}: precision {s['precision']:.2f}  recall {s['recall']:.2f}  F1 {s['f1']:.2f}  "
            f"TP {s['tp']} FP {s['fp']} FN {s['fn']} TN {s['tn']}  net benefit {s['net_benefit']:.0f} USD"
        )
    print(f"\nPlots saved in {FIG_DIR}")
    print(f"Model saved in {MODEL_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
