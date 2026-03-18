"""
utils.py — Shared helper functions for the Bank Customer Churn Prediction project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, average_precision_score,
    precision_recall_curve
)

# ── Plotting defaults ──────────────────────────────────────────────────────────

PLOT_STYLE = {
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

def set_plot_style():
    """Apply consistent plot styling across all notebooks."""
    plt.rcParams.update(PLOT_STYLE)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw(path: str = "data/raw/bank_churn.csv") -> pd.DataFrame:
    """Load and do minimal type-casting on the raw dataset."""
    df = pd.read_csv(path)
    return df


def load_processed(path: str = "data/processed/df_validated.parquet") -> pd.DataFrame:
    """Load the validated, cleaned dataset."""
    return pd.read_parquet(path)


# ── Feature engineering ────────────────────────────────────────────────────────

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three business-hypothesis-driven features:
      - BalanceSalaryRatio : financial stress relative to income
      - ProductsPerYear    : product adoption rate relative to tenure
      - IsHighValue        : flag for high-balance, multi-product customers
    """
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductsPerYear"]    = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["IsHighValue"]        = (
        (df["Balance"] > df["Balance"].quantile(0.75)) &
        (df["NumOfProducts"] >= 2)
    ).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Geography and Gender (France as reference category)."""
    return pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)


FEATURE_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "BalanceSalaryRatio", "ProductsPerYear", "IsHighValue",
    "Geography_Germany", "Geography_Spain", "Gender_Male",
]


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def print_classification_report(y_true, y_pred, y_prob=None):
    """Print a full classification report plus ROC-AUC and AP if probabilities supplied."""
    print(classification_report(y_true, y_pred))
    if y_prob is not None:
        print(f"ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")
        print(f"Avg Prec: {average_precision_score(y_true, y_prob):.4f}")


def plot_roc_curve(y_true, y_prob, label: str = "Model", ax=None):
    """Plot an ROC curve on the given axes (or create new ones)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_confusion_matrix(y_true, y_pred, labels=("No Churn", "Churn"), ax=None):
    """Plot a labelled confusion matrix heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return ax


# ── Business / cost analysis ───────────────────────────────────────────────────

def compute_cost_curve(
    y_true,
    y_prob,
    cost_fn: float = 800.0,
    cost_fp: float = 30.0,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """
    Compute the total business cost at each decision threshold.

    Parameters
    ----------
    y_true       : true binary labels
    y_prob       : predicted probabilities for the positive class
    cost_fn      : cost of a false negative (missed churner), default €800
    cost_fp      : cost of a false positive (unnecessary retention offer), default €30
    n_thresholds : number of threshold values to evaluate

    Returns
    -------
    DataFrame with columns [threshold, cost, fn, fp, tp, tn]
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "cost": fn * cost_fn + fp * cost_fp,
            "fn": fn, "fp": fp, "tp": tp, "tn": tn,
        })
    return pd.DataFrame(rows)


def plot_cost_curve(cost_df: pd.DataFrame, ax=None):
    """Plot total business cost vs decision threshold, marking the optimum."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cost_df["threshold"], cost_df["cost"], color="steelblue")
    opt = cost_df.loc[cost_df["cost"].idxmin()]
    ax.axvline(opt["threshold"], color="crimson", linestyle="--",
               label=f"Optimum = {opt['threshold']:.2f}  (€{opt['cost']:,.0f})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Cost (€)")
    ax.set_title("Cost-Sensitive Threshold Selection")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend()
    return ax, opt
