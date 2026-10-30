"""
Shared preprocessing for the UCI Credit Card Default project.
=============================================================
Used by BOTH train.py and app.py so the data seen during training and the
data seen at prediction time is always identical.

NO PIPELINE: cleaning, feature engineering and scaling are done explicitly.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "UCI_Credit_Card.csv"

TARGET = "default.payment.next.month"

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #
ID_COL = "ID"

DEMO_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
# NOTE: UCI names the most recent month PAY_0 (it should logically be PAY_1).
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

RAW_FEATURES = DEMO_COLS + PAY_COLS + BILL_COLS + PAYAMT_COLS  # 23 columns

ENGINEERED_FEATURES = [
    "credit_utilization",       # BILL_AMT1 / LIMIT_BAL
    "avg_utilization_6m",       # mean(BILL_AMT1..6) / LIMIT_BAL
    "late_months",              # how many of the last 6 months were late
    "max_delay_months",         # worst repayment delay seen
    "payment_to_bill_ratio",    # PAY_AMT1 vs BILL_AMT1
    "bill_trend_6m",            # BILL_AMT1 - BILL_AMT6 (debt growing?)
]

FEATURE_COLS = RAW_FEATURES + ENGINEERED_FEATURES  # 29 model inputs

# --------------------------------------------------------------------------- #
# Human readable dictionaries (used by the UI)
# --------------------------------------------------------------------------- #
PAY_STATUS_LABELS = {
    -2: "-2 : No consumption",
    -1: "-1 : Paid in full",
    0: "0 : Revolving credit (minimum paid)",
    1: "1 : Payment 1 month late",
    2: "2 : Payment 2 months late",
    3: "3 : Payment 3 months late",
    4: "4 : Payment 4 months late",
    5: "5 : Payment 5 months late",
    6: "6 : Payment 6 months late",
    7: "7 : Payment 7 months late",
    8: "8 : Payment 8 months late",
    9: "9 : 9+ months late",
}

EDUCATION_LABELS = {
    1: "1 : Graduate school",
    2: "2 : University",
    3: "3 : High school",
    4: "4 : Other",
}

MARRIAGE_LABELS = {1: "1 : Married", 2: "2 : Single", 3: "3 : Other"}

SEX_LABELS = {1: "1 : Male", 2: "2 : Female"}

FEATURE_GROUPS = {
    "Demographics": DEMO_COLS,
    "Repayment status": PAY_COLS,
    "Billed amount": BILL_COLS,
    "Payment amount": PAYAMT_COLS,
    "Behavioral ratio": ENGINEERED_FEATURES,
}

# One clear definition for every column shown in the UI.
# "description" = what the attribute IS, "use" = what it is USED FOR by the model.
COLUMN_INFO = {
    "LIMIT_BAL": {
        "group": "Demographics",
        "type": "Numeric (NT$)",
        "description": "Approved credit limit, including the individual and family supplementary limit.",
        "use": "Shows how much the bank trusts this customer; it is also the base for the utilization ratios.",
    },
    "SEX": {
        "group": "Demographics",
        "type": "Categorical",
        "description": "Cardholder gender: 1 = male, 2 = female.",
        "use": "Background variable with a small but real difference in default rate between groups.",
    },
    "EDUCATION": {
        "group": "Demographics",
        "type": "Categorical",
        "description": "Education level: 1 = graduate school, 2 = university, 3 = high school, 4 = other.",
        "use": "Background variable; education level slightly changes how likely a customer is to default.",
    },
    "MARRIAGE": {
        "group": "Demographics",
        "type": "Categorical",
        "description": "Marital status: 1 = married, 2 = single, 3 = other.",
        "use": "Background variable; married and single customers show slightly different default rates.",
    },
    "AGE": {
        "group": "Demographics",
        "type": "Numeric (years)",
        "description": "Cardholder age in years.",
        "use": "A life-stage proxy; very young customers default slightly more often than middle-aged ones.",
    },
    "PAY_0": {
        "group": "Repayment status",
        "type": "Ordinal",
        "description": "Repayment status in the most recent month. -2 = no consumption, -1 = paid in full, 0 = revolving, 1+ = months late.",
        "use": "The single strongest predictor: a late payment last month dramatically raises default risk.",
    },
    "PAY_2": {
        "group": "Repayment status",
        "type": "Ordinal",
        "description": "Repayment status 2 months ago (same scale as PAY_0).",
        "use": "Recent payment history; late payments 2 months ago still raise today's default risk.",
    },
    "PAY_3": {
        "group": "Repayment status",
        "type": "Ordinal",
        "description": "Repayment status 3 months ago (same scale as PAY_0).",
        "use": "Older payment history that completes the 6-month behaviour trail.",
    },
    "PAY_4": {
        "group": "Repayment status",
        "type": "Ordinal",
        "description": "Repayment status 4 months ago (same scale as PAY_0).",
        "use": "Older payment history that completes the 6-month behaviour trail.",
    },
    "PAY_5": {
        "group": "Repayment status",
        "type": "Ordinal",
        "description": "Repayment status 5 months ago (same scale as PAY_0).",
        "use": "Older payment history that completes the 6-month behaviour trail.",
    },
    "PAY_6": {
        "group": "Repayment status",
        "type": "Ordinal",
        "description": "Repayment status 6 months ago (same scale as PAY_0).",
        "use": "The oldest month in the window; least influential but part of the full history.",
    },
    "BILL_AMT1": {
        "group": "Billed amount",
        "type": "Numeric (NT$)",
        "description": "Latest monthly statement balance (most recent month).",
        "use": "The debt currently outstanding; combined with the limit it gives credit utilization.",
    },
    "BILL_AMT2": {
        "group": "Billed amount",
        "type": "Numeric (NT$)",
        "description": "Statement balance 2 months ago.",
        "use": "Older debt level used to see whether the balance is stable or growing.",
    },
    "BILL_AMT3": {
        "group": "Billed amount",
        "type": "Numeric (NT$)",
        "description": "Statement balance 3 months ago.",
        "use": "Older debt level used to see whether the balance is stable or growing.",
    },
    "BILL_AMT4": {
        "group": "Billed amount",
        "type": "Numeric (NT$)",
        "description": "Statement balance 4 months ago.",
        "use": "Older debt level used to see whether the balance is stable or growing.",
    },
    "BILL_AMT5": {
        "group": "Billed amount",
        "type": "Numeric (NT$)",
        "description": "Statement balance 5 months ago.",
        "use": "Older debt level used to see whether the balance is stable or growing.",
    },
    "BILL_AMT6": {
        "group": "Billed amount",
        "type": "Numeric (NT$)",
        "description": "Statement balance 6 months ago.",
        "use": "The starting debt of the window; compared with BILL_AMT1 to get the debt trend.",
    },
    "PAY_AMT1": {
        "group": "Payment amount",
        "type": "Numeric (NT$)",
        "description": "Cash amount paid in the most recent month.",
        "use": "Real repayment effort: a high payment against a high bill means the customer pays statements off.",
    },
    "PAY_AMT2": {
        "group": "Payment amount",
        "type": "Numeric (NT$)",
        "description": "Cash amount paid 2 months ago.",
        "use": "Older repayment effort that helps separate occasional payers from consistent ones.",
    },
    "PAY_AMT3": {
        "group": "Payment amount",
        "type": "Numeric (NT$)",
        "description": "Cash amount paid 3 months ago.",
        "use": "Older repayment effort that helps separate occasional payers from consistent ones.",
    },
    "PAY_AMT4": {
        "group": "Payment amount",
        "type": "Numeric (NT$)",
        "description": "Cash amount paid 4 months ago.",
        "use": "Older repayment effort that helps separate occasional payers from consistent ones.",
    },
    "PAY_AMT5": {
        "group": "Payment amount",
        "type": "Numeric (NT$)",
        "description": "Cash amount paid 5 months ago.",
        "use": "Older repayment effort that helps separate occasional payers from consistent ones.",
    },
    "PAY_AMT6": {
        "group": "Payment amount",
        "type": "Numeric (NT$)",
        "description": "Cash amount paid 6 months ago.",
        "use": "Older repayment effort that helps separate occasional payers from consistent ones.",
    },
    TARGET: {
        "group": "Target",
        "type": "Binary",
        "description": "1 = the client defaults on the next payment; 0 = the client pays.",
        "use": "The answer the models learn to predict — this is what the whole project is about.",
    },
    "credit_utilization": {
        "group": "Behavioral ratio",
        "type": "Engineered",
        "description": "BILL_AMT1 / LIMIT_BAL — how much of the available credit is currently used.",
        "use": "Customers using most of their limit are much more likely to default.",
    },
    "avg_utilization_6m": {
        "group": "Behavioral ratio",
        "type": "Engineered",
        "description": "Average of BILL_AMT1..6 divided by LIMIT_BAL — persistent credit usage.",
        "use": "Catches customers who run maxed-out for months, not just in the latest statement.",
    },
    "late_months": {
        "group": "Behavioral ratio",
        "type": "Engineered",
        "description": "Number of the last 6 months with a payment delay (PAY status > 0).",
        "use": "Counts how often the customer slipped; more late months means higher risk.",
    },
    "max_delay_months": {
        "group": "Behavioral ratio",
        "type": "Engineered",
        "description": "Worst payment delay across the last 6 months.",
        "use": "A single long delay (e.g. 3+ months) is a serious warning sign even if other months were fine.",
    },
    "payment_to_bill_ratio": {
        "group": "Behavioral ratio",
        "type": "Engineered",
        "description": "PAY_AMT1 / (|BILL_AMT1| + 1) — is the customer paying off the statement?",
        "use": "Near-zero values mean the customer pays the minimum (or nothing) on a large bill — risky.",
    },
    "bill_trend_6m": {
        "group": "Behavioral ratio",
        "type": "Engineered",
        "description": "BILL_AMT1 - BILL_AMT6 — is the outstanding debt growing over time?",
        "use": "Rising debt over 6 months signals financial stress; falling debt signals recovery.",
    },
}

# Kept for compatibility: list of (attribute, type, description) tuples.
COLUMN_DICTIONARY = [
    (name, info["type"], info["description"]) for name, info in COLUMN_INFO.items()
]


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """Load the raw CSV from disk."""
    return pd.read_csv(DATA_FILE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning applied identically at train time and at prediction time.

    UCI contains undocumented EDUCATION values 0/5/6 and MARRIAGE value 0;
    they are folded into the documented "Other" bucket. The ID column is
    dropped and duplicate rows are removed.
    """
    out = df.copy()
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})
    if "ID" in out.columns:
        out = out.drop(columns=["ID"])
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into the final 29-column model matrix."""
    X = df[RAW_FEATURES].astype(float).copy()

    limit = X["LIMIT_BAL"].replace(0, np.nan)

    X["credit_utilization"] = (X["BILL_AMT1"] / limit).fillna(0.0)
    X["avg_utilization_6m"] = (X[BILL_COLS].mean(axis=1) / limit).fillna(0.0)
    X["late_months"] = (X[PAY_COLS] > 0).sum(axis=1).astype(float)
    X["max_delay_months"] = X[PAY_COLS].max(axis=1).astype(float)
    X["payment_to_bill_ratio"] = (
        X["PAY_AMT1"] / (X["BILL_AMT1"].abs() + 1.0)
    ).clip(0.0, 20.0)
    X["bill_trend_6m"] = (X["BILL_AMT1"] - X["BILL_AMT6"]).astype(float)

    return X[FEATURE_COLS]


def prepare_xy(df: pd.DataFrame):
    """Clean the data and return the model matrix + target."""
    df = clean(df)
    y = df[TARGET].astype(int)
    X = build_features(df)
    return X, y


def form_defaults(df: pd.DataFrame) -> dict:
    """Median/mode of every raw column — used as Streamlit widget defaults."""
    df = clean(df)
    defaults = {}
    for col in DEMO_COLS + BILL_COLS + PAYAMT_COLS:
        defaults[col] = float(df[col].median())
    for col in PAY_COLS:
        defaults[col] = int(df[col].mode().iloc[0])
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        defaults[col] = int(df[col].mode().iloc[0])
    defaults["AGE"] = int(defaults["AGE"])
    return defaults
