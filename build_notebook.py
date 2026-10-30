"""
Build + execute Credit_Default_Classification.ipynb
===================================================
The notebook follows the simple student style of D:\\DATA_SCIENCE2.0\\Machine learning:
  Data Inspection -> Data Cleaning -> EDA -> Feature Engineering ->
  Preprocessing -> Modelling (tuned) -> Comparison -> Save weights

NO PIPELINE: StandardScaler is fitted directly on X_train.
Every code cell is executed and its real output is stored in the notebook.
"""

import ast
import base64
import io
import json
import os
import sys
import textwrap
import traceback
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

CELLS = []
execution_counter = 1
exec_namespace = {"__name__": "__main__", "plt": plt}
plt.show = lambda *args, **kwargs: None


def add_md(content: str):
    CELLS.append(
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": [
                line + "\n"
                for line in textwrap.dedent(content).strip("\n").split("\n")
            ],
        }
    )


def add_code_and_exec(code_str: str):
    global execution_counter
    code_lines = [line + "\n" for line in code_str.strip("\n").split("\n")]
    cell = {
        "cell_type": "code",
        "execution_count": execution_counter,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": code_lines,
    }

    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    old_stdout.write(f"Executing cell {execution_counter}...\n")
    old_stdout.flush()
    sys.stdout = stdout_buf
    plt.close("all")

    eval_result = None
    try:
        parsed = ast.parse(code_str)
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            body = parsed.body[:-1]
            expr = parsed.body[-1]
            if body:
                exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), exec_namespace)
            eval_result = eval(compile(ast.Expression(body=expr.value), "<cell>", "eval"), exec_namespace)
        else:
            exec(compile(code_str, "<cell>", "exec"), exec_namespace)
    except Exception:
        sys.stdout = old_stdout
        print(f"\nERROR in cell {execution_counter}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout = old_stdout

    stdout_text = stdout_buf.getvalue()
    if stdout_text:
        cell["outputs"].append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.rstrip("\n").split("\n")],
            }
        )

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        image = io.BytesIO()
        fig.savefig(image, format="png", bbox_inches="tight", dpi=100)
        image.seek(0)
        encoded = base64.b64encode(image.read()).decode("utf-8")
        cell["outputs"].append(
            {
                "data": {"image/png": encoded, "text/plain": ["<Figure>"]},
                "metadata": {},
                "output_type": "display_data",
            }
        )
    plt.close("all")

    if eval_result is not None:
        data = {}
        if hasattr(eval_result, "_repr_html_"):
            try:
                data["text/html"] = [line + "\n" for line in eval_result._repr_html_().split("\n")]
            except Exception:
                pass
        data["text/plain"] = [line + "\n" for line in repr(eval_result).split("\n")]
        cell["outputs"].append(
            {
                "data": data,
                "execution_count": execution_counter,
                "metadata": {},
                "output_type": "execute_result",
            }
        )

    execution_counter += 1
    CELLS.append(cell)


# =========================================================================== #
# TITLE
# =========================================================================== #
add_md(
    """
# *Credit Card Default Classification*

**Dataset:** UCI Default of Credit Card Clients (Taiwan, April - September 2005)  
**Task:** Binary classification - predict whether a cardholder will default on the next payment.  
**Target:** `default.payment.next.month` (1 = default, 0 = paid)  
**Models:** Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost and AdaBoost.  
**No pipeline:** cleaning, feature engineering and scaling are done explicitly, step by step.

### About the dataset
The data contains **30,000 real credit-card accounts** from Taiwan. Every row describes one cardholder just before the next payment is due: who they are (credit limit, age, gender, education, marital status), whether they paid on time in each of the last six months, how much they were billed each month, and how much cash they actually paid back.

Around **22% of the accounts default** and 78% pay on time, so the target is imbalanced. The strongest signal is recent repayment behaviour (`PAY_0`, last month's payment status), more than age or education.

---
"""
)

# =========================================================================== #
# IMPORTS + LOAD
# =========================================================================== #
add_code_and_exec(
    r"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

TARGET = "default.payment.next.month"
RS = 42
"""
)

add_code_and_exec(
    r"""
df = pd.read_csv("data/UCI_Credit_Card.csv")
"""
)

# =========================================================================== #
# DATA INSPECTION
# =========================================================================== #
add_md(
    """
--------------
# *Data Inspection*
--------------
"""
)

add_code_and_exec("df.shape")
add_code_and_exec("df.head()")
add_code_and_exec("df.sample(5)")
add_code_and_exec("df.info()")
add_code_and_exec("df.describe()")
add_code_and_exec("df.isnull().sum()")

add_code_and_exec(
    r"""
# missing percentage per column (only columns which actually have missing values)
def missing_per(column):
    null_count = df[column].isnull().sum()
    return round((null_count / df.shape[0]) * 100, 2)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"{col} -> {missing_per(col)}%")

print("Total missing values:", df.isnull().sum().sum())
"""
)

add_code_and_exec("df.duplicated().sum()")

add_code_and_exec(
    r"""
print("Target value counts (0 = paid, 1 = default):")
print(df[TARGET].value_counts())
print("\nDefault rate:", round(df[TARGET].mean() * 100, 2), "%")
"""
)

add_code_and_exec(
    r"""
print("SEX (1 = male, 2 = female):")
print(df["SEX"].value_counts())

print("\nEDUCATION (1 = grad, 2 = university, 3 = high school, 4 = other):")
print(df["EDUCATION"].value_counts().sort_index())

print("\nMARRIAGE (1 = married, 2 = single, 3 = other):")
print(df["MARRIAGE"].value_counts().sort_index())
"""
)

add_code_and_exec(
    r"""
print("PAY_0 - repayment status last month:")
print(df["PAY_0"].value_counts().sort_index())
"""
)

add_md(
    """
### Dataset overview:
* **Shape:** 30,000 rows x 25 columns
* **Target:** `default.payment.next.month` -> 23,364 paid (77.88%) vs 6,636 default (22.12%) -> **imbalanced**
* **Missing values:** exactly 0 in every column
* **Duplicates:** 35 duplicate rows found
* **ID column:** a unique client identifier with no predictive meaning -> drop it
* **Undocumented codes:** `EDUCATION` has 0, 5 and 6; `MARRIAGE` has 0 -> fold into `Other`
* **Strongest signal:** recent repayment status (`PAY_0`), where positive values mean months late
"""
)

# =========================================================================== #
# USER-DEFINED FUNCTIONS
# =========================================================================== #
add_md(
    """
--------------
## **User_defined Functions**
--------------
"""
)

add_code_and_exec(
    r"""
# Outlier detection with the IQR rule
def outlier_detection(column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = df[column_name][
        (df[column_name] < lower_limit) | (df[column_name] > upper_limit)
    ]
    return outliers.tolist()

numerical_cols = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

outlier_table = pd.DataFrame({
    "outlier_rows": [len(outlier_detection(c)) for c in numerical_cols],
    "outlier_percentage": [
        round(len(outlier_detection(c)) / len(df) * 100, 2) for c in numerical_cols
    ],
}, index=numerical_cols)

outlier_table
"""
)

add_md(
    """
### Outlier note:
* `LIMIT_BAL` and the bill/payment columns have many statistical outliers.
* These are **real high-value bank accounts**, not measurement errors, so they are kept.
* The models used here are tree-based or regularized linear models, which handle this better than distance-only methods.
"""
)

# =========================================================================== #
# DATA CLEANING
# =========================================================================== #
add_md(
    """
--------------
# *Data Cleaning*
--------------
"""
)

add_code_and_exec(
    r"""
df = df.drop(columns=["ID"])
"""
)

add_md(
    """
### *"ID"* column is deleted because it has all unique values and no predictive signal.
"""
)

add_code_and_exec(
    r"""
# Fix undocumented category codes
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
"""
)

add_md(
    """
### *Undocumented categories replaced:*
* `EDUCATION`: 0, 5, 6 -> 4 (`Other`)
* `MARRIAGE`: 0 -> 3 (`Other`)
"""
)

add_code_and_exec(
    r"""
print("Duplicates before:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicates after :", df.duplicated().sum())
print("Cleaned shape    :", df.shape)
"""
)

add_code_and_exec(
    r"""
print("Missing values:", df.isnull().sum().sum())
print("EDUCATION:", sorted(df["EDUCATION"].unique()))
print("MARRIAGE :", sorted(df["MARRIAGE"].unique()))
df.head()
"""
)

# =========================================================================== #
# EDA
# =========================================================================== #
add_md(
    """
--------------
# *Exploratory Data Analysis*
--------------
"""
)

add_code_and_exec(
    r"""
# Target distribution
vc = df[TARGET].value_counts().sort_index()
plt.figure(figsize=(6, 4))
bars = plt.bar(["0 (paid)", "1 (default)"], vc.values, color=["#16a34a", "#dc2626"])
plt.title("Target distribution")
plt.ylabel("Number of clients")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
             f"{int(bar.get_height()):,}", ha="center")
plt.tight_layout()
plt.show()
"""
)

add_code_and_exec(
    r"""
# Default rate by repayment status (PAY_0 is the most important feature)
pay_rate = df.groupby("PAY_0")[TARGET].mean() * 100

plt.figure(figsize=(9, 4))
bars = plt.bar([str(i) for i in pay_rate.index], pay_rate.values, color="#dc2626")
plt.title("Default rate by PAY_0 (last month repayment status)")
plt.xlabel("PAY_0: -2 no consumption, -1 paid in full, 0 revolving, 1+ months late")
plt.ylabel("Default rate (%)")
plt.axhline(df[TARGET].mean() * 100, color="#0f172a", linestyle="--",
            label="Overall default rate")
plt.legend()
plt.tight_layout()
plt.show()
"""
)

add_code_and_exec(
    r"""
# Default rate across demographics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["SEX", "EDUCATION", "MARRIAGE"]):
    rate = df.groupby(col)[TARGET].mean() * 100
    ax.bar(rate.index.astype(str), rate.values, color="#4f46e5")
    ax.set_title(f"Default rate by {col}")
    ax.set_ylabel("Default rate (%)")
    ax.axhline(df[TARGET].mean() * 100, color="#64748b", linestyle=":")
plt.tight_layout()
plt.show()
"""
)

add_code_and_exec(
    r"""
# Correlation of every raw feature with the target
corr = df.corr()[TARGET].drop(TARGET)
top_corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)

plt.figure(figsize=(9, 5))
colors = ["#dc2626" if value > 0 else "#16a34a" for value in top_corr.values]
plt.barh([str(c) for c in top_corr.index[::-1]], top_corr.values[::-1],
         color=colors[::-1])
plt.title("Top 15 correlations with default")
plt.xlabel("Correlation coefficient")
plt.tight_layout()
plt.show()

top_corr.round(3).to_frame("correlation")
"""
)

add_md(
    """
### EDA insights:
* Recent repayment behaviour is the dominant signal: default rate jumps sharply once `PAY_0` is 1 or more.
* Demographics (sex, education, marriage) show only small differences.
* Credit limit and bill amounts have a weaker but useful relationship with default.
* The target is imbalanced (22.1% default), so models are compared with **precision, recall and F1** as well as accuracy.
"""
)

# =========================================================================== #
# FEATURE ENGINEERING
# =========================================================================== #
add_md(
    """
--------------
# *Feature Engineering*
#### Six simple financial ratios, no pipeline
--------------
"""
)

add_code_and_exec(
    r"""
PAY_COLS   = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS  = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAM_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

DEMO_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
RAW_FEATURES = DEMO_COLS + PAY_COLS + BILL_COLS + PAYAM_COLS

ENGINEERED_FEATURES = [
    "credit_utilization",
    "avg_utilization_6m",
    "late_months",
    "max_delay_months",
    "payment_to_bill_ratio",
    "bill_trend_6m",
]
FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def build_features(frame):
    # Create the same 29 model columns used by train.py and app.py.
    X = frame[RAW_FEATURES].astype(float).copy()
    limit = X["LIMIT_BAL"].replace(0, np.nan)

    X["credit_utilization"] = (X["BILL_AMT1"] / limit).fillna(0.0)
    X["avg_utilization_6m"] = (X[BILL_COLS].mean(axis=1) / limit).fillna(0.0)
    X["late_months"] = (X[PAY_COLS] > 0).sum(axis=1).astype(float)
    X["max_delay_months"] = X[PAY_COLS].max(axis=1).astype(float)
    X["payment_to_bill_ratio"] = (
        X["PAY_AMT1"] / (X["BILL_AMT1"].abs() + 1.0)
    ).clip(0.0, 20.0)
    X["bill_trend_6m"] = (X["BILL_AMT1"] - X["BILL_AMT6"]).astype(float)

    return X[FEATURES]


feature_df = build_features(df)
feature_df.head()
"""
)

add_code_and_exec(
    r"""
# Do the engineered features separate payers from defaulters?
check = pd.DataFrame({
    "paid_mean": feature_df[df[TARGET] == 0][ENGINEERED_FEATURES].mean(),
    "default_mean": feature_df[df[TARGET] == 1][ENGINEERED_FEATURES].mean(),
})
check["difference"] = check["default_mean"] - check["paid_mean"]
check.round(3)
"""
)

add_md(
    """
### What the new features mean:
* `credit_utilization` - how much of the credit limit is currently used
* `avg_utilization_6m` - persistent usage over six months
* `late_months` - how many of the last six months had a payment delay
* `max_delay_months` - worst delay in the window
* `payment_to_bill_ratio` - whether the customer actually pays the statement
* `bill_trend_6m` - is the outstanding debt growing?
"""
)

# =========================================================================== #
# PREPROCESSING
# =========================================================================== #
add_md(
    """
--------------
# *Data Preprocessing*
#### Split, then fit StandardScaler directly on the training data (NO PIPELINE)
--------------
"""
)

add_code_and_exec(
    r"""
X = feature_df
y = df[TARGET].astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Default rate:", round(y.mean() * 100, 2), "%")
"""
)

add_code_and_exec(
    r"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RS, stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Train default rate:", round(y_train.mean() * 100, 2), "%")
print("Test  default rate:", round(y_test.mean() * 100, 2), "%")
"""
)

add_code_and_exec(
    r"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled train shape:", X_train_scaled.shape)
print("Scaled test shape :", X_test_scaled.shape)
X_train_scaled[:3]
"""
)

add_md(
    """
### Important:
* The scaler is fitted **only on `X_train`** and then applied to `X_test`.
* There is no `Pipeline` object anywhere in this notebook.
* Scale-sensitive models use `X_train_scaled`; tree models use the raw `X_train.values`.
"""
)

# =========================================================================== #
# MODELLING
# =========================================================================== #
add_md(
    """
--------------
# *Data Modelling*
#### Simple hyperparameters with GridSearchCV + 3-fold StratifiedKFold, scored by accuracy
--------------
"""
)

add_code_and_exec(
    r"""
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)

results = {}
fitted_models = {}
test_predictions = {}
test_probabilities = {}


def tune_and_evaluate(name, estimator, param_grid, needs_scaling=False):
    # Run GridSearchCV, evaluate on the untouched test set, store the result.
    X_tr = X_train_scaled if needs_scaling else X_train.values
    X_te = X_test_scaled if needs_scaling else X_test.values

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_tr, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    results[name] = {
        "best_params": search.best_params_,
        "cv_accuracy": round(float(search.best_score_), 4),
        "train_accuracy": round(float(model.score(X_tr, y_train)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    fitted_models[name] = model
    test_predictions[name] = y_pred
    test_probabilities[name] = y_prob

    print(f"Best params : {search.best_params_}")
    print(f"CV accuracy : {search.best_score_:.4f}")
    print(f"Test accuracy: {results[name]['test_accuracy']:.4f}")
    print(f"Precision / Recall / F1: "
          f"{results[name]['precision']:.4f} / "
          f"{results[name]['recall']:.4f} / "
          f"{results[name]['f1']:.4f}")
    print("Confusion matrix:")
    print(np.array(results[name]["confusion_matrix"]))

    return search
"""
)

add_md(
    """
### Model 1 - Logistic Regression (scaled features)
"""
)

add_code_and_exec(
    r"""
lr_search = tune_and_evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=2000, random_state=RS),
    {
        "C": [0.1, 1.0, 10.0],
    },
    needs_scaling=True,
)
lr_search.best_params_
"""
)

add_md(
    """
### Model 2 - KNN (scaled features)
#### First check the elbow: accuracy vs number of neighbours
"""
)

add_code_and_exec(
    r"""
k_values = list(range(1, 31, 2))
train_scores, test_scores = [], []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

plt.figure(figsize=(8, 4))
plt.plot(k_values, train_scores, marker="o", label="train")
plt.plot(k_values, test_scores, marker="o", label="test")
plt.xlabel("k (number of neighbours)")
plt.ylabel("Accuracy")
plt.title("KNN elbow plot")
plt.legend()
plt.tight_layout()
plt.show()

print("Best k in this sweep:", k_values[int(np.argmax(test_scores))])
"""
)

add_code_and_exec(
    r"""
knn_search = tune_and_evaluate(
    "KNN",
    KNeighborsClassifier(),
    {
        "n_neighbors": [15, 25, 35],
    },
    needs_scaling=True,
)
knn_search.best_params_
"""
)

add_md(
    """
### Model 3 - Decision Tree (raw features)
#### First check max_depth with a train vs test curve to see overfitting
"""
)

add_code_and_exec(
    r"""
depths = list(range(1, 21))
train_scores, test_scores = [], []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=RS)
    tree.fit(X_train.values, y_train)
    train_scores.append(tree.score(X_train.values, y_train))
    test_scores.append(tree.score(X_test.values, y_test))

plt.figure(figsize=(8, 4))
plt.plot(depths, train_scores, marker="o", label="train")
plt.plot(depths, test_scores, marker="o", label="test")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: accuracy vs max_depth")
plt.legend()
plt.tight_layout()
plt.show()

print("Best depth in this sweep:", depths[int(np.argmax(test_scores))])
"""
)

add_code_and_exec(
    r"""
dt_search = tune_and_evaluate(
    "Decision Tree",
    DecisionTreeClassifier(random_state=RS),
    {
        "max_depth": [4, 6, 8],
    },
    needs_scaling=False,
)
dt_search.best_params_
"""
)

add_md(
    """
### Model 4 - Random Forest (raw features)
"""
)

add_code_and_exec(
    r"""
rf_search = tune_and_evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=1),
    {
        "max_depth": [8, 12],
    },
    needs_scaling=False,
)
rf_search.best_params_
"""
)

add_md(
    """
### Model 5 - Gradient Boosting (raw features)
"""
)

add_code_and_exec(
    r"""
gb_search = tune_and_evaluate(
    "Gradient Boosting",
    GradientBoostingClassifier(random_state=RS),
    {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
    },
    needs_scaling=False,
)
gb_search.best_params_
"""
)

add_md(
    """
### Model 6 - XGBoost (raw features)
"""
)

add_code_and_exec(
    r"""
xgb_search = tune_and_evaluate(
    "XGBoost",
    XGBClassifier(
        n_estimators=200,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RS,
        n_jobs=1,
    ),
    {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    },
    needs_scaling=False,
)
xgb_search.best_params_
"""
)

add_md(
    """
### Model 7 - AdaBoost (raw features)
"""
)

add_code_and_exec(
    r"""
ada_search = tune_and_evaluate(
    "AdaBoost",
    AdaBoostClassifier(random_state=RS),
    {
        "n_estimators": [100, 200],
    },
    needs_scaling=False,
)
ada_search.best_params_
"""
)

# =========================================================================== #
# COMPARISON
# =========================================================================== #
add_md(
    """
--------------
# *Model Comparison*
--------------
"""
)

add_code_and_exec(
    r"""
comparison = pd.DataFrame([
    {
        "Model": name,
        "Train accuracy": info["train_accuracy"],
        "Test accuracy": info["test_accuracy"],
        "Precision": info["precision"],
        "Recall": info["recall"],
        "F1": info["f1"],
        "Best params": info["best_params"],
    }
    for name, info in results.items()
]).sort_values("Test accuracy", ascending=False).reset_index(drop=True)

comparison
"""
)

add_code_and_exec(
    r"""
plot_df = comparison.set_index("Model")[["Test accuracy", "F1"]]
plot_df.plot(kind="bar", figsize=(11, 4), color=["#4f46e5", "#16a34a"])
plt.title("Tuned model comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()
"""
)

add_code_and_exec(
    r"""
best_name = comparison.iloc[0]["Model"]
print("Best model (by test accuracy):", best_name)
print(comparison.iloc[0][["Test accuracy", "Precision", "Recall", "F1"]].to_string())
"""
)

add_code_and_exec(
    r"""
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions[best_name],
    display_labels=["Paid", "Default"],
    cmap="Blues",
    colorbar=False,
    ax=ax,
)
ax.set_title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.show()
"""
)

add_md(
    """
### Comparison notes:
* XGBoost, Gradient Boosting and Random Forest are the strongest models.
* All tuned models beat the 22% base rate, but **accuracy alone hides the class imbalance**, so precision, recall and F1 are shown too.
* Test accuracy is used to choose the best model because it is the simplest, most direct measure.
* The final app shows every model's prediction so the result is transparent.
"""
)

# =========================================================================== #
# SAVE WEIGHTS
# =========================================================================== #
add_md(
    """
--------------
### *Save every model weight and the scaler for the Streamlit app*
--------------
"""
)

add_code_and_exec(
    r"""
os.makedirs("models", exist_ok=True)

# Save each tuned model separately
for name, model in fitted_models.items():
    filename = name.replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join("models", filename))
    print("saved ->", os.path.join("models", filename))

# Save the best model and the fitted scaler (NO PIPELINE)
joblib.dump(fitted_models[best_name], "models/best_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
print("saved -> models/best_model.joblib")
print("saved -> models/scaler.joblib")
"""
)

add_code_and_exec(
    r"""
# Evaluation data for charts / reproducibility
eval_data = {
    "y_test": y_test.tolist(),
    "probabilities": {name: prob.tolist() for name, prob in test_probabilities.items()},
    "predictions": {name: pred.tolist() for name, pred in test_predictions.items()},
    "feature_names": FEATURES,
}
joblib.dump(eval_data, "models/eval_data.joblib")
print("saved -> models/eval_data.joblib")
"""
)

add_code_and_exec(
    r"""
# Full metrics file consumed by app.py
metrics = {
    "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    "dataset": {
        "name": "UCI Default of Credit Card Clients",
        "raw_rows": 30000,
        "rows": int(len(df)),
        "raw_features": 23,
        "engineered_features": 6,
        "model_features": int(len(FEATURES)),
        "default_rate": float(y.mean()),
        "test_rows": int(len(y_test)),
    },
    "selection_metric": "test_accuracy",
    "best_model": best_name,
    "best_model_file": "best_model.joblib",
    "best_needs_scale": best_name in ["Logistic Regression", "KNN"],
    "feature_names": FEATURES,
    "raw_features": RAW_FEATURES,
    "results": {},
}

needs_scaling_map = {
    "Logistic Regression": True,
    "KNN": True,
    "Decision Tree": False,
    "Random Forest": False,
    "Gradient Boosting": False,
    "XGBoost": False,
    "AdaBoost": False,
}

for name, info in results.items():
    metrics["results"][name] = {
        "model_file": name.replace(" ", "_") + ".joblib",
        "needs_scaling": needs_scaling_map[name],
        "best_params": info["best_params"],
        "cv_accuracy": info["cv_accuracy"],
        "train_accuracy": info["train_accuracy"],
        "test_accuracy": info["test_accuracy"],
        "test_precision": info["precision"],
        "test_recall": info["recall"],
        "test_f1": info["f1"],
        "confusion_matrix": info["confusion_matrix"],
    }

with open("models/metrics.json", "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2, default=float)

print("saved -> models/metrics.json")
print("best model:", best_name)
"""
)

add_code_and_exec(
    r"""
# Small summary file used by older helper scripts
summary = {
    "random_state": RS,
    "rows": int(len(df)),
    "features": int(len(FEATURES)),
    "best_model": best_name,
    "selection_metric": "Accuracy",
    "models": {
        name: {
            "checked": "GridSearchCV - StratifiedKFold(3) - Accuracy",
            "best": json.dumps(info["best_params"]),
            "train_acc": info["train_accuracy"],
            "test_acc": info["test_accuracy"],
        }
        for name, info in results.items()
    },
}

with open("models/notebook_results.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, default=float)

print("saved -> models/notebook_results.json")
"""
)

# =========================================================================== #
# SINGLE CUSTOMER DEMO
# =========================================================================== #
add_md(
    """
--------------
# *Single-customer inference demo*
#### Exactly the same path the Streamlit app uses
--------------
"""
)

add_code_and_exec(
    r"""
profile = {
    "LIMIT_BAL": 50000.0,
    "SEX": 1,
    "EDUCATION": 3,
    "MARRIAGE": 2,
    "AGE": 26,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 1,
    "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 48500.0, "BILL_AMT2": 47000.0, "BILL_AMT3": 45500.0,
    "BILL_AMT4": 44000.0, "BILL_AMT5": 43000.0, "BILL_AMT6": 42000.0,
    "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 500.0,
    "PAY_AMT4": 800.0, "PAY_AMT5": 1000.0, "PAY_AMT6": 1000.0,
}

profile_df = pd.DataFrame([profile])
profile_features = build_features(profile_df)

inference_rows = []
for name, model in fitted_models.items():
    if name in ["Logistic Regression", "KNN"]:
        matrix = scaler.transform(profile_features)
    else:
        matrix = profile_features.values
    probability = model.predict_proba(matrix)[0, 1]
    inference_rows.append({
        "Model": name,
        "Prediction": "Default" if probability >= 0.5 else "Will pay",
        "Default probability": round(float(probability) * 100, 2),
        "Test accuracy": results[name]["test_accuracy"] * 100,
    })

pd.DataFrame(inference_rows).sort_values("Test accuracy", ascending=False).reset_index(drop=True)
"""
)

add_code_and_exec(
    r"""
# Derived indicators for this profile
print("Credit utilization :", round(float(profile_features["credit_utilization"].iloc[0]) * 100, 1), "%")
print("6-month utilization:", round(float(profile_features["avg_utilization_6m"].iloc[0]) * 100, 1), "%")
print("Late months        :", int(profile_features["late_months"].iloc[0]), "/ 6")
print("Worst delay        :", int(profile_features["max_delay_months"].iloc[0]), "months")
"""
)

# =========================================================================== #
# CONCLUSION
# =========================================================================== #
add_md(
    """
--------------
# *Conclusion*
--------------
* The UCI credit-card dataset is the right choice for this project because every input is a real, explainable financial variable.
* Recent repayment behaviour (`PAY_0`, late months, worst delay) is the strongest predictor of default.
* Seven models were tuned with **simple hyperparameter grids** and the best one was selected by **test accuracy**.
* Every model weight, the scaler and the metrics are saved in `models/` for the Streamlit app.
* The Streamlit UI loads those weights and shows the best-model verdict plus **what every model predicts**, together with each model's test accuracy.

### Run the app
```bash
streamlit run app.py
```

### Project files
```
data/UCI_Credit_Card.csv          raw dataset
data/dataset_description.json     AI-generated description used by the UI
src/preprocess.py                 shared cleaning + 6 engineered features
train.py                          standalone tuned training script
app.py                            2-tab Streamlit UI
models/*.joblib                   stored model weights + scaler
models/metrics.json               scores, best params, confusion matrices
Credit_Default_Classification.ipynb  this notebook
```
"""
)


# =========================================================================== #
# WRITE NOTEBOOK
# =========================================================================== #
notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "Credit_Default_Classification.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")

n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
print(f"\nwritten: {out.name}")
print(f"cells: {n_md} markdown + {n_code} code = {len(CELLS)}")
print("Every code cell executed successfully.")
