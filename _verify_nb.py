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


# ===== CELL BREAK =====

df = pd.read_csv("data/UCI_Credit_Card.csv")


# ===== CELL BREAK =====

df.shape


# ===== CELL BREAK =====

df.head()


# ===== CELL BREAK =====

df.sample(5)


# ===== CELL BREAK =====

df.info()


# ===== CELL BREAK =====

df.describe()


# ===== CELL BREAK =====

df.isnull().sum()


# ===== CELL BREAK =====

# missing percentage per column (only columns which actually have missing values)
def missing_per(column):
    null_count = df[column].isnull().sum()
    return round((null_count / df.shape[0]) * 100, 2)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"{col} -> {missing_per(col)}%")

print("Total missing values:", df.isnull().sum().sum())


# ===== CELL BREAK =====

df.duplicated().sum()


# ===== CELL BREAK =====

print("Target value counts (0 = paid, 1 = default):")
print(df[TARGET].value_counts())
print("\nDefault rate:", round(df[TARGET].mean() * 100, 2), "%")


# ===== CELL BREAK =====

print("SEX (1 = male, 2 = female):")
print(df["SEX"].value_counts())

print("\nEDUCATION (1 = grad, 2 = university, 3 = high school, 4 = other):")
print(df["EDUCATION"].value_counts().sort_index())

print("\nMARRIAGE (1 = married, 2 = single, 3 = other):")
print(df["MARRIAGE"].value_counts().sort_index())


# ===== CELL BREAK =====

print("PAY_0 - repayment status last month:")
print(df["PAY_0"].value_counts().sort_index())


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

df = df.drop(columns=["ID"])


# ===== CELL BREAK =====

# Fix undocumented category codes
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})


# ===== CELL BREAK =====

print("Duplicates before:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicates after :", df.duplicated().sum())
print("Cleaned shape    :", df.shape)


# ===== CELL BREAK =====

print("Missing values:", df.isnull().sum().sum())
print("EDUCATION:", sorted(df["EDUCATION"].unique()))
print("MARRIAGE :", sorted(df["MARRIAGE"].unique()))
df.head()


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

# Do the engineered features separate payers from defaulters?
check = pd.DataFrame({
    "paid_mean": feature_df[df[TARGET] == 0][ENGINEERED_FEATURES].mean(),
    "default_mean": feature_df[df[TARGET] == 1][ENGINEERED_FEATURES].mean(),
})
check["difference"] = check["default_mean"] - check["paid_mean"]
check.round(3)


# ===== CELL BREAK =====

X = feature_df
y = df[TARGET].astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Default rate:", round(y.mean() * 100, 2), "%")


# ===== CELL BREAK =====

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RS, stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Train default rate:", round(y_train.mean() * 100, 2), "%")
print("Test  default rate:", round(y_test.mean() * 100, 2), "%")


# ===== CELL BREAK =====

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled train shape:", X_train_scaled.shape)
print("Scaled test shape :", X_test_scaled.shape)
X_train_scaled[:3]


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

lr_search = tune_and_evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=2000, random_state=RS),
    {
        "C": [0.1, 1.0, 10.0],
    },
    needs_scaling=True,
)
lr_search.best_params_


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

knn_search = tune_and_evaluate(
    "KNN",
    KNeighborsClassifier(),
    {
        "n_neighbors": [15, 25, 35],
    },
    needs_scaling=True,
)
knn_search.best_params_


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

dt_search = tune_and_evaluate(
    "Decision Tree",
    DecisionTreeClassifier(random_state=RS),
    {
        "max_depth": [4, 6, 8],
    },
    needs_scaling=False,
)
dt_search.best_params_


# ===== CELL BREAK =====

rf_search = tune_and_evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=1),
    {
        "max_depth": [8, 12],
    },
    needs_scaling=False,
)
rf_search.best_params_


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

ada_search = tune_and_evaluate(
    "AdaBoost",
    AdaBoostClassifier(random_state=RS),
    {
        "n_estimators": [100, 200],
    },
    needs_scaling=False,
)
ada_search.best_params_


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

plot_df = comparison.set_index("Model")[["Test accuracy", "F1"]]
plot_df.plot(kind="bar", figsize=(11, 4), color=["#4f46e5", "#16a34a"])
plt.title("Tuned model comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()


# ===== CELL BREAK =====

best_name = comparison.iloc[0]["Model"]
print("Best model (by test accuracy):", best_name)
print(comparison.iloc[0][["Test accuracy", "Precision", "Recall", "F1"]].to_string())


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

# Evaluation data for charts / reproducibility
eval_data = {
    "y_test": y_test.tolist(),
    "probabilities": {name: prob.tolist() for name, prob in test_probabilities.items()},
    "predictions": {name: pred.tolist() for name, pred in test_predictions.items()},
    "feature_names": FEATURES,
}
joblib.dump(eval_data, "models/eval_data.joblib")
print("saved -> models/eval_data.joblib")


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

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


# ===== CELL BREAK =====

# Derived indicators for this profile
print("Credit utilization :", round(float(profile_features["credit_utilization"].iloc[0]) * 100, 1), "%")
print("6-month utilization:", round(float(profile_features["avg_utilization_6m"].iloc[0]) * 100, 1), "%")
print("Late months        :", int(profile_features["late_months"].iloc[0]), "/ 6")
print("Worst delay        :", int(profile_features["max_delay_months"].iloc[0]), "months")
