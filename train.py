"""
Train and tune the UCI Credit Card Default classifiers (NO PIPELINE).
=====================================================================
Workflow
--------
1. Load data/UCI_Credit_Card.csv
2. Clean + engineer the 6 behavioral features (shared src/preprocess.py)
3. Stratified 80/20 train-test split
4. Fit StandardScaler directly on X_train (used only by scale-sensitive models)
5. Tune 7 classifiers with GridSearchCV (3-fold stratified CV, accuracy)
6. Evaluate on the untouched test set (accuracy + precision / recall / F1)
7. Save every fitted model + scaler + metrics.json + eval_data.joblib

Run:
    python train.py
    python train.py --quick              # small grid, for a quick smoke test
    python train.py --output-dir models  # custom artifact folder
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocess import FEATURE_COLS, RAW_FEATURES, TARGET, load_data, prepare_xy

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------- #
# Model definitions: estimator, search grid, whether scaling is required
# --------------------------------------------------------------------------- #
MODEL_SPECS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "params": {
            "C": [0.1, 1.0, 10.0],
        },
        "needs_scaling": True,
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [15, 25, 35],
        },
        "needs_scaling": True,
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "params": {
            "max_depth": [4, 6, 8],
        },
        "needs_scaling": False,
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
        ),
        "params": {
            "max_depth": [8, 12],
        },
        "needs_scaling": False,
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100],
            "learning_rate": [0.05, 0.1],
        },
        "needs_scaling": False,
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            n_estimators=200,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "params": {
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        },
        "needs_scaling": False,
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100, 200],
        },
        "needs_scaling": False,
    },
}


def clean_value(value):
    """Make numpy / sklearn parameter values JSON-friendly."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def clean_dict(mapping):
    return {str(k): clean_value(v) for k, v in mapping.items()}


def limit_grid(grid):
    """Quick mode: keep the first value of every hyperparameter."""
    return {key: values[:1] for key, values in grid.items()}


def train_all(output_dir: Path, quick: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_folds = 2 if quick else 3

    print("=" * 76)
    print("UCI CREDIT CARD DEFAULT — TUNED CLASSIFIERS (NO PIPELINE)")
    print("=" * 76)

    # 1. Data ---------------------------------------------------------------- #
    raw = load_data()
    X, y = prepare_xy(raw)

    print(f"Raw data         : {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print(f"Model matrix     : {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Target           : {TARGET}")
    print(f"Default rate     : {y.mean():.2%}")
    print(f"Missing values   : {int(raw.isnull().sum().sum())}")

    # 2. Split --------------------------------------------------------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train / Test     : {len(X_train):,} / {len(X_test):,}")

    # 3. Scale directly (NO PIPELINE) --------------------------------------- #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, output_dir / "scaler.joblib")
    print(f"Saved scaler     : {output_dir / 'scaler.joblib'}")

    X_train_raw = X_train.values
    X_test_raw = X_test.values

    # 4. Tune every model ---------------------------------------------------- #
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    eval_data = {
        "y_test": y_test.tolist(),
        "probabilities": {},
        "predictions": {},
        "feature_names": list(FEATURE_COLS),
    }

    best_name = None
    best_acc = -1.0
    best_model = None
    best_needs_scale = False

    for name, spec in MODEL_SPECS.items():
        started = time.time()
        needs_scaling = spec["needs_scaling"]
        X_tr = X_train_scaled if needs_scaling else X_train_raw
        X_te = X_test_scaled if needs_scaling else X_test_raw
        grid = limit_grid(spec["params"]) if quick else spec["params"]

        print(f"\n{'-' * 76}")
        print(f"Training: {name}")
        print(f"Grid     : {grid}")

        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=grid,
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

        test_acc = float(accuracy_score(y_test, y_pred))
        train_acc = float(model.score(X_tr, y_train))
        cm = confusion_matrix(y_test, y_pred).tolist()

        safe_name = name.replace(" ", "_")
        model_file = f"{safe_name}.joblib"
        joblib.dump(model, output_dir / model_file)

        results[name] = {
            "model_file": model_file,
            "needs_scaling": needs_scaling,
            "best_params": clean_dict(search.best_params_),
            "cv_accuracy": float(search.best_score_),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": cm,
            "training_seconds": round(time.time() - started, 2),
        }

        eval_data["probabilities"][name] = [float(p) for p in y_prob]
        eval_data["predictions"][name] = [int(p) for p in y_pred]

        print(f"Best params : {search.best_params_}")
        print(f"CV accuracy : {search.best_score_:.4f}")
        print(f"Test acc    : {test_acc:.4f}")
        print(f"Precision   : {results[name]['test_precision']:.4f} | "
              f"Recall: {results[name]['test_recall']:.4f} | "
              f"F1: {results[name]['test_f1']:.4f}")
        print(f"Confusion   : {cm}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_name = name
            best_model = model
            best_needs_scale = needs_scaling

    # 5. Save best model + metadata ----------------------------------------- #
    assert best_model is not None and best_name is not None
    joblib.dump(best_model, output_dir / "best_model.joblib")
    joblib.dump(eval_data, output_dir / "eval_data.joblib")

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": "UCI Default of Credit Card Clients",
            "raw_rows": int(raw.shape[0]),
            "rows": int(len(X)),
            "raw_features": 23,
            "engineered_features": 6,
            "model_features": int(len(FEATURE_COLS)),
            "default_rate": float(y.mean()),
            "test_rows": int(len(X_test)),
        },
        "selection_metric": "test_accuracy",
        "best_model": best_name,
        "best_model_file": "best_model.joblib",
        "best_needs_scale": best_needs_scale,
        "feature_names": list(FEATURE_COLS),
        "raw_features": list(RAW_FEATURES),
        "results": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Small summary used by the optional notebook page / older scripts.
    notebook_summary = {
        "random_state": RANDOM_STATE,
        "rows": int(len(X)),
        "features": int(len(FEATURE_COLS)),
        "best_model": best_name,
        "selection_metric": "Accuracy",
        "models": {
            name: {
                "checked": "GridSearchCV · StratifiedKFold(%d) · Accuracy" % cv_folds,
                "best": json.dumps(info["best_params"]),
                "train_acc": info["train_accuracy"],
                "test_acc": info["test_accuracy"],
            }
            for name, info in results.items()
        },
    }
    (output_dir / "notebook_results.json").write_text(
        json.dumps(notebook_summary, indent=2), encoding="utf-8"
    )

    # Keep the app's dataset description beside the model artifacts too.
    description_file = DATA_DIR / "dataset_description.json"
    if description_file.exists():
        shutil.copyfile(description_file, output_dir / "dataset_description.json")

    # 6. Final leaderboard --------------------------------------------------- #
    leaderboard = pd.DataFrame(
        [
            {
                "Model": name,
                "Train accuracy": info["train_accuracy"],
                "Test accuracy": info["test_accuracy"],
                "Precision": info["test_precision"],
                "Recall": info["test_recall"],
                "F1": info["test_f1"],
            }
            for name, info in results.items()
        ]
    ).sort_values("Test accuracy", ascending=False)

    print(f"\n{'=' * 76}")
    print("FINAL LEADERBOARD")
    print(leaderboard.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBEST MODEL: {best_name} (test accuracy = {best_acc:.4f})")
    print(f"Artifacts saved in: {output_dir}")
    print("=" * 76)

    return metrics, leaderboard


def main():
    parser = argparse.ArgumentParser(description="Train UCI credit-card default models.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "models"),
        help="Folder for model weights and metrics.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny grid and 2 CV folds for a fast smoke test.",
    )
    args = parser.parse_args()
    train_all(Path(args.output_dir), quick=args.quick)


if __name__ == "__main__":
    main()
