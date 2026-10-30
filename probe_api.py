"""Probe sklearn 1.9 API details used by the notebook before generating it."""
import warnings

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

X = np.random.RandomState(0).randn(600, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

warnings.simplefilter("error")  # turn warnings into errors to catch deprecations


def probe(label, fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fn()
        print(f"OK    : {label}")
    except Exception as e:
        print(f"FAIL  : {label} -> {type(e).__name__}: {e}")


# --- Logistic regression: new l1_ratio API (penalty= is deprecated in 1.9) ---
probe("LR l1_ratio=0 (L2) saga",
      lambda: LogisticRegression(C=1.0, l1_ratio=0, solver="saga").fit(X, y))
probe("LR l1_ratio=1 (L1) saga",
      lambda: LogisticRegression(C=1.0, l1_ratio=1, solver="saga").fit(X, y))
probe("LR l1_ratio=0.5 (elasticnet) saga",
      lambda: LogisticRegression(C=1.0, l1_ratio=0.5, solver="saga").fit(X, y))
probe("LR l1_ratio=0 (L2) lbfgs",
      lambda: LogisticRegression(C=1.0, l1_ratio=0, solver="lbfgs").fit(X, y))
probe("LR l1_ratio=1 (L1) liblinear",
      lambda: LogisticRegression(C=1.0, l1_ratio=1, solver="liblinear").fit(X, y))
probe("LR default (no params)",
      lambda: LogisticRegression(max_iter=1000).fit(X, y))

# --- BaggingClassifier: base_estimator renamed? ---
probe("Bagging estimator=DT",
      lambda: BaggingClassifier(estimator=DecisionTreeClassifier(),
                                n_estimators=5, random_state=0).fit(X, y))
probe("Bagging base_estimator=DT (old name)",
      lambda: BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                n_estimators=5, random_state=0).fit(X, y))

# --- AdaBoost ---
probe("AdaBoost defaults",
      lambda: AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y))

# --- SVM ---
probe("LinearSVC dual=auto",
      lambda: LinearSVC(C=1.0, dual="auto", max_iter=5000).fit(X, y))
probe("SVC rbf predict_proba",
      lambda: SVC(kernel="rbf", probability=True, random_state=0)
      .fit(X, y).predict_proba(X))
print("LinearSVC has predict_proba:",
      hasattr(LinearSVC(C=1.0), "predict_proba"))
