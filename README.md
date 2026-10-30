# 💳 Credit Card Default Prediction

A clean end-to-end machine-learning project on the **UCI Default of Credit Card Clients** dataset:
30,000 real credit-card accounts, 23 readable financial features, and a binary target
(`default.payment.next.month`).

The project contains:

- a **step-by-step notebook** with real executed outputs,
- a **standalone training script** that tunes seven classifiers with simple hyperparameter grids
  and saves their weights,
- a **two-page Streamlit UI** with a dark sidebar that explains the dataset and scores a customer
  profile with every stored model.

---

## 📊 About the dataset

**UCI Default of Credit Card Clients** — 30,000 credit-card accounts from Taiwan, observed between
April and September 2005. Each row describes one cardholder just before the next payment is due:
who they are (credit limit, age, gender, education, marital status), whether they paid on time in
each of the last six months, how much they were billed each month, and how much cash they actually
paid back.

- **Target:** `default.payment.next.month` (1 = default, 0 = paid)
- **Split:** 22.1% default vs 77.9% paid (imbalanced target)
- **Strongest signal:** recent repayment status `PAY_0`
- Every input is a real, explainable financial variable — which is exactly why it suits an
  interactive prediction form.

---

## 📁 Project structure

```
.
├── data/
│   ├── UCI_Credit_Card.csv              # raw dataset (30,000 × 25)
│   └── dataset_description.json         # plain-language dataset explanation used by the UI
├── src/
│   ├── __init__.py
│   └── preprocess.py                    # shared cleaning + 6 engineered features
├── models/
│   ├── Logistic_Regression.joblib       # tuned model weights
│   ├── KNN.joblib
│   ├── Decision_Tree.joblib
│   ├── Random_Forest.joblib
│   ├── Gradient_Boosting.joblib
│   ├── XGBoost.joblib
│   ├── AdaBoost.joblib
│   ├── best_model.joblib                # best model by test accuracy
│   ├── scaler.joblib                    # fitted StandardScaler
│   ├── metrics.json                     # scores, best params, confusion matrices
│   ├── eval_data.joblib                 # test predictions / probabilities
│   └── notebook_results.json            # compact notebook summary
├── Credit_Default_Classification.ipynb  # main notebook (NO PIPELINE)
├── train.py                             # tuned training script (NO PIPELINE)
├── app.py                               # Streamlit UI
├── build_notebook.py                    # rebuilds the executed notebook
├── requirements.txt                     # pinned deployment dependencies
├── runtime.txt                          # Python version for Streamlit Cloud
├── .streamlit/config.toml               # dark theme + headless server
└── README.md
```

> `Credit_Default_Classification_Pipeline.ipynb` is an older experiment and is **not** used by the
> current app. The main notebook is `Credit_Default_Classification.ipynb`.

---

## 📓 Notebook — `Credit_Default_Classification.ipynb`

The notebook follows a simple, practical, step-by-step style:

1. **Data Inspection** — shape, head, sample, info, describe, missing values, duplicates, target counts.
2. **User-defined functions** — missing-percentage and IQR outlier detection.
3. **Data Cleaning** — drop `ID`, fix undocumented `EDUCATION` / `MARRIAGE` codes, remove duplicates.
4. **Exploratory Data Analysis** — target distribution, default rate by `PAY_0`, demographics, correlations.
5. **Feature Engineering** — six financial ratios: utilization, average utilization, late months, worst delay, payment-to-bill ratio, bill trend.
6. **Data Preprocessing** — stratified 80/20 split, `StandardScaler` fitted directly on `X_train` (**no Pipeline**).
7. **Modelling** — seven classifiers tuned with `GridSearchCV` + 3-fold `StratifiedKFold`, scored by
   **accuracy** with simple grids:
   Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, AdaBoost.
8. **Model Comparison** — leaderboard and chart by accuracy / precision / recall / F1, plus one
   confusion matrix for the best model.
9. **Save Weights** — every `.joblib` model, `scaler.joblib`, `metrics.json`, `eval_data.joblib`.
10. **Single-customer inference demo** — exactly the path the Streamlit app uses.
11. **Conclusion** — key insights and project notes.

Run it with:

```bash
jupyter notebook Credit_Default_Classification.ipynb
```

---

## 🖥️ Streamlit UI — `app.py`

A focused dark interface with two large navigation buttons in the left sidebar:

### 📊 Dataset overview
- A concise plain-language summary of the UCI dataset and what it predicts.
- Target chips: target column, positive/negative classes and the 22.1% default rate.
- Responsive attribute-family cards showing every column group and what it is used for.
- **Interactive attribute explorer**: definition, purpose, statistics and a distribution chart split by outcome.
- A data dictionary expander with every raw and engineered attribute.

### 🔮 Make a prediction
- 1-click profiles: **Prime payer**, **Revolving borrower**, **High-risk delinquent**, plus reset.
- Short input form for the important fields; earlier months are optional and pre-filled.
- **Model selector with all seven saved models** — the chosen model drives the verdict.
- Loads the saved `.joblib` weights — **no training inside the UI**.
- Selected-model verdict, default probability gauge and test accuracy.
- Collapsible **Compare all models** table and chart, plus model details with precision, recall and F1.

Launch it:

```bash
streamlit run app.py
```

---

## 🧠 Models and evaluation

Seven models are tuned with cross-validation using **simple hyperparameter grids**:

| Model | Scaling | Tuned hyperparameters |
|---|---|---|
| Logistic Regression | ✅ | `C` |
| KNN | ✅ | `n_neighbors` |
| Decision Tree | — | `max_depth` |
| Random Forest | — | `max_depth` |
| Gradient Boosting | — | `n_estimators`, `learning_rate` |
| XGBoost | — | `max_depth`, `learning_rate` |
| AdaBoost | — | `n_estimators` |

The **best model is selected by test accuracy**. `metrics.json` stores accuracy, precision, recall,
F1, the confusion matrix and the best parameters for every model.

Current run (reproducible with `random_state=42`): **XGBoost** is the best model, with about
**82.0% test accuracy**.

---

## 🚀 Setup

```bash
python -m pip install -r requirements.txt
```

Rebuild the executed notebook (this also retrains and saves the model weights):

```bash
python build_notebook.py
```

Or run the standalone training script:

```bash
python train.py
```

Quick smoke test with a tiny grid:

```bash
python train.py --quick --output-dir models_quick
```

Then start the app:

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

Streamlit Community Cloud deploys directly from a GitHub repository. The app needs only the small
data and model artifacts already in this project — no API keys, database or secrets.

### 1. Push the project to GitHub

The repository is small because `.gitignore` excludes the unrelated `data/creditcard.csv` (144 MB)
and `data/healthcare_fraud_detection.csv`. Keep `data/UCI_Credit_Card.csv` and `models/` in the repo.

```bash
git init
git add .
git commit -m "Credit card default Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

If Git asks who you are, set your details once:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

### 2. Create the app

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Choose the repository and branch `main`.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**. The first build installs the pinned packages in `requirements.txt`.

The Python version comes from `runtime.txt` (`python-3.12`), and the dark theme is applied from
`.streamlit/config.toml`. When the build finishes, Streamlit gives you a public URL such as
`https://<app-name>.streamlit.app`.

> Every push to `main` redeploys the app automatically.

---

## 🔁 Train / serve consistency

`src/preprocess.py` is the single source of truth for:

- cleaning (`ID` removal, undocumented category mapping, duplicate removal),
- the six engineered features,
- the final 29-column feature order.

`train.py`, the notebook and `app.py` all follow the same rules. The scaler is fitted once on
the training split and reused by the app for scale-sensitive models, so there is no train/serve
mismatch and no Pipeline object anywhere.
