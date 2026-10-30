import importlib

for m in ["sklearn", "streamlit", "xgboost", "plotly", "joblib", "shap"]:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"{m}: OK {ver}")
    except Exception:
        print(f"{m}: MISSING")
