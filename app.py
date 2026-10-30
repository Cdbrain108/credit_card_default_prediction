"""
Credit Card Default Prediction — Streamlit UI
=============================================
A simple dark-themed app with two large sidebar navigation buttons:

1. Dataset overview
   - concise plain-language dataset summary and target chips
   - attribute-family cards (what each group is and what it is used for)
   - interactive attribute explorer with precomputed statistics and charts

2. Make a prediction
   - 1-click customer profiles + a short input form
   - a selector for all seven saved models
   - loads the stored joblib weights only (no training in the UI)
   - selected-model verdict, probability gauge and test accuracy

Run:
    streamlit run app.py
"""

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocess import (
    COLUMN_INFO,
    EDUCATION_LABELS,
    ENGINEERED_FEATURES,
    MARRIAGE_LABELS,
    PAY_STATUS_LABELS,
    RAW_FEATURES,
    SEX_LABELS,
    TARGET,
    build_features,
    clean,
    form_defaults,
    load_data,
)

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


# --------------------------------------------------------------------------- #
# Minimal, responsive design system
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      :root {
        --ink: #f1f5f9;
        --muted: #94a3b8;
        --line: #263548;
        --card: #111827;
        --accent: #38bdf8;
        --accent-soft: #082f49;
        --success: #4ade80;
        --success-soft: #052e24;
        --danger: #f87171;
        --danger-soft: #3b1018;
        --warning: #fbbf24;
        --warning-soft: #3a2a08;
      }

      html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        color: var(--ink);
      }

      .main .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
      }

      #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

      /* Section headings */
      .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: var(--ink);
        margin: 8px 0 10px 0;
      }
      .section-sub {
        font-size: 0.86rem;
        color: var(--muted);
        margin: -4px 0 12px 0;
      }

      /* Small info cards */
      .info-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        height: 100%;
      }
      .info-card .label {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
      }
      .info-card .body {
        font-size: 0.88rem;
        line-height: 1.55;
        color: #cbd5e1;
      }
      .info-card .use {
        margin-top: 10px;
        background: var(--accent-soft);
        border: 1px solid #164e63;
        border-radius: 9px;
        padding: 8px 11px;
        font-size: 0.78rem;
        line-height: 1.45;
        color: #bae6fd;
      }
      .info-card .use b { color: #e0f2fe; }

      /* Verdict */
      .verdict {
        border-radius: 16px;
        padding: 22px;
        border: 1px solid var(--line);
        background: var(--card);
      }
      .verdict.danger {
        background: var(--danger-soft);
        border-color: #7f1d1d;
      }
      .verdict.safe {
        background: var(--success-soft);
        border-color: #14532d;
      }
      .verdict.warn {
        background: var(--warning-soft);
        border-color: #713f12;
      }
      .verdict .eyebrow {
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
      }
      .verdict .title {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 7px;
      }
      .verdict.danger .title { color: var(--danger); }
      .verdict.safe .title { color: var(--success); }
      .verdict.warn .title { color: var(--warning); }
      .verdict .copy {
        font-size: 0.87rem;
        line-height: 1.55;
        color: #94a3b8;
      }
      .verdict .metric {
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 10px;
        color: var(--ink);
      }
      .verdict .metric small {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--muted);
      }

      /* Stat strip — wraps so boxes stay readable in narrow columns */
      .stat-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .stat-box {
        flex: 1 1 130px;
        min-width: 130px;
        background: var(--accent-soft);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px;
      }
      .stat-box .k {
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .stat-box .v {
        font-size: 1.08rem;
        font-weight: 800;
        color: var(--ink);
        margin-top: 3px;
      }
      .stat-box .n {
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 2px;
      }

      /* Buttons — consistent size and contrast */
      [data-testid="stButton"] > button,
      [data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 48px;
        font-size: 0.95rem;
      }
      [data-testid="stFormSubmitButton"] > button {
        min-height: 56px;
        font-size: 1.02rem;
        font-weight: 800;
      }
      [data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primaryFormSubmit"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #082f49;
      }
      [data-testid="stFormSubmitButton"] > button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: #7dd3fc;
        border-color: #7dd3fc;
        color: #082f49;
      }
      [data-baseweb="input"], [data-baseweb="select"] {
        border-radius: 9px !important;
      }
      [data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        background: var(--card);
      }
      [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border-color: var(--line) !important;
        background: var(--card);
      }
      [data-testid="stDataFrame"] {
        font-size: 0.82rem;
      }
      div[data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 18px 8px 18px;
        background: var(--card);
      }

      html, body, [data-testid="stAppViewContainer"] {
        background: #0b1220 !important;
        color: #e2e8f0 !important;
      }
      [data-testid="stSidebar"] {
        background: #080d17 !important;
        border-right: 1px solid #243247;
      }
      [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 60px;
        justify-content: flex-start;
        padding: 0 16px;
        border: 1px solid #263548;
        border-radius: 12px;
        background: #111827;
        color: #cbd5e1;
        font-size: 1rem;
        font-weight: 750;
        text-align: left;
        transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
      }
      [data-testid="stSidebar"] .stButton > button:hover {
        background: #172033;
        border-color: #0ea5e9;
        color: #e0f2fe;
      }
      [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #082f49;
        border: 1px solid #0ea5e9;
        color: #e0f2fe;
      }
      .main .block-container { padding-top: 2rem; }
      .info-card, .verdict, div[data-testid="stForm"],
      [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] {
        background: #111827 !important;
        border-color: #263548 !important;
        color: #e2e8f0 !important;
      }
      .section-title, .info-card .body, .verdict .title, .verdict .metric { color: #e2e8f0; }
      .section-sub, .verdict .copy, .info-card .label { color: #94a3b8; }
      .stat-box { background: #0b1220; border-color: #263548; }
      .stat-box .v { color: #e2e8f0; }
      [data-baseweb="input"], [data-baseweb="select"] > div {
        background: #0b1220 !important;
        border-color: #334155 !important;
      }
      [data-baseweb="input"] input, [data-baseweb="select"] div { color: #e2e8f0 !important; }
      [data-testid="stWidgetLabel"] p, .stCaption, label { color: #cbd5e1 !important; }
      .verdict.safe { background: #052e24 !important; border-color: #166534 !important; }
      .verdict.warn { background: #3a2a08 !important; border-color: #854d0e !important; }
      .verdict.danger { background: #3b1018 !important; border-color: #991b1b !important; }

      /* Attribute family cards */
      .attr-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
        gap: 12px;
        margin: 12px 0 20px 0;
      }
      .attr-family {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
      }
      .attr-head {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 9px;
      }
      .attr-icon {
        width: 34px;
        height: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 9px;
        background: #082f49;
        border: 1px solid #164e63;
        font-size: 1rem;
      }
      .attr-name {
        font-size: 1rem;
        font-weight: 800;
        color: #e2e8f0;
      }
      .attr-cols {
        font-size: 0.78rem;
        line-height: 1.7;
        color: #bae6fd;
        background: #0b1220;
        border: 1px dashed #334155;
        border-radius: 9px;
        padding: 8px 10px;
        word-break: break-word;
        overflow-wrap: anywhere;
      }
      .attr-purpose {
        margin-top: 9px;
        font-size: 0.82rem;
        line-height: 1.55;
        color: #94a3b8;
      }
      .target-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 4px 0;
      }
      .target-chip {
        font-size: 0.78rem;
        color: #cbd5e1;
        background: #111827;
        border: 1px solid #263548;
        border-radius: 999px;
        padding: 6px 12px;
      }
      .target-chip b { color: #e0f2fe; }

      @media (max-width: 760px) {
        .main .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        .verdict .title { font-size: 1.15rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
          flex: 1 1 100% !important;
          min-width: 100% !important;
          width: 100% !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading stored model weights...")
def load_artifacts():
    """Load only stored joblib weights + metadata. No training happens here."""
    metrics_path = MODEL_DIR / "metrics.json"
    scaler_path = MODEL_DIR / "scaler.joblib"
    desc_path = DATA_DIR / "dataset_description.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            "models/metrics.json is missing. Run `python train.py` first."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    description = (
        json.loads(desc_path.read_text(encoding="utf-8"))
        if desc_path.exists()
        else {}
    )

    models = {}
    for name, info in metrics.get("results", {}).items():
        filename = info.get("model_file", name.replace(" ", "_") + ".joblib")
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)

    if not models:
        raise FileNotFoundError(
            "No model files found in models/. Run `python train.py` first."
        )

    return models, scaler, metrics, description


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    """Compute the form defaults and the full explorer frame ONCE.

    The frame already contains the 29 model inputs plus the target and its
    label, so attribute switches never re-run cleaning or feature building.
    """
    raw = load_data()
    cleaned = clean(raw)
    defaults = form_defaults(raw)
    explorer = build_features(cleaned)
    explorer[TARGET] = cleaned[TARGET]
    explorer["_target_label"] = explorer[TARGET].map({0: "Paid", 1: "Default"})
    return defaults, explorer


try:
    MODELS, SCALER, METRICS, DESCRIPTION = load_artifacts()
    DEFAULTS, EXPLORER = load_dataset()
except Exception as exc:
    st.error(f"Could not start the app: {exc}")
    st.stop()


BEST_NAME = METRICS.get("best_model", next(iter(MODELS)))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def pct(value, digits=1):
    return f"{float(value) * 100:.{digits}f}%"


def plotly_style(fig, height=300):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=54, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#cbd5e1", size=12),
        title={"x": 0.5, "xanchor": "center", "y": 0.96, "font": {"size": 15, "color": "#e2e8f0"}},
        xaxis=dict(gridcolor="#1e2b40", zerolinecolor="#334155", color="#cbd5e1"),
        yaxis=dict(gridcolor="#1e2b40", zerolinecolor="#334155", color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 💳 Credit Risk")
    st.caption("UCI credit card default")
    if "page_navigation" not in st.session_state:
        st.session_state.page_navigation = "Dataset overview"
    page_choices = [
        ("Dataset overview", "📊  Dataset overview"),
        ("Make a prediction", "🔮  Make a prediction"),
    ]
    for page_value, page_label in page_choices:
        if st.button(
            page_label,
            key=f"nav_{page_value}",
            type="primary" if st.session_state.page_navigation == page_value else "secondary",
            width="stretch",
        ):
            st.session_state.page_navigation = page_value
            st.rerun()
    page = st.session_state.page_navigation
    st.markdown("---")
    st.caption("30,000 customer records · 7 saved models")


# =========================================================================== #
# TAB 1 — DATASET & ATTRIBUTES
# =========================================================================== #
if page == "Dataset overview":
    st.title("Credit card default")
    st.caption("UCI Default of Credit Card Clients · Taiwan · 2005")
    description = DESCRIPTION.get("summary", "30,000 credit-card accounts used to study next-month payment default.")
    st.markdown(
        f"<div class='info-card'><div class='body'>{html.escape(description)}</div></div>",
        unsafe_allow_html=True,
    )
    target_info = DESCRIPTION.get("target", {})
    st.markdown(
        f"""
        <div class="target-chips">
          <span class="target-chip">🎯 <b>{html.escape(target_info.get('name', TARGET))}</b></span>
          <span class="target-chip">{html.escape(target_info.get('positive', '1 = default'))}</span>
          <span class="target-chip">{html.escape(target_info.get('negative', '0 = paid'))}</span>
          <span class="target-chip"><b>{target_info.get('default_rate_pct', 22.12):.1f}%</b> default rate</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### What the data contains")
    st.caption("23 raw attributes in four groups, plus 6 engineered ratios.")
    families = DESCRIPTION.get("attribute_families", [])
    if families:
        cards_html = "".join(
            (
                '<div class="attr-family">'
                f'<div class="attr-head"><span class="attr-icon">{item.get("icon", "•")}</span>'
                f'<span class="attr-name">{html.escape(item.get("name", ""))}</span></div>'
                f'<div class="attr-cols">{html.escape(item.get("columns", ""))}</div>'
                f'<div class="attr-purpose"><b>What it\'s used for:</b> {html.escape(item.get("use", ""))}</div>'
                "</div>"
            )
            for item in families
        )
        st.markdown(f'<div class="attr-grid">{cards_html}</div>', unsafe_allow_html=True)

    st.markdown("#### Explore an attribute")

    all_attributes = [TARGET] + RAW_FEATURES + ENGINEERED_FEATURES
    selected = st.selectbox(
        "Choose an attribute",
        options=all_attributes,
        index=all_attributes.index("PAY_0"),
        format_func=lambda c: f"{c}  ·  {COLUMN_INFO.get(c, {}).get('group', '')}",
        label_visibility="collapsed",
    )

    info = COLUMN_INFO.get(selected, {})
    series = EXPLORER[selected]

    stat_col, chart_col = st.columns([1, 1.55], gap="medium")
    with stat_col:
        if pd.api.types.is_numeric_dtype(series):
            stats = [
                ("Mean", f"{series.mean():,.2f}"),
                ("Median", f"{series.median():,.2f}"),
                ("Minimum", f"{series.min():,.2f}"),
                ("Maximum", f"{series.max():,.2f}"),
                ("Std. deviation", f"{series.std():,.2f}"),
                ("Unique values", f"{series.nunique():,}"),
            ]
        else:
            stats = [
                ("Rows", f"{len(series):,}"),
                ("Unique", f"{series.nunique():,}"),
                ("Most common", str(series.mode().iloc[0]) if not series.mode().empty else "—"),
                ("Missing", f"{series.isna().sum():,}"),
            ]

        stat_html = "".join(
            f'<div class="stat-box"><div class="k">{k}</div><div class="v">{v}</div></div>'
            for k, v in stats
        )
        st.markdown(
            f"""
            <div class="info-card">
              <div class="label">{html.escape(info.get('group', 'Attribute'))} · {html.escape(info.get('type', ''))}</div>
              <div style="font-size:1.18rem;font-weight:800;margin-bottom:6px;">{html.escape(selected)}</div>
              <div class="body" style="margin-bottom:12px;"><b>What it is:</b> {html.escape(info.get('description', ''))}</div>
              <div class="use">🎯 <b>What it's used for:</b> {html.escape(info.get('use', 'Helps the models estimate default risk.'))}</div>
              <div class="stat-strip" style="margin-top:14px;">{stat_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with chart_col:
        chart_df = EXPLORER
        if selected == TARGET:
            counts = chart_df[TARGET].value_counts().sort_index().reset_index()
            counts.columns = [TARGET, "Count"]
            counts["Label"] = counts[TARGET].map({0: "Paid", 1: "Default"})
            fig = px.bar(
                counts,
                x="Label",
                y="Count",
                color="Label",
                text="Count",
                color_discrete_map={"Paid": "#16a34a", "Default": "#dc2626"},
                title="Target distribution",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
        elif series.nunique() <= 12:
            grouped = (
                chart_df.groupby([selected, "_target_label"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                grouped,
                x=selected,
                y="Count",
                color="_target_label",
                barmode="group",
                color_discrete_map={"Paid": "#16a34a", "Default": "#dc2626"},
                title=f"{selected} distribution by outcome",
                labels={"_target_label": "Outcome"},
            )
        else:
            fig = px.histogram(
                chart_df,
                x=selected,
                color="_target_label",
                nbins=40,
                barmode="overlay",
                opacity=0.68,
                color_discrete_map={"Paid": "#16a34a", "Default": "#dc2626"},
                title=f"{selected} distribution by outcome",
                labels={"_target_label": "Outcome"},
            )
        st.plotly_chart(plotly_style(fig, height=330), width="stretch")

    with st.expander("Data dictionary", expanded=False):
        st.dataframe(
            pd.DataFrame([
                {"Attribute": name, "Group": meta["group"], "Definition": meta["description"]}
                for name, meta in COLUMN_INFO.items()
            ]),
            width="stretch",
            hide_index=True,
        )


# =========================================================================== #
# TAB 2 — MAKE A PREDICTION
# =========================================================================== #
PRESETS = {
    "Prime payer": {
        "LIMIT_BAL": 350000.0, "AGE": 38, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 1,
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 12000.0, "BILL_AMT2": 11000.0, "BILL_AMT3": 10000.0,
        "BILL_AMT4": 10000.0, "BILL_AMT5": 9000.0, "BILL_AMT6": 9000.0,
        "PAY_AMT1": 12000.0, "PAY_AMT2": 11000.0, "PAY_AMT3": 10000.0,
        "PAY_AMT4": 10000.0, "PAY_AMT5": 9000.0, "PAY_AMT6": 9000.0,
    },
    "Revolving borrower": {
        "LIMIT_BAL": 120000.0, "AGE": 29, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 65000.0, "BILL_AMT2": 62000.0, "BILL_AMT3": 60000.0,
        "BILL_AMT4": 58000.0, "BILL_AMT5": 56000.0, "BILL_AMT6": 54000.0,
        "PAY_AMT1": 3500.0, "PAY_AMT2": 3200.0, "PAY_AMT3": 3000.0,
        "PAY_AMT4": 2800.0, "PAY_AMT5": 2600.0, "PAY_AMT6": 2500.0,
    },
    "High-risk delinquent": {
        "LIMIT_BAL": 50000.0, "AGE": 26, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 2,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": 1, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 48500.0, "BILL_AMT2": 47000.0, "BILL_AMT3": 45500.0,
        "BILL_AMT4": 44000.0, "BILL_AMT5": 43000.0, "BILL_AMT6": 42000.0,
        "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 500.0,
        "PAY_AMT4": 800.0, "PAY_AMT5": 1000.0, "PAY_AMT6": 1000.0,
    },
}


def init_form_state():
    for col in RAW_FEATURES:
        key = f"in_{col}"
        if key not in st.session_state:
            st.session_state[key] = DEFAULTS[col]


def apply_preset(values):
    for col in RAW_FEATURES:
        st.session_state[f"in_{col}"] = values.get(col, DEFAULTS[col])
    st.session_state.pop("prediction", None)


def reset_form():
    for col in RAW_FEATURES:
        st.session_state[f"in_{col}"] = DEFAULTS[col]
    st.session_state.pop("prediction", None)


def score_profile(inputs):
    row = pd.DataFrame([inputs])
    X_row = build_features(row)
    rows = []
    for name, model in MODELS.items():
        info = METRICS.get("results", {}).get(name, {})
        needs_scaling = bool(info.get("needs_scaling", False))
        if needs_scaling:
            if SCALER is None:
                continue
            matrix = SCALER.transform(X_row)
        else:
            matrix = X_row.values
        prob = float(model.predict_proba(matrix)[0, 1])
        rows.append(
            {
                "model": name,
                "prob": prob,
                "pred": int(prob >= 0.5),
                "accuracy": float(info.get("test_accuracy", 0.0)),
                "params": info.get("best_params", {}),
            }
        )
    rows.sort(key=lambda r: (r["model"] != BEST_NAME, -r["accuracy"]))
    return rows, X_row


init_form_state()

if page == "Make a prediction":
    st.title("Make a prediction")

    preset_cols = st.columns(4, gap="small")
    preset_labels = list(PRESETS.keys()) + ["Reset to median"]
    for col, label in zip(preset_cols, preset_labels):
        with col:
            if st.button(label, width="stretch", key=f"preset_{label}"):
                if label == "Reset to median":
                    reset_form()
                else:
                    apply_preset(PRESETS[label])
                st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    model_options = list(MODELS.keys())
    if st.session_state.get("prediction_model") not in model_options:
        st.session_state["prediction_model"] = BEST_NAME if BEST_NAME in model_options else model_options[0]
    model_col, model_note_col = st.columns([1.15, 1], gap="medium")
    with model_col:
        selected_model = st.selectbox(
            "Choose the model for this prediction",
            options=model_options,
            key="prediction_model",
        )
    with model_note_col:
        selected_accuracy = float(
            METRICS.get("results", {}).get(selected_model, {}).get("test_accuracy", 0.0)
        )
        st.caption(f"{selected_model} · test accuracy {pct(selected_accuracy)}")

    with st.form("credit_profile_form", clear_on_submit=False):
        st.markdown("##### Customer profile")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            limit_bal = st.number_input(
                "Credit limit (NT$)",
                min_value=10000.0, max_value=1000000.0, step=10000.0,
                key="in_LIMIT_BAL",
            )
            age = st.slider("Age", min_value=18, max_value=80, step=1, key="in_AGE")
        with c2:
            sex = st.selectbox(
                "Gender", options=[1, 2],
                format_func=lambda x: SEX_LABELS[x],
                key="in_SEX",
            )
            education = st.selectbox(
                "Education", options=[1, 2, 3, 4],
                format_func=lambda x: EDUCATION_LABELS[x],
                key="in_EDUCATION",
            )
        with c3:
            marriage = st.selectbox(
                "Marital status", options=[1, 2, 3],
                format_func=lambda x: MARRIAGE_LABELS[x],
                key="in_MARRIAGE",
            )

        st.markdown("##### Repayment behaviour")
        st.caption("PAY_0 is the most recent month; values 1 or above indicate late payment.")
        p1, p2, p3 = st.columns(3, gap="medium")
        status_options = sorted(PAY_STATUS_LABELS)
        with p1:
            pay_0 = st.selectbox(
                "PAY_0 — last month", options=status_options,
                format_func=lambda x: PAY_STATUS_LABELS[x], key="in_PAY_0",
            )
        with p2:
            pay_2 = st.selectbox(
                "PAY_2 — 2 months ago", options=status_options,
                format_func=lambda x: PAY_STATUS_LABELS[x], key="in_PAY_2",
            )
        with p3:
            pay_3 = st.selectbox(
                "PAY_3 — 3 months ago", options=status_options,
                format_func=lambda x: PAY_STATUS_LABELS[x], key="in_PAY_3",
            )
        with st.expander("Earlier repayment months (PAY_4 – PAY_6)", expanded=False):
            q1, q2, q3 = st.columns(3, gap="medium")
            with q1:
                st.selectbox(
                    "PAY_4 — 4 months ago", options=status_options,
                    format_func=lambda x: PAY_STATUS_LABELS[x], key="in_PAY_4",
                )
            with q2:
                st.selectbox(
                    "PAY_5 — 5 months ago", options=status_options,
                    format_func=lambda x: PAY_STATUS_LABELS[x], key="in_PAY_5",
                )
            with q3:
                st.selectbox(
                    "PAY_6 — 6 months ago", options=status_options,
                    format_func=lambda x: PAY_STATUS_LABELS[x], key="in_PAY_6",
                )

        st.markdown("##### Latest statement")
        b1, b2 = st.columns(2, gap="medium")
        with b1:
            st.number_input(
                "BILL_AMT1 — latest statement (NT$)",
                min_value=-100000.0, max_value=1000000.0, step=1000.0,
                key="in_BILL_AMT1",
            )
        with b2:
            st.number_input(
                "PAY_AMT1 — latest cash payment (NT$)",
                min_value=0.0, max_value=1000000.0, step=100.0,
                key="in_PAY_AMT1",
            )

        with st.expander("Earlier bills and payments (months 2 – 6)", expanded=False):
            st.caption("Optional. Leave the defaults if you only know the latest statement.")
            month_cols = st.columns(5, gap="small")
            for col, month in zip(month_cols, range(2, 7)):
                with col:
                    st.number_input(
                        f"Bill {month}",
                        min_value=-100000.0, max_value=1000000.0, step=1000.0,
                        key=f"in_BILL_AMT{month}",
                    )
                    st.number_input(
                        f"Paid {month}",
                        min_value=0.0, max_value=1000000.0, step=100.0,
                        key=f"in_PAY_AMT{month}",
                    )

        submitted = st.form_submit_button(
            "Predict default risk", type="primary", width="stretch"
        )

    if submitted:
        inputs = {col: st.session_state[f"in_{col}"] for col in RAW_FEATURES}
        rows, X_row = score_profile(inputs)
        if not rows:
            st.error("No stored model could score this profile. Check the files in models/.")
        else:
            st.session_state["prediction"] = {
                "inputs": inputs,
                "rows": rows,
                "X_row": X_row,
            }

    prediction = st.session_state.get("prediction")
    if prediction and prediction.get("rows"):
        rows = prediction["rows"]
        X_row = prediction["X_row"]
        selected_row = next(
            (row for row in rows if row["model"] == selected_model),
            rows[0],
        )
        prob = selected_row["prob"]
        pred = selected_row["pred"]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Assessment</div>', unsafe_allow_html=True)

        if pred == 1:
            verdict_class = "danger"
            verdict_title = "High default risk"
            verdict_copy = (
                f"{selected_model} expects this account to default on the next payment. "
                "Treat the application as high risk and review it manually."
            )
        elif prob >= 0.30:
            verdict_class = "warn"
            verdict_title = "Medium default risk"
            verdict_copy = (
                "The account is not an immediate default, but the probability is "
                "elevated. Monitor the next statement and payment closely."
            )
        else:
            verdict_class = "safe"
            verdict_title = "Low default risk"
            verdict_copy = (
                f"{selected_model} expects this account to pay on time. "
                "The profile is consistent with a standard, performing customer."
            )

        left, right = st.columns([1.05, 1], gap="medium")
        with left:
            st.markdown(
                f"""
                <div class="verdict {verdict_class}">
                  <div class="eyebrow">Selected model · {html.escape(selected_row['model'])}</div>
                  <div class="title">{verdict_title}</div>
                  <div class="copy">{verdict_copy}</div>
                  <div class="metric">{pct(prob)} <small>probability of default</small></div>
                  <div style="margin-top:10px;font-size:0.78rem;color:#64748b;">
                    Test accuracy <b>{pct(selected_row['accuracy'])}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Test accuracy is the model's overall score on the 20% hold-out set, "
                "not a confidence score for this individual customer."
            )

        with right:
            gauge = go.Figure(
                go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                    number={"suffix": "%", "font": {"size": 32, "color": "#e2e8f0"}},
                    title={"text": "Default probability", "font": {"size": 14, "color": "#cbd5e1"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                        "bar": {
                            "color": "#dc2626" if pred else ("#ca8a04" if prob >= 0.30 else "#16a34a"),
                            "thickness": 0.55,
                        },
                        "bgcolor": "#111827",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30], "color": "#052e24"},
                            {"range": [30, 50], "color": "#3a2a08"},
                            {"range": [50, 100], "color": "#3b1018"},
                        ],
                        "threshold": {
                        "line": {"color": "#e2e8f0", "width": 2},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                )
            )
            gauge.update_layout(
                height=250,
                margin=dict(l=25, r=25, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Segoe UI, Arial, sans-serif", "color": "#cbd5e1"},
            )
            st.plotly_chart(gauge, width="stretch")

        with st.expander("Compare all models", expanded=False):
            st.markdown("**Model results for this customer**")
            table_rows = []
            for row in rows:
                table_rows.append(
                    {
                        "Model": ("Best · " if row["model"] == BEST_NAME else "") + row["model"],
                        "Prediction": "Default" if row["pred"] else "Will pay",
                        "Default probability": row["prob"] * 100,
                        "Test accuracy": row["accuracy"] * 100,
                    }
                )
            table_df = pd.DataFrame(table_rows)
            st.dataframe(
                table_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Model": st.column_config.TextColumn(width="medium"),
                    "Prediction": st.column_config.TextColumn(width="small"),
                    "Default probability": st.column_config.ProgressColumn(
                        "Default probability", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "Test accuracy": st.column_config.ProgressColumn(
                        "Test accuracy", format="%.1f%%", min_value=0, max_value=100
                    ),
                },
            )
            st.caption(
                f"{sum(r['pred'] for r in rows)} of {len(rows)} models predict default. "
                "Choose a model above to change the main verdict."
            )

            chart_rows = sorted(rows, key=lambda r: r["prob"])
            chart_df = pd.DataFrame(
                {"Model": [r["model"] for r in chart_rows], "Probability": [r["prob"] * 100 for r in chart_rows]}
            )
            prob_fig = px.bar(chart_df, x="Probability", y="Model", orientation="h", text="Probability")
            prob_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_color="#0ea5e9")
            prob_fig.update_layout(
                title="Default probability by model",
                xaxis=dict(range=[0, 108], title="Probability (%)"),
                yaxis=dict(title=""),
                showlegend=False,
            )
            st.plotly_chart(plotly_style(prob_fig, height=300), width="stretch")

        with st.expander("Model details", expanded=False):
            detail_rows = []
            for name in [r["model"] for r in rows]:
                info = METRICS.get("results", {}).get(name, {})
                detail_rows.append(
                    {
                        "Model": name,
                        "Best hyperparameters": json.dumps(info.get("best_params", {})),
                        "Test accuracy": info.get("test_accuracy", np.nan),
                        "Precision": info.get("test_precision", np.nan),
                        "Recall": info.get("test_recall", np.nan),
                        "F1": info.get("test_f1", np.nan),
                    }
                )
            st.dataframe(
                pd.DataFrame(detail_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Best hyperparameters": st.column_config.TextColumn(width="large"),
                    "Test accuracy": st.column_config.NumberColumn(format="%.4f"),
                    "Precision": st.column_config.NumberColumn(format="%.4f"),
                    "Recall": st.column_config.NumberColumn(format="%.4f"),
                    "F1": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.caption(
                "These are the simple parameters selected by GridSearchCV on the training set. "
                "The test set was never used for tuning."
            )
