
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_PATHS = [
    ROOT / "data" / "creditcard.csv",
    ROOT / "FraudGuardAI" / "creditcard.csv",
    ROOT / "creditcard.csv",
]
MODEL_PATH = ROOT / "models" / "fraud_xgboost_pipeline.joblib"
DRIFT_PATH = ROOT / "reports" / "drift_summary.json"

BLUE = "#459cff"
CYAN = "#4bdcff"
PURPLE = "#9d7aff"
RED = "#ff5575"
GREEN = "#2fe5ad"
BG = "#071326"


def setup(title="FraudGuard AI"):
    st.set_page_config(
        page_title=title,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    :root {
        --fg-blue:#459cff;
        --fg-cyan:#4bdcff;
        --fg-text:#eaf3ff;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 75% 0%,
            rgba(48,94,184,.20), transparent 40%),
          radial-gradient(ellipse at 15% 75%,
            rgba(96,47,178,.12), transparent 45%),
          #071326;
        color:#eaf3ff;
    }
    [data-testid="stSidebar"] {
        background:linear-gradient(
            175deg,#101f3b 0%,#081428 100%);
        border-right:1px solid rgba(109,159,255,.17);
    }
    [data-testid="stSidebar"] * {
        color:#dceaff;
    }
    [data-testid="stHeader"] {
        background:transparent;
    }
    .block-container {
        padding-top:1.4rem;
        max-width:1500px;
    }
    h1,h2,h3 {
        color:#f3f8ff!important;
        letter-spacing:-.025em;
    }
    p,li,label {
        color:#d4e2f5;
    }
    .fg-hero {
        padding:26px 30px;
        border-radius:22px;
        background:linear-gradient(
            115deg,rgba(26,56,105,.95),
            rgba(12,29,56,.95));
        border:1px solid rgba(100,167,255,.26);
        box-shadow:
            0 16px 35px rgba(0,0,0,.25),
            inset 0 1px 0 rgba(255,255,255,.09);
        margin-bottom:22px;
    }
    .fg-hero h1 {
        font-size:2.1rem;
        margin:0 0 7px 0;
    }
    .fg-hero p {
        margin:0;
        color:#a9c4e8;
    }
    .fg-kpi {
        min-height:128px;
        padding:19px 21px;
        border-radius:18px;
        background:linear-gradient(
            145deg,#172c4c,#0c1b32);
        border:1px solid rgba(104,161,244,.20);
        box-shadow:
            7px 12px 25px rgba(0,0,0,.26),
            inset 0 1px 0 rgba(255,255,255,.08);
        margin-bottom:12px;
    }
    .fg-kpi .label {
        color:#a7bfdc;
        font-size:.86rem;
    }
    .fg-kpi .value {
        color:#f6faff;
        font-size:1.85rem;
        font-weight:800;
        margin-top:12px;
    }
    .fg-kpi .sub {
        color:#5bd9ff;
        font-size:.76rem;
        margin-top:5px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:rgba(15,32,57,.72);
        border-color:rgba(95,155,255,.17)!important;
        border-radius:18px;
    }
    div.stButton>button {
        border-radius:11px;
        border:1px solid #367be3;
        background:linear-gradient(120deg,#2167d0,#439bfa);
        color:white;
        font-weight:700;
        box-shadow:0 7px 16px rgba(16,86,189,.22);
    }
    div.stButton>button:hover {
        color:white;
        border-color:#72c8ff;
    }
    [data-testid="stMetric"] {
        background:#10233e;
        border:1px solid #29476a;
        padding:17px;
        border-radius:14px;
    }
    [data-testid="stDataFrame"] {
        border:1px solid #284467;
        border-radius:12px;
    }
    hr {border-color:#284467;}
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            "## 🛡️ FraudGuard AI\n"
            "**Fraud Intelligence Platform**"
        )
        st.caption("Cost-sensitive learning • XGBoost • Drift")
        st.divider()


def hero(title, subtitle):
    st.markdown(
        '<div class="fg-hero">'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def kpi(label, value, subtitle=""):
    import html

    st.markdown(
        '<div class="fg-kpi">'
        f'<div class="label">{html.escape(str(label))}</div>'
        f'<div class="value">{html.escape(str(value))}</div>'
        f'<div class="sub">{html.escape(str(subtitle))}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def plot(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dceaff", size=12),
        margin=dict(l=18, r=18, t=35, b=25),
        legend=dict(orientation="h", y=-.22),
        xaxis=dict(
            gridcolor="rgba(130,167,210,.10)",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(130,167,210,.13)",
            zeroline=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner="Loading transaction dataset...")
def load_data():
    path = next((p for p in DATA_PATHS if p.exists()), None)

    if path is None:
        raise FileNotFoundError(
            "creditcard.csv not found. Put it inside data/ "
            "or FraudGuardAI/."
        )

    df = pd.read_csv(path)
    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset missing columns: " + ", ".join(sorted(missing))
        )

    for col in ("Time", "Amount", "Class"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Time", "Amount", "Class"])
    df = df[df["Class"].isin([0, 1])].copy()
    df["Class"] = df["Class"].astype(int)
    df["transaction_id"] = df.index + 1

    return df


def get_data():
    try:
        return load_data()
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        st.stop()


@st.cache_resource(show_spinner="Loading XGBoost model...")
def model_bundle():
    from src.inference import load_model_bundle
    return load_model_bundle(MODEL_PATH)


def predict(df):
    from src.inference import predict_transactions
    return predict_transactions(df, bundle_path=MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_drift():
    if not DRIFT_PATH.exists():
        raise FileNotFoundError(str(DRIFT_PATH))
    with DRIFT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def ordered_transactions(df):
    return df.sort_values(
        ["Time", "transaction_id"],
        kind="stable",
    )


def prediction_batch(df):
    """Preserve row order while attaching model results."""
    result = predict(df).reset_index(drop=True)
    output = df.reset_index(drop=True).copy()

    for column in (
        "fraud_probability",
        "predicted_label",
        "predicted_class",
        "threshold_used",
    ):
        output[column] = result[column].to_numpy()

    output["actual_label"] = np.where(
        output["Class"] == 1, "Fraud", "Legitimate"
    )
    return output


def read_optional_csv(*paths):
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return None
