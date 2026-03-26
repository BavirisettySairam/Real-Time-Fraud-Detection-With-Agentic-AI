# =============================================================================
# Real-Time Fraud Detection with Agentic AI — Main Detection Page
# =============================================================================

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Real-Time Fraud Detection with Agentic AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_TRANSACTION_PATH = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_PATH = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_PATH = DATA_DIR / "test_identity.csv"
LIGHTGBM_METRICS_PATH = MODELS_DIR / "lightgbm_metrics.json"

TRANSACTION_REQUIRED_COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]
IDENTITY_REQUIRED_COLUMNS = [
    "TransactionID", "DeviceType", "DeviceInfo",
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18",
    "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25",
    "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32",
    "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
]

# =============================================================================
# MINIMAL CSS — white/grey + blue accent
# =============================================================================

st.markdown("""
<style>
    .decision-banner {
        padding: 1rem 1.5rem; border-radius: 8px; text-align: center;
        font-size: 1.3rem; font-weight: 700; margin: 0.5rem 0 1rem 0;
    }
    .decision-approve { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .decision-review  { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .decision-block   { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .agent-card {
        background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem; text-align: center;
    }
    .agent-card-title { font-weight: 600; font-size: 0.9rem; color: #495057; margin-bottom: 0.3rem; }
    .agent-card-score { font-size: 1.5rem; font-weight: 700; }
    .agent-card-desc  { font-size: 0.75rem; color: #6c757d; margin-top: 0.2rem; }
    .score-high   { color: #dc3545; }
    .score-medium { color: #ffc107; }
    .score-low    { color: #28a745; }
    .api-status { display: flex; align-items: center; gap: 8px; padding: 0.3rem 0; }
    .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .dot-green { background-color: #28a745; }
    .dot-red   { background-color: #dc3545; }
    .step-item { padding: 0.25rem 0.5rem; margin: 0.1rem 0; border-radius: 4px; font-size: 0.85rem; }
    .step-active  { background: #e3f2fd; border-left: 3px solid #1976d2; color: #1565c0; }
    .step-done    { background: #e8f5e9; border-left: 3px solid #2e7d32; color: #2e7d32; }
    .step-waiting { background: #f5f5f5; border-left: 3px solid #bdbdbd; color: #9e9e9e; }
    @media (max-width: 768px) {
        [data-testid="column"] { width: 100% !important; flex: 100% !important; min-width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=4)
        if r.status_code == 200:
            return {"ok": True, "data": r.json()}
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "Connection refused"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def call_predict_api(transaction_data: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/api/v1/predict", json=transaction_data, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 422:
            st.error("Invalid transaction data. Check the input fields.")
        elif r.status_code == 429:
            st.error("Rate limit reached. Wait a moment and retry.")
        else:
            st.error(f"Prediction failed (HTTP {r.status_code}).")
        return None
    except requests.exceptions.ConnectionError:
        return _run_local_fallback(transaction_data)
    except requests.exceptions.Timeout:
        st.error("Request timed out (30s).")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def call_explain_api(transaction_data: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/api/v1/explain", json=transaction_data, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 422:
            st.error("Invalid transaction data.")
        elif r.status_code == 429:
            st.error("Rate limit reached.")
        else:
            st.error(f"Explanation failed (HTTP {r.status_code}). Falling back to predict.")
        return None
    except requests.exceptions.ConnectionError:
        st.warning("API unreachable — running locally (no SHAP).")
        return _run_local_fallback(transaction_data)
    except requests.exceptions.Timeout:
        st.error("Explanation timed out (60s).")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def _run_local_fallback(transaction_data: dict) -> dict | None:
    orchestrator = None
    try:
        import numpy as np
        from ml.preprocessing import FeaturePipeline
        from services.orchestrator import AgentOrchestrator
        pipeline_path = MODELS_DIR / "feature_pipeline.pkl"
        if not pipeline_path.exists():
            st.error("Feature pipeline not found. Run `python train_agents.py` first.")
            return None
        pipeline = FeaturePipeline.load(str(pipeline_path))
        features = pipeline.transform(pd.DataFrame([transaction_data])).values[0].astype(np.float32)
        orchestrator = AgentOrchestrator()
        result = orchestrator.analyze(transaction_data, pipeline_features=features)
        score = result.get("final_score", 0.5)
        return {
            "transaction_id": transaction_data.get("TransactionID"),
            "fraud_probability": score,
            "decision": result.get("final_decision", "REVIEW"),
            "risk_level": "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW",
            "explanation": result.get("explanation", "Analysis complete"),
            "agent_scores": {
                "vibe_checker": result.get("vibe_score", 0.5),
                "agent_ensemble": result.get("agent_ensemble_score", 0.5),
                "era_tracker": result.get("era_score", 0.5),
                "og_check": result.get("og_score", 0.5),
            },
            "rule_violations": result.get("og_violations", []),
            "processing_time_ms": result.get("processing_time_ms", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        st.error(f"Local analysis failed: {e}")
        return None
    finally:
        if orchestrator is not None:
            orchestrator.close()


def _sanitize_for_json(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


# =============================================================================
# DATA LOADERS
# =============================================================================

@st.cache_data
def load_reference_payload() -> dict:
    transaction_path = TEST_TRANSACTION_PATH if TEST_TRANSACTION_PATH.exists() else TRAIN_TRANSACTION_PATH
    identity_path = TEST_IDENTITY_PATH if TEST_IDENTITY_PATH.exists() else TRAIN_IDENTITY_PATH
    if not transaction_path.exists():
        return {}
    tx_row = pd.read_csv(transaction_path, nrows=1)
    if tx_row.empty:
        return {}
    payload = {k: _sanitize_for_json(v) for k, v in tx_row.iloc[0].to_dict().items()}
    if identity_path.exists():
        id_row = pd.read_csv(identity_path, nrows=1)
        if not id_row.empty:
            payload.update({k: _sanitize_for_json(v) for k, v in id_row.iloc[0].to_dict().items()})
    return payload


def build_full_payload(overrides: dict) -> dict:
    base = dict(load_reference_payload())
    for k, v in overrides.items():
        base[k] = _sanitize_for_json(v)
    for col in TRANSACTION_REQUIRED_COLUMNS + IDENTITY_REQUIRED_COLUMNS:
        base.setdefault(col, None)
    if base.get("TransactionAmt") is None:
        base["TransactionAmt"] = 0.0
    if base.get("ProductCD") is None:
        base["ProductCD"] = "W"
    return base


@st.cache_data
def load_transaction_by_id(transaction_id: int) -> dict:
    tx_path = TRAIN_TRANSACTION_PATH
    id_path = TRAIN_IDENTITY_PATH
    if not tx_path.exists():
        return {}
    tid_col = pd.read_csv(tx_path, usecols=["TransactionID"])
    matches = tid_col.index[tid_col["TransactionID"] == transaction_id].tolist()
    if not matches:
        return {}
    row_idx = matches[0]
    tx_row = pd.read_csv(tx_path, skiprows=range(1, row_idx + 1), nrows=1)
    payload = {k: _sanitize_for_json(v) for k, v in tx_row.iloc[0].to_dict().items()}
    if id_path.exists():
        id_tid = pd.read_csv(id_path, usecols=["TransactionID"])
        id_matches = id_tid.index[id_tid["TransactionID"] == transaction_id].tolist()
        if id_matches:
            id_row = pd.read_csv(id_path, skiprows=range(1, id_matches[0] + 1), nrows=1)
            payload.update({k: _sanitize_for_json(v) for k, v in id_row.iloc[0].to_dict().items()})
    payload.pop("isFraud", None)
    return payload


def _get_default(reference_payload: dict, field: str, fallback=None):
    value = reference_payload.get(field, fallback)
    return fallback if value is None else value


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> tuple[bool, list[str]]:
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


# =============================================================================
# CHART BUILDERS
# =============================================================================

def create_gauge_chart(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"size": 16, "color": "#495057"}},
        number={"suffix": "%", "font": {"size": 32, "color": "#212529"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#adb5bd"},
            "bar": {"color": "#6c757d"},
            "bgcolor": "#f8f9fa",
            "steps": [
                {"range": [0, 40], "color": "#d4edda"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "#dc3545", "width": 3}, "thickness": 0.75, "value": 70},
        },
    ))
    fig.update_layout(
        height=250, margin=dict(t=40, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_shap_chart(top_features: list[dict]) -> go.Figure | None:
    if not top_features:
        return None
    sorted_feats = sorted(top_features, key=lambda f: abs(f.get("shap_value", 0)), reverse=True)[:10]
    sorted_feats = list(reversed(sorted_feats))
    names = [f.get("feature_name") or f.get("raw_name", "?") for f in sorted_feats]
    shap_vals = [f.get("shap_value", 0) for f in sorted_feats]
    colors = ["#dc3545" if v > 0 else "#1976d2" for v in shap_vals]
    fig = go.Figure(data=[go.Bar(
        x=shap_vals, y=names, orientation="h", marker_color=colors,
        text=[f"{v:+.4f}" for v in shap_vals], textposition="outside",
        textfont={"color": "#495057", "size": 11},
    )])
    fig.update_layout(
        title={"text": "SHAP Feature Impact (Top 10)", "font": {"color": "#495057", "size": 14}},
        xaxis_title="SHAP Value",
        xaxis=dict(gridcolor="#e9ecef", zerolinecolor="#adb5bd", color="#495057"),
        yaxis=dict(color="#495057"),
        height=max(320, len(sorted_feats) * 36 + 80),
        margin=dict(l=180, r=80, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#495057"},
    )
    fig.add_annotation(
        x=0.98, y=1.12, xref="paper", yref="paper",
        text='<span style="color:#dc3545">■</span> Increases fraud risk &nbsp; '
             '<span style="color:#1976d2">■</span> Decreases fraud risk',
        showarrow=False, font=dict(size=11, color="#6c757d"),
    )
    return fig


def create_agent_bar_chart(scores: dict) -> go.Figure:
    agents = list(scores.keys())
    values = [scores[a] * 100 for a in agents]
    colors = ["#dc3545" if v > 70 else "#ffc107" if v > 40 else "#28a745" for v in values]
    fig = go.Figure(data=[go.Bar(
        x=agents, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="auto",
        textfont={"color": "#495057"},
    )])
    fig.update_layout(
        xaxis_title="Agent", yaxis_title="Risk Score (%)",
        yaxis=dict(range=[0, 100], gridcolor="#e9ecef"),
        height=280, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#495057"},
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    st.sidebar.markdown("### API Status")
    health = check_api_health()
    if health["ok"]:
        d = health["data"]
        st.sidebar.markdown(
            f'<div class="api-status"><span class="status-dot dot-green"></span> '
            f'Online — v{d.get("version", "?")} — up {d.get("uptime_seconds", 0):.0f}s</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f'<div class="api-status"><span class="status-dot dot-red"></span> '
            f'Offline — {health["error"]}</div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Mode", ["Single Transaction", "Batch Upload", "Demo Mode"])
    return mode


# =============================================================================
# AGENT LOADING ANIMATION
# =============================================================================

_AGENT_STEPS = [
    ("Vibe Checker", "Scoring with LGB + XGB ensemble…"),
    ("Era Tracker", "Analyzing 24 h behavioural window…"),
    ("OG Check", "Running business rules + LGB…"),
    ("The Yapper", "Generating SHAP explanation…"),
]


def run_with_agent_steps(transaction_data: dict, use_explain: bool = True) -> dict | None:
    step_container = st.container()
    with step_container:
        placeholders = []
        for name, _ in _AGENT_STEPS:
            ph = st.empty()
            ph.markdown(
                f'<div class="step-item step-waiting">⏳ {name} — waiting</div>',
                unsafe_allow_html=True,
            )
            placeholders.append(ph)

    for i, (name, desc) in enumerate(_AGENT_STEPS):
        placeholders[i].markdown(
            f'<div class="step-item step-active">⚡ {name} — {desc}</div>',
            unsafe_allow_html=True,
        )
        if i < len(_AGENT_STEPS) - 1:
            time.sleep(0.25)

    result = call_explain_api(transaction_data) if use_explain else call_predict_api(transaction_data)

    for i, (name, _) in enumerate(_AGENT_STEPS):
        placeholders[i].markdown(
            f'<div class="step-item step-done">✓ {name} — done</div>',
            unsafe_allow_html=True,
        )

    return result


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def display_results(result: dict):
    decision = result.get("decision", "REVIEW")
    fraud_prob = result.get("fraud_probability", 0)
    risk_level = result.get("risk_level", "MEDIUM")

    # Decision banner
    dec_map = {
        "APPROVE": ("decision-approve", "✅ APPROVE"),
        "REVIEW": ("decision-review", "⚠️ REVIEW"),
        "BLOCK": ("decision-block", "🚫 BLOCK"),
    }
    css_cls, label = dec_map.get(decision, ("decision-review", f"⚠️ {decision}"))
    st.markdown(
        f'<div class="decision-banner {css_cls}">'
        f'{label} &nbsp;—&nbsp; Fraud Probability: {fraud_prob * 100:.1f}% &nbsp;—&nbsp; Risk: {risk_level}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metrics row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Fraud Probability", f"{fraud_prob * 100:.1f}%")
    with c2:
        st.metric("Processing Time", f"{result.get('processing_time_ms', 0):.0f} ms")
    with c3:
        violations = result.get("rule_violations", [])
        st.metric("Rule Violations", len(violations))

    # Gauge chart
    st.plotly_chart(
        create_gauge_chart(fraud_prob, "Fraud Probability"),
        use_container_width=True,
    )

    # Agent scores — cards
    agent_scores = result.get("agent_scores", {})
    _agent_labels = {
        "vibe_checker": ("Vibe Checker", "LGB + XGB ensemble"),
        "era_tracker": ("Era Tracker", "CatBoost behavioural"),
        "og_check": ("OG Check", "Rules + LGB hybrid"),
    }
    display_agents = {k: v for k, v in agent_scores.items() if k != "agent_ensemble"}
    if display_agents:
        st.plotly_chart(create_agent_bar_chart(display_agents), use_container_width=True)
        cols = st.columns(len(display_agents))
        for col, (key, score) in zip(cols, display_agents.items()):
            label, subtitle = _agent_labels.get(key, (key, ""))
            score_cls = "score-high" if score > 0.7 else "score-medium" if score > 0.4 else "score-low"
            with col:
                st.markdown(
                    f'<div class="agent-card">'
                    f'<div class="agent-card-title">{label}</div>'
                    f'<div class="agent-card-score {score_cls}">{score * 100:.1f}%</div>'
                    f'<div class="agent-card-desc">{subtitle}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Why was this flagged?
    st.markdown("#### Why was this flagged?")

    # SHAP chart
    top_features = result.get("top_features", [])
    if top_features:
        shap_fig = create_shap_chart(top_features)
        if shap_fig:
            st.plotly_chart(shap_fig, use_container_width=True)
        shap_ok = result.get("shap_available", False)
        llm_ok = result.get("llm_used", False)
        st.caption(f"SHAP: {'available' if shap_ok else 'unavailable'} · LLM: {'used' if llm_ok else 'template fallback'}")

    # Agent insights
    risk_factors = result.get("top_risk_factors", [])
    if risk_factors:
        st.markdown("**Risk Factors**")
        for rf in risk_factors:
            st.markdown(f"- {rf}")

    # LLM explanation
    text = result.get("natural_language_explanation") or result.get("explanation", "")
    if text:
        st.info(text)

    # Confidence & recommended action
    confidence = result.get("confidence_factors", [])
    if confidence:
        with st.expander("Confidence Assessment"):
            for cf in confidence:
                st.markdown(f"- {cf}")

    rec = result.get("recommended_action")
    if rec:
        st.markdown(f"**Recommended Action:** {rec}")

    if violations:
        with st.expander("Rule Violations"):
            for v in violations:
                st.warning(f"• {v}")


# =============================================================================
# SINGLE TRANSACTION VIEW
# =============================================================================

def single_transaction_view():
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Input")
        reference_payload = load_reference_payload()
        if reference_payload:
            st.caption(f"{len(reference_payload)} base fields loaded")

        with st.form("transaction_form"):
            base = reference_payload or {}

            amount = st.number_input("Amount ($)", min_value=0.0, max_value=100000.0,
                value=float(_get_default(base, "TransactionAmt", 150.0)), step=10.0)
            transaction_dt = st.number_input("TransactionDT", min_value=0.0,
                value=float(_get_default(base, "TransactionDT", 0.0)), step=3600.0)
            product_options = ["W", "H", "C", "S", "R"]
            product_default = str(_get_default(base, "ProductCD", "W"))
            product = st.selectbox("ProductCD", product_options,
                index=product_options.index(product_default) if product_default in product_options else 0)

            ca, cb = st.columns(2)
            with ca:
                card1 = st.number_input("card1", value=int(float(_get_default(base, "card1", 12345))), step=1)
                card2 = st.number_input("card2", value=float(_get_default(base, "card2", 321.0)), step=1.0)
                card3 = st.number_input("card3", value=float(_get_default(base, "card3", 150.0)), step=1.0)
                card5 = st.number_input("card5", value=float(_get_default(base, "card5", 226.0)), step=1.0)
                card_opts = ["visa", "mastercard", "discover", "american express", "amex", "debit", "credit", "charge card"]
                card4_def = str(_get_default(base, "card4", "visa"))
                card_type = st.selectbox("card4", card_opts,
                    index=card_opts.index(card4_def) if card4_def in card_opts else 0)
                card6_def = str(_get_default(base, "card6", "debit"))
                card6 = st.text_input("card6", card6_def)
            with cb:
                hour_def = int(_get_default(base, "hour", int((transaction_dt // 3600) % 24)))
                hour = st.slider("Hour of Day", 0, 23, hour_def)
                day = st.selectbox("Day of Week",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=int(_get_default(base, "day", 0)) % 7)
                addr1 = st.number_input("addr1", value=float(_get_default(base, "addr1", 325.0)), step=1.0)
                addr2 = st.number_input("addr2", value=float(_get_default(base, "addr2", 87.0)), step=1.0)

            with st.expander("M1-M9, emails", expanded=False):
                p_email = st.text_input("P_emaildomain", str(_get_default(base, "P_emaildomain", "gmail.com")))
                r_email = st.text_input("R_emaildomain", str(_get_default(base, "R_emaildomain", "gmail.com")))
                m_options = ["", "T", "F"]
                m_values = {}
                for m_key in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]:
                    dv = str(_get_default(base, m_key, ""))
                    m_values[m_key] = st.selectbox(m_key, m_options,
                        index=m_options.index(dv) if dv in m_options else 0, key=f"single_{m_key}")

            with st.expander("Identity / Device fields", expanded=False):
                device_type = st.text_input("DeviceType", str(_get_default(base, "DeviceType", "desktop")))
                device_info = st.text_input("DeviceInfo", str(_get_default(base, "DeviceInfo", "Windows")))
                identity_inputs = {}
                for id_col in [
                    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19",
                    "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27",
                    "id_28", "id_29", "id_30", "id_31", "id_32", "id_33", "id_34", "id_35",
                    "id_36", "id_37", "id_38",
                ]:
                    identity_inputs[id_col] = st.text_input(id_col,
                        str(_get_default(base, id_col, "")), key=f"single_{id_col}")

            with st.expander("Velocity fields", expanded=False):
                txn_count_1h = st.number_input("txn_count_1h", min_value=0,
                    value=int(float(_get_default(base, "txn_count_1h", 3))), step=1)
                amount_1h = st.number_input("amount_1h", min_value=0.0,
                    value=float(_get_default(base, "amount_1h", 200.0)), step=10.0)

            submitted = st.form_submit_button("Analyze Transaction", use_container_width=True)

    with col_right:
        st.markdown("#### Result")
        if submitted:
            overrides = {
                "TransactionID": int(float(_get_default(base, "TransactionID", 9999999))),
                "TransactionDT": float(transaction_dt),
                "TransactionAmt": amount, "ProductCD": product,
                "card1": card1, "card2": card2, "card3": card3,
                "card4": card_type, "card5": card5, "card6": card6,
                "addr1": addr1, "addr2": addr2,
                "hour": hour,
                "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day),
                "P_emaildomain": p_email, "R_emaildomain": r_email,
                "DeviceType": device_type, "DeviceInfo": device_info,
                "txn_count_1h": txn_count_1h, "amount_1h": amount_1h,
            }
            overrides.update(m_values)
            overrides.update(identity_inputs)
            transaction_data = build_full_payload(overrides)
            st.caption(f"Payload: {sum(1 for v in transaction_data.values() if v is not None)}/{len(transaction_data)} fields")
            result = run_with_agent_steps(transaction_data, use_explain=True)
            if result:
                display_results(result)
        else:
            st.info("Fill in the transaction details and click **Analyze Transaction**.")


# =============================================================================
# BATCH UPLOAD VIEW
# =============================================================================

def batch_upload_view():
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Upload")
        st.caption("Transaction CSV required; Identity CSV optional.")
        uploaded_tx = st.file_uploader("Transaction CSV", type=["csv"], key="batch_tx")
        uploaded_id = st.file_uploader("Identity CSV (optional)", type=["csv"], key="batch_id")

    if not uploaded_tx:
        return

    # --- Read only headers first to validate without loading all data ---
    tx_header = pd.read_csv(uploaded_tx, nrows=0)
    tx_valid, tx_missing = _validate_required_columns(tx_header, TRANSACTION_REQUIRED_COLUMNS)
    if not tx_valid:
        st.error(f"Transaction CSV missing columns: {tx_missing}")
        return
    has_labels = "isFraud" in tx_header.columns
    uploaded_tx.seek(0)  # reset for later read

    # Count rows without loading data (read only 1 column)
    row_count_df = pd.read_csv(uploaded_tx, usecols=["TransactionID"])
    total_rows = len(row_count_df)
    all_tx_ids = row_count_df["TransactionID"].values
    uploaded_tx.seek(0)

    # If labels present, get fraud indices for stratified sampling
    if has_labels:
        label_df = pd.read_csv(uploaded_tx, usecols=["TransactionID", "isFraud"])
        fraud_ids = set(label_df.loc[label_df["isFraud"] == 1, "TransactionID"].values)
        uploaded_tx.seek(0)
    else:
        label_df = None
        fraud_ids = set()

    id_header = None
    if uploaded_id:
        id_header = pd.read_csv(uploaded_id, nrows=0)
        id_valid, id_missing = _validate_required_columns(id_header, IDENTITY_REQUIRED_COLUMNS)
        if not id_valid:
            st.error(f"Identity CSV missing columns: {id_missing}")
            return
        uploaded_id.seek(0)

    with col_left:
        st.write(f"**{total_rows}** transactions detected")
        if has_labels:
            n_fraud_total = len(fraud_ids)
            st.caption(f"{n_fraud_total} fraud ({n_fraud_total/total_rows*100:.1f}%) / {total_rows - n_fraud_total} normal")
        analyze_all = st.checkbox("Analyze all (slow for large files)", value=False)

        if analyze_all:
            max_sample = min(total_rows, 2000)
            sample_ids = all_tx_ids[:max_sample]
            if analyze_all and total_rows > max_sample:
                st.warning(f"Capped at {max_sample} to avoid memory issues.")
        else:
            max_n = min(total_rows, 500)
            n_txns = st.slider("Sample size", min_value=5, max_value=max_n, value=min(20, max_n), step=5)
            if has_labels and label_df is not None:
                n_fraud = min(int(n_txns * 0.7), n_fraud_total)
                n_normal = n_txns - n_fraud
                fraud_arr = label_df.loc[label_df["isFraud"] == 1, "TransactionID"].values
                normal_arr = label_df.loc[label_df["isFraud"] == 0, "TransactionID"].values
                rng = __import__("numpy").random.RandomState(42)
                picked_fraud = rng.choice(fraud_arr, size=min(n_fraud, len(fraud_arr)), replace=False)
                picked_normal = rng.choice(normal_arr, size=min(n_normal, len(normal_arr)), replace=False)
                sample_ids = __import__("numpy").concatenate([picked_fraud, picked_normal])
                rng.shuffle(sample_ids)
                st.caption(f"{len(picked_fraud)} fraud + {len(picked_normal)} normal = {len(sample_ids)} sampled")
            else:
                rng = __import__("numpy").random.RandomState(42)
                sample_ids = rng.choice(all_tx_ids, size=min(n_txns, total_rows), replace=False)

        run_batch = st.button("Analyze Batch", use_container_width=True)

    with col_right:
        st.markdown("#### Results")
        if not run_batch:
            st.info("Configure sample and click **Analyze Batch**.")
            return

        # --- Now load ONLY the sampled rows via chunked reading ---
        sample_set = set(int(x) for x in sample_ids)
        with st.spinner(f"Loading {len(sample_set)} sampled rows…"):
            chunks = []
            for chunk in pd.read_csv(uploaded_tx, chunksize=5000):
                match = chunk[chunk["TransactionID"].isin(sample_set)]
                if not match.empty:
                    chunks.append(match)
                if sum(len(c) for c in chunks) >= len(sample_set):
                    break
            sample_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            del chunks

            if uploaded_id and not sample_df.empty:
                id_df = pd.read_csv(uploaded_id)
                sample_df = sample_df.merge(id_df, on="TransactionID", how="left")
                del id_df

        results = []
        progress = st.progress(0)
        status_text = st.empty()
        total = len(sample_df)

        for idx, (i, row) in enumerate(sample_df.iterrows()):
            status_text.text(f"Processing {idx + 1}/{total}…")
            payload = build_full_payload(row.to_dict())
            res = call_predict_api(payload)
            if res:
                entry = {
                    "TransactionID": row.get("TransactionID", i),
                    "fraud_prob": res.get("fraud_probability", 0.5),
                    "decision": res.get("decision", "REVIEW"),
                    "risk_level": res.get("risk_level", "MEDIUM"),
                }
                if has_labels:
                    entry["actual_fraud"] = int(row.get("isFraud", 0))
                results.append(entry)
            progress.progress((idx + 1) / total)

        status_text.empty()
        if not results:
            st.error("No results returned. Is the API running?")
            return

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("High Risk", len(results_df[results_df["risk_level"] == "HIGH"]))
        with c2:
            st.metric("Blocked", len(results_df[results_df["decision"] == "BLOCK"]))
        with c3:
            st.metric("Avg Fraud Prob", f"{results_df['fraud_prob'].mean() * 100:.1f}%")

        if has_labels and "actual_fraud" in results_df.columns:
            results_df["predicted_fraud"] = (results_df["fraud_prob"] >= 0.5).astype(int)
            correct = (results_df["predicted_fraud"] == results_df["actual_fraud"]).sum()
            total_r = len(results_df)
            acc = correct / total_r if total_r > 0 else 0
            tp = ((results_df["predicted_fraud"] == 1) & (results_df["actual_fraud"] == 1)).sum()
            fp = ((results_df["predicted_fraud"] == 1) & (results_df["actual_fraud"] == 0)).sum()
            fn = ((results_df["predicted_fraud"] == 0) & (results_df["actual_fraud"] == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                st.metric("Accuracy", f"{acc * 100:.1f}%")
            with ac2:
                st.metric("Precision", f"{prec * 100:.1f}%")
            with ac3:
                st.metric("Recall", f"{rec * 100:.1f}%")
            with ac4:
                st.metric("Correct / Total", f"{correct} / {total_r}")


# =============================================================================
# DEMO MODE — 6 curated scenarios
# =============================================================================

DEMO_SCENARIOS = [
    {"name": "Routine Purchase", "tag": "LEGIT",
     "desc": "Typical daytime purchase under $100 — expect APPROVE.", "tid": 3338576},
    {"name": "Recurring Subscription", "tag": "LEGIT",
     "desc": "Low-value recurring charge during business hours.", "tid": 3222562},
    {"name": "High-Value Midnight", "tag": "FRAUD",
     "desc": "Large transaction at 3 AM — real fraud case. Expect BLOCK.", "tid": 3499538},
    {"name": "Gift Card Fraud", "tag": "FRAUD",
     "desc": "Gift card purchase (ProductCD=C) — classic carding. Expect BLOCK.", "tid": 3250034},
    {"name": "Borderline — Unusual but Plausible", "tag": "REVIEW",
     "desc": "Medium amount, evening transaction — expect REVIEW.", "tid": 3424194},
    {"name": "Agent Disagreement", "tag": "MIXED",
     "desc": "High-value daytime purchase — agents disagree. Watch fusion resolve.", "tid": 3467581},
]


def demo_mode_view():
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Select Scenario")
        st.caption("Real transactions from training data — all features included.")

        selected_idx = st.selectbox(
            "Scenario",
            range(len(DEMO_SCENARIOS)),
            index=4,  # Default to Borderline (3424194)
            format_func=lambda i: f"{DEMO_SCENARIOS[i]['name']} ({DEMO_SCENARIOS[i]['tag']})",
        )
        scenario = DEMO_SCENARIOS[selected_idx]
        st.markdown(f"**{scenario['name']}** — {scenario['desc']}")
        st.caption(f"TransactionID: {scenario['tid']}")

        payload = load_transaction_by_id(scenario["tid"])
        if not payload:
            st.error(f"Could not load TransactionID {scenario['tid']}")
            return

        with st.expander("View payload", expanded=False):
            st.json(payload)

        run_demo = st.button("Run Demo Analysis", use_container_width=True)

    with col_right:
        st.markdown("#### Result")
        if run_demo:
            result = run_with_agent_steps(payload, use_explain=True)
            if result:
                display_results(result)
        else:
            st.info("Select a scenario and click **Run Demo Analysis**.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.title("Real-Time Fraud Detection with Agentic AI")
    mode = render_sidebar()

    if mode == "Single Transaction":
        single_transaction_view()
    elif mode == "Batch Upload":
        batch_upload_view()
    else:
        demo_mode_view()


if __name__ == "__main__":
    main()
