# =============================================================================
# FRAUD DETECTION STREAMLIT APP - Main Entry Point
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
LIGHTGBM_METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "lightgbm_metrics.json"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_TRANSACTION_PATH = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_PATH = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_PATH = DATA_DIR / "test_identity.csv"

TRANSACTION_REQUIRED_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

IDENTITY_REQUIRED_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_13",
    "id_14",
    "id_15",
    "id_16",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_23",
    "id_24",
    "id_25",
    "id_26",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_32",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]

# Custom CSS — professional grey palette, responsive
st.markdown("""
<style>
    /* ---- Global ---- */
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #e0e0e0;
        text-align: center; margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center; color: #9e9e9e; font-size: 1rem; margin-bottom: 1.5rem;
    }

    /* ---- Risk banners ---- */
    .risk-banner {
        padding: 1rem; border-radius: 8px; text-align: center;
        font-size: 1.4rem; font-weight: 700; margin-bottom: 1rem;
    }
    .risk-high   { background-color: #c62828; color: #fff; }
    .risk-medium { background-color: #f9a825; color: #212121; }
    .risk-low    { background-color: #2e7d32; color: #fff; }

    /* ---- Sidebar sections ---- */
    .sidebar-section-title {
        font-size: 0.85rem; font-weight: 600; color: #bdbdbd;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin: 1rem 0 0.4rem 0;
    }
    .health-dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px;
    }
    .health-green { background-color: #4caf50; }
    .health-red   { background-color: #ef5350; }

    /* ---- Agent loading steps ---- */
    .agent-step {
        padding: 0.3rem 0.6rem; margin: 0.15rem 0;
        border-radius: 4px; font-size: 0.85rem; color: #e0e0e0;
    }
    .agent-step-active  { background-color: #37474f; border-left: 3px solid #42a5f5; }
    .agent-step-done    { background-color: #263238; border-left: 3px solid #66bb6a; }
    .agent-step-waiting { background-color: #1e1e1e; border-left: 3px solid #616161; color: #757575; }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background-color: #2c2c2c; border: 1px solid #424242;
        border-radius: 8px; padding: 0.8rem;
    }
    div[data-testid="stMetric"] label { color: #9e9e9e !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e0e0e0 !important; }

    /* ---- Demo scenario cards ---- */
    .demo-card {
        background-color: #2c2c2c; border: 1px solid #424242;
        border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;
    }
    .demo-card-title { color: #e0e0e0; font-weight: 600; margin-bottom: 0.3rem; }
    .demo-card-desc  { color: #9e9e9e; font-size: 0.85rem; }
    .demo-tag {
        display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; margin-right: 0.3rem;
    }
    .tag-legit   { background-color: #1b5e20; color: #a5d6a7; }
    .tag-fraud   { background-color: #b71c1c; color: #ef9a9a; }
    .tag-review  { background-color: #e65100; color: #ffcc80; }
    .tag-disagree { background-color: #4a148c; color: #ce93d8; }

    /* ---- Responsive ---- */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important; flex: 100% !important; min-width: 100% !important;
        }
        .main-header { font-size: 1.6rem; }
        .risk-banner { font-size: 1.1rem; padding: 0.7rem; }
    }
    @media (max-width: 480px) {
        .main-header { font-size: 1.3rem; }
        div[data-testid="stMetric"] { padding: 0.5rem; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_api_health() -> dict:
    """Check API health. Returns {ok: bool, data: dict or error: str}."""
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
    """Call /api/v1/predict. Returns response dict or None."""
    try:
        r = requests.post(f"{API_URL}/api/v1/predict", json=transaction_data, timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"Prediction failed — API returned {r.status_code}: {r.text[:300]}")
        return None
    except requests.exceptions.ConnectionError:
        return _run_local_fallback(transaction_data)
    except requests.exceptions.Timeout:
        st.error("Request timed out after 30 s. The API may be overloaded.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def call_explain_api(transaction_data: dict) -> dict | None:
    """Call /api/v1/explain for full SHAP + LLM response."""
    try:
        r = requests.post(f"{API_URL}/api/v1/explain", json=transaction_data, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.error(f"Explain failed — API returned {r.status_code}: {r.text[:300]}")
        return None
    except requests.exceptions.ConnectionError:
        st.warning("API not reachable — falling back to predict-only (no SHAP chart).")
        return _run_local_fallback(transaction_data)
    except requests.exceptions.Timeout:
        st.error("Explain request timed out after 60 s.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def _run_local_fallback(transaction_data: dict) -> dict | None:
    """Run analysis locally when API is unreachable."""
    orchestrator = None
    try:
        from services.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        result = orchestrator.analyze(transaction_data)
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
        st.error(f"Local analysis also failed: {e}")
        return None
    finally:
        if orchestrator is not None:
            orchestrator.close()


def _sanitize_for_json(value):
    """Convert pandas/numpy scalars and NaNs to JSON-safe primitives."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


@st.cache_data
def load_reference_payload() -> dict:
    """Load one full-schema row to seed API requests with all feature keys."""
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
    """Overlay user inputs on top of a reference schema row."""
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


def _get_default(reference_payload: dict, field: str, fallback=None):
    value = reference_payload.get(field, fallback)
    return fallback if value is None else value


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> tuple[bool, list[str]]:
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def get_risk_css_class(risk_level: str) -> str:
    return {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}.get(risk_level, "")


@st.cache_data
def load_model_versions() -> dict:
    """Load model artifact timestamps for sidebar display."""
    models_dir = Path(__file__).resolve().parents[1] / "models"
    versions = {}
    for name, path in [
        ("Vibe LGB", models_dir / "vibe_lgb.txt"),
        ("Vibe XGB", models_dir / "vibe_xgb.json"),
        ("Era CatBoost", models_dir / "era_tracker_catboost.cbm"),
        ("OG LGB", models_dir / "og_check_lgb.txt"),
    ]:
        if path.exists():
            stat = path.stat()
            versions[name] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        else:
            versions[name] = "missing"
    return versions


@st.cache_data
def load_lightgbm_status() -> dict:
    status = {"roc_auc": 0.0, "test_predicted_fraud_rate": 0.0, "best_threshold": 0.5}
    if LIGHTGBM_METRICS_PATH.exists():
        try:
            with LIGHTGBM_METRICS_PATH.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            rs = payload.get("runtime_eval_snapshot", {})
            ts = payload.get("test_inference_snapshot", {})
            status["roc_auc"] = float(rs.get("roc_auc", 0))
            status["test_predicted_fraud_rate"] = float(ts.get("predicted_fraud_rate", 0))
            status["best_threshold"] = float(payload.get("best_threshold", 0.5))
        except Exception:
            pass
    return status


# =============================================================================
# CHART BUILDERS
# =============================================================================

CHART_BG = "rgba(0,0,0,0)"
GREY_FONT = {"color": "#9e9e9e"}


def create_gauge_chart(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"size": 18, "color": "#e0e0e0"}},
        number={"suffix": "%", "font": {"size": 36, "color": "#e0e0e0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#616161"},
            "bar": {"color": "#78909c"},
            "bgcolor": "#2c2c2c",
            "steps": [
                {"range": [0, 40], "color": "#1b5e20"},
                {"range": [40, 70], "color": "#e65100"},
                {"range": [70, 100], "color": "#b71c1c"},
            ],
            "threshold": {"line": {"color": "#ef5350", "width": 3}, "thickness": 0.75, "value": 70},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=30, l=40, r=40),
                      paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
    return fig


def create_agent_scores_chart(scores: dict) -> go.Figure:
    agents = list(scores.keys())
    values = [scores[a] * 100 for a in agents]
    palette = ["#78909c", "#90a4ae", "#b0bec5", "#607d8b", "#546e7a"]
    colors = [palette[i % len(palette)] for i in range(len(agents))]

    fig = go.Figure(data=[go.Bar(
        x=agents, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="auto",
        textfont={"color": "#e0e0e0"},
    )])
    fig.update_layout(
        title={"text": "Agent Score Breakdown", "font": {"color": "#bdbdbd", "size": 16}},
        xaxis_title="Agent", yaxis_title="Risk Score (%)",
        yaxis=dict(range=[0, 100], gridcolor="#424242"),
        xaxis=dict(color="#9e9e9e"), height=300,
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=GREY_FONT,
    )
    return fig


def create_shap_chart(top_features: list[dict]) -> go.Figure | None:
    """Horizontal bar chart — red = fraud-pushing, blue = legit-pushing. Top 10."""
    if not top_features:
        return None

    sorted_feats = sorted(top_features, key=lambda f: abs(f.get("shap_value", 0)), reverse=True)[:10]
    sorted_feats = list(reversed(sorted_feats))  # highest absolute at top

    names = [f.get("feature_name") or f.get("raw_name", "?") for f in sorted_feats]
    shap_vals = [f.get("shap_value", 0) for f in sorted_feats]
    colors = ["#c62828" if v > 0 else "#1565c0" for v in shap_vals]

    fig = go.Figure(data=[go.Bar(
        x=shap_vals, y=names, orientation="h", marker_color=colors,
        text=[f"{v:+.4f}" for v in shap_vals], textposition="outside",
        textfont={"color": "#bdbdbd", "size": 11},
    )])
    fig.update_layout(
        title={"text": "SHAP Feature Impact (Top 10)", "font": {"color": "#bdbdbd", "size": 16}},
        xaxis_title="SHAP Value",
        xaxis=dict(gridcolor="#424242", zerolinecolor="#616161", color="#9e9e9e"),
        yaxis=dict(color="#9e9e9e"),
        height=max(320, len(sorted_feats) * 36 + 80),
        margin=dict(l=180, r=80, t=50, b=40),
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=GREY_FONT,
    )
    fig.add_annotation(
        x=0.98, y=1.12, xref="paper", yref="paper",
        text='<span style="color:#c62828">■</span> Fraud-pushing &nbsp; '
             '<span style="color:#1565c0">■</span> Legit-pushing',
        showarrow=False, font=dict(size=11, color="#9e9e9e"),
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    """Render sidebar with health check, weights, thresholds, model versions."""
    st.sidebar.title("⚙️ Settings")

    mode = st.sidebar.radio("Analysis Mode", ["Single Transaction", "Batch Upload", "Demo Mode"])
    st.sidebar.markdown("---")

    # ---- API Health ----
    st.sidebar.markdown('<p class="sidebar-section-title">API Status</p>', unsafe_allow_html=True)
    health = check_api_health()
    if health["ok"]:
        d = health["data"]
        st.sidebar.markdown(
            f'<span class="health-dot health-green"></span> **Online** — v{d.get("version", "?")} — up {d.get("uptime_seconds", 0):.0f}s',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f'<span class="health-dot health-red"></span> **Offline** — {health["error"]}',
            unsafe_allow_html=True,
        )

    # ---- Fusion Weights ----
    st.sidebar.markdown('<p class="sidebar-section-title">Fusion Weights</p>', unsafe_allow_html=True)
    st.sidebar.markdown(
        "Vibe < 0.8: **60%** Vibe · **25%** Era · **15%** OG\n\n"
        "Vibe ≥ 0.8: **100%** Vibe (high-confidence override)"
    )

    # ---- Decision Thresholds ----
    st.sidebar.markdown('<p class="sidebar-section-title">Decision Thresholds</p>', unsafe_allow_html=True)
    st.sidebar.markdown("≥ 0.70 → **BLOCK** · ≥ 0.40 → **REVIEW** · < 0.40 → **APPROVE**")

    # ---- Latency (real measured) ----
    st.sidebar.markdown('<p class="sidebar-section-title">Latency (measured)</p>', unsafe_allow_html=True)
    st.sidebar.markdown("p50 **~620ms** · p95 **~780ms** · p99 **~880ms**")
    st.sidebar.caption("Single uvicorn worker, local dev. Target: 200–300ms in production.")

    # ---- Model Artifacts ----
    st.sidebar.markdown('<p class="sidebar-section-title">Model Artifacts</p>', unsafe_allow_html=True)
    versions = load_model_versions()
    for name, ts in versions.items():
        icon = "✅" if ts != "missing" else "❌"
        st.sidebar.caption(f"{icon} {name}: {ts}")

    lgb = load_lightgbm_status()
    st.sidebar.metric("Ensemble ROC-AUC", f"{lgb['roc_auc']:.4f}")

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
    """Call the API with per-agent loading indicators."""
    step_container = st.container()
    with step_container:
        placeholders = []
        for name, _ in _AGENT_STEPS:
            ph = st.empty()
            ph.markdown(
                f'<div class="agent-step agent-step-waiting">⏳ {name} — waiting</div>',
                unsafe_allow_html=True,
            )
            placeholders.append(ph)

    # Animate through steps
    for i, (name, desc) in enumerate(_AGENT_STEPS):
        placeholders[i].markdown(
            f'<div class="agent-step agent-step-active">⚡ {name} — {desc}</div>',
            unsafe_allow_html=True,
        )
        if i < len(_AGENT_STEPS) - 1:
            time.sleep(0.25)

    result = call_explain_api(transaction_data) if use_explain else call_predict_api(transaction_data)

    for i, (name, _) in enumerate(_AGENT_STEPS):
        placeholders[i].markdown(
            f'<div class="agent-step agent-step-done">✓ {name} — done</div>',
            unsafe_allow_html=True,
        )

    return result


# =============================================================================
# DISPLAY RESULTS (with SHAP chart)
# =============================================================================

def display_results(result: dict):
    """Display analysis results with SHAP chart, agent scores, explanation."""
    risk_level = result.get("risk_level", "MEDIUM")
    css_class = get_risk_css_class(risk_level)
    decision = result.get("decision", "REVIEW")

    st.markdown(
        f'<div class="risk-banner {css_class}">Risk: {risk_level} &nbsp;|&nbsp; Decision: {decision}</div>',
        unsafe_allow_html=True,
    )

    # Metric row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Fraud Probability", f"{result.get('fraud_probability', 0) * 100:.1f}%")
    with c2:
        st.metric("Processing Time", f"{result.get('processing_time_ms', 0):.0f} ms")
    with c3:
        violations = result.get("rule_violations", [])
        st.metric("Rule Violations", len(violations))

    # Gauge
    st.plotly_chart(
        create_gauge_chart(result.get("fraud_probability", 0.5), "Fraud Probability"),
        use_container_width=True,
    )

    # SHAP horizontal bar chart
    top_features = result.get("top_features", [])
    if top_features:
        shap_fig = create_shap_chart(top_features)
        if shap_fig:
            st.plotly_chart(shap_fig, use_container_width=True)
        shap_ok = result.get("shap_available", False)
        llm_ok = result.get("llm_used", False)
        st.caption(f"SHAP: {'available' if shap_ok else 'unavailable'} · LLM: {'used' if llm_ok else 'template fallback'}")

    # Agent scores
    if result.get("agent_scores"):
        st.plotly_chart(create_agent_scores_chart(result["agent_scores"]), use_container_width=True)

    # Explanation
    st.subheader("Explanation")
    text = result.get("natural_language_explanation") or result.get("explanation", "No explanation available")
    st.info(text)

    # Risk factors
    risk_factors = result.get("top_risk_factors", [])
    if risk_factors:
        st.subheader("Risk Factors")
        for rf in risk_factors:
            st.markdown(f"- {rf}")

    # Confidence assessment
    confidence = result.get("confidence_factors", [])
    if confidence:
        with st.expander("Confidence Assessment"):
            for cf in confidence:
                st.markdown(f"- {cf}")

    # Recommended action
    rec = result.get("recommended_action")
    if rec:
        st.markdown(f"**Recommended Action:** {rec}")

    # Rule violations
    if violations:
        st.subheader("Rule Violations")
        for v in violations:
            st.warning(f"• {v}")


# =============================================================================
# SINGLE TRANSACTION VIEW
# =============================================================================

def single_transaction_view():
    st.header("Transaction Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Details")

        reference_payload = load_reference_payload()
        if reference_payload:
            st.caption(f"Full-feature mode: {len(reference_payload)} base fields loaded")
        else:
            st.warning("Reference dataset not found. Requests will include only entered fields.")

        with st.form("transaction_form"):
            base = reference_payload or {}

            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0,
                value=float(_get_default(base, "TransactionAmt", 150.0)), step=10.0)
            transaction_dt = st.number_input("TransactionDT (timedelta seconds)", min_value=0.0,
                value=float(_get_default(base, "TransactionDT", 0.0)), step=3600.0)
            product_options = ["W", "H", "C", "S", "R"]
            product_default = str(_get_default(base, "ProductCD", "W"))
            product = st.selectbox("Product Code", product_options,
                index=product_options.index(product_default) if product_default in product_options else 0)

            col_a, col_b = st.columns(2)
            with col_a:
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

            with col_b:
                hour_def = int(_get_default(base, "hour", int((transaction_dt // 3600) % 24)))
                hour = st.slider("Hour of Day", 0, 23, hour_def)
                day = st.selectbox("Day of Week",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=int(_get_default(base, "day", 0)) % 7)
                addr1 = st.number_input("addr1", value=float(_get_default(base, "addr1", 325.0)), step=1.0)
                addr2 = st.number_input("addr2", value=float(_get_default(base, "addr2", 87.0)), step=1.0)

            with st.expander("M1-M9, emails", expanded=True):
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

            submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

    with col2:
        st.subheader("Analysis Result")

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

            non_null = sum(1 for v in transaction_data.values() if v is not None)
            st.caption(f"Payload: {non_null}/{len(transaction_data)} fields populated")

            result = run_with_agent_steps(transaction_data, use_explain=True)
            if result:
                display_results(result)
        else:
            st.info("Fill in the transaction details and click 'Analyze Transaction'")


# =============================================================================
# BATCH UPLOAD VIEW
# =============================================================================

def batch_upload_view():
    st.header("Batch Analysis")
    st.caption("Upload test-style CSVs. Transaction CSV required; identity CSV optional.")

    uploaded_tx = st.file_uploader("Transaction CSV", type=["csv"], key="batch_tx")
    uploaded_id = st.file_uploader("Identity CSV (optional)", type=["csv"], key="batch_id")

    if not uploaded_tx:
        return

    tx_df = pd.read_csv(uploaded_tx)
    tx_valid, tx_missing = _validate_required_columns(tx_df, TRANSACTION_REQUIRED_COLUMNS)
    if not tx_valid:
        st.error(f"Transaction CSV missing columns: {tx_missing}")
        return

    if uploaded_id:
        id_df = pd.read_csv(uploaded_id)
        id_valid, id_missing = _validate_required_columns(id_df, IDENTITY_REQUIRED_COLUMNS)
        if not id_valid:
            st.error(f"Identity CSV missing columns: {id_missing}")
            return
        df = tx_df.merge(id_df, on="TransactionID", how="left")
    else:
        st.warning("No identity CSV — predictions use transaction-only features.")
        df = tx_df

    st.write(f"Loaded **{len(df)}** transactions")
    st.dataframe(df.head())

    if st.button("🔍 Analyze All", use_container_width=True):
        results = []
        progress = st.progress(0)
        status_text = st.empty()

        for i, row in df.iterrows():
            status_text.text(f"Processing {i + 1}/{len(df)}…")
            payload = build_full_payload(row.to_dict())
            res = call_predict_api(payload)
            if res:
                results.append({
                    "TransactionID": row.get("TransactionID", i),
                    "fraud_prob": res.get("fraud_probability", 0.5),
                    "decision": res.get("decision", "REVIEW"),
                    "risk_level": res.get("risk_level", "MEDIUM"),
                })
            progress.progress((i + 1) / len(df))

        status_text.empty()
        if not results:
            st.error("No results returned. Is the API running?")
            return

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("High Risk", len(results_df[results_df["risk_level"] == "HIGH"]))
        with c2:
            st.metric("Blocked", len(results_df[results_df["decision"] == "BLOCK"]))
        with c3:
            st.metric("Avg Fraud Prob", f"{results_df['fraud_prob'].mean() * 100:.1f}%")


# =============================================================================
# DEMO MODE — 6 curated scenarios
# =============================================================================

DEMO_SCENARIOS = [
    {
        "name": "Routine Purchase",
        "tag": "legit", "tag_label": "LEGIT",
        "description": "Typical daytime Visa purchase under $100. All fields normal — expect APPROVE with low scores across all agents.",
        "overrides": {
            "TransactionAmt": 67.50, "ProductCD": "W", "card4": "visa", "card6": "debit",
            "hour": 14, "day": 2, "addr1": 325.0, "addr2": 87.0,
            "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
            "M1": "T", "M2": "T", "M3": "T", "M4": "T", "M5": "T", "M6": "T",
            "DeviceType": "desktop", "DeviceInfo": "Windows",
            "txn_count_1h": 1, "amount_1h": 67.50,
        },
    },
    {
        "name": "Recurring Subscription",
        "tag": "legit", "tag_label": "LEGIT",
        "description": "Low-value recurring charge from a known email domain during business hours. Classic legitimate pattern.",
        "overrides": {
            "TransactionAmt": 12.99, "ProductCD": "S", "card4": "mastercard", "card6": "credit",
            "hour": 10, "day": 1, "addr1": 200.0, "addr2": 87.0,
            "P_emaildomain": "outlook.com", "R_emaildomain": "outlook.com",
            "M1": "T", "M2": "T", "M3": "T", "M4": "T", "M5": "T", "M6": "T",
            "DeviceType": "mobile", "DeviceInfo": "iOS Device",
            "txn_count_1h": 0, "amount_1h": 0.0,
        },
    },
    {
        "name": "High-Value Midnight Transaction",
        "tag": "fraud", "tag_label": "FRAUD",
        "description": "Large transaction at 3 AM with mismatched emails and high velocity. Expect BLOCK — all agents should flag this.",
        "overrides": {
            "TransactionAmt": 8500.00, "ProductCD": "H", "card4": "visa", "card6": "debit",
            "hour": 3, "day": 5, "addr1": 0.0, "addr2": 0.0,
            "P_emaildomain": "protonmail.com", "R_emaildomain": "yahoo.com",
            "M1": "F", "M2": "F", "M3": "T", "M5": "F", "M6": "F",
            "DeviceType": "desktop", "DeviceInfo": "Linux",
            "txn_count_1h": 12, "amount_1h": 25000.0,
        },
    },
    {
        "name": "Gift Card Fraud Pattern",
        "tag": "fraud", "tag_label": "FRAUD",
        "description": "Multiple gift card purchases in rapid succession with missing identity data. Classic carding pattern. Expect BLOCK.",
        "overrides": {
            "TransactionAmt": 500.00, "ProductCD": "C", "card4": "discover", "card6": "credit",
            "hour": 2, "day": 0, "addr1": 0.0, "addr2": 0.0,
            "P_emaildomain": "", "R_emaildomain": "",
            "DeviceType": "", "DeviceInfo": "",
            "txn_count_1h": 8, "amount_1h": 4000.0,
        },
    },
    {
        "name": "Borderline — Unusual but Plausible",
        "tag": "review", "tag_label": "REVIEW",
        "description": "Medium amount, evening, slightly elevated velocity. Not clearly fraud or legit — expect REVIEW. Tests the blend zone (score 0.4–0.7).",
        "overrides": {
            "TransactionAmt": 350.00, "ProductCD": "W", "card4": "mastercard", "card6": "credit",
            "hour": 21, "day": 4, "addr1": 150.0, "addr2": 87.0,
            "P_emaildomain": "yahoo.com", "R_emaildomain": "",
            "M1": "T", "M2": "T", "M3": "F", "M4": "T", "M5": "F", "M6": "T",
            "DeviceType": "mobile", "DeviceInfo": "Android",
            "txn_count_1h": 5, "amount_1h": 1200.0,
        },
    },
    {
        "name": "Agent Disagreement — Mixed Signals",
        "tag": "disagree", "tag_label": "DISAGREEMENT",
        "description": "Designed to split the agents: high amount triggers OG rules, but normal time/device keeps Era calm. "
                       "Watch how the fusion formula resolves conflicting signals via dynamic weighting.",
        "overrides": {
            "TransactionAmt": 2500.00, "ProductCD": "W", "card4": "visa", "card6": "debit",
            "hour": 15, "day": 3, "addr1": 325.0, "addr2": 87.0,
            "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
            "M1": "T", "M2": "T", "M3": "T", "M4": "T", "M5": "T", "M6": "F",
            "DeviceType": "desktop", "DeviceInfo": "Windows",
            "txn_count_1h": 3, "amount_1h": 3500.0,
        },
    },
]


def demo_mode_view():
    st.header("Demo Mode")
    st.markdown("Six curated scenarios demonstrating different fraud patterns and system behaviour.")

    for scenario in DEMO_SCENARIOS:
        tag_class = f"tag-{scenario['tag']}"
        st.markdown(
            f'<div class="demo-card">'
            f'<div class="demo-card-title">{scenario["name"]} '
            f'<span class="demo-tag {tag_class}">{scenario["tag_label"]}</span></div>'
            f'<div class="demo-card-desc">{scenario["description"]}</div></div>',
            unsafe_allow_html=True,
        )

    selected_idx = st.selectbox(
        "Choose scenario",
        range(len(DEMO_SCENARIOS)),
        format_func=lambda i: f"{DEMO_SCENARIOS[i]['name']} ({DEMO_SCENARIOS[i]['tag_label']})",
    )
    scenario = DEMO_SCENARIOS[selected_idx]

    with st.expander("View payload", expanded=False):
        st.json(scenario["overrides"])

    if st.button("🔍 Run Demo Analysis", use_container_width=True):
        payload = build_full_payload(scenario["overrides"])
        result = run_with_agent_steps(payload, use_explain=True)
        if result:
            display_results(result)


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🔍 Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-agent ensemble scoring · SHAP explainability · LLM explanations</p>',
        unsafe_allow_html=True,
    )
    mode = render_sidebar()

    if mode == "Single Transaction":
        single_transaction_view()
    elif mode == "Batch Upload":
        batch_upload_view()
    else:
        demo_mode_view()


if __name__ == "__main__":
    main()
