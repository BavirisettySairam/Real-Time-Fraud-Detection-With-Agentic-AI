# =============================================================================
# FRAUD DETECTION STREAMLIT APP - Main Entry Point
# =============================================================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🔍 Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
GNN_METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "gnn_metrics.json"
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high { background-color: #ff6b6b; padding: 0.5rem 1rem; border-radius: 5px; color: white; }
    .risk-medium { background-color: #feca57; padding: 0.5rem 1rem; border-radius: 5px; color: black; }
    .risk-low { background-color: #5f27cd; padding: 0.5rem 1rem; border-radius: 5px; color: white; }
    .stMetric > div { background-color: #f8f9fa; padding: 1rem; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def call_api(transaction_data: dict) -> dict:
    """Call the fraud detection API"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json=transaction_data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        # Fallback: run local analysis
        return run_local_analysis(transaction_data)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


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
    """Load one full-schema transaction payload to seed API requests with all feature keys."""
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
    """Create a full-feature payload by overlaying user inputs on top of a reference schema row."""
    base = dict(load_reference_payload())
    for k, v in overrides.items():
        base[k] = _sanitize_for_json(v)

    # Ensure competition-required transaction/identity fields always exist in the payload.
    for col in TRANSACTION_REQUIRED_COLUMNS + IDENTITY_REQUIRED_COLUMNS:
        base.setdefault(col, None)

    # Guarantee core required fields for API validation.
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


def run_local_analysis(transaction_data: dict) -> dict:
    """Run analysis locally if API is not available"""
    orchestrator = None
    try:
        from services.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        result = orchestrator.analyze(transaction_data)
        
        return {
            "transaction_id": transaction_data.get("TransactionID"),
            "fraud_probability": result.get("final_score", 0.5),
            "decision": result.get("final_decision", "REVIEW"),
            "risk_level": "HIGH" if result.get("final_score", 0) > 0.7 else "MEDIUM" if result.get("final_score", 0) > 0.4 else "LOW",
            "explanation": result.get("explanation", "Analysis complete"),
            "agent_scores": {
                "vibe_checker": result.get("vibe_score", 0.5),
                "agent_ensemble": result.get("agent_ensemble_score", 0.5),
                "era_tracker": result.get("era_score", 0.5),
                "og_check": result.get("og_score", 0.5)
            },
            "rule_violations": result.get("og_violations", []),
            "processing_time_ms": result.get("processing_time_ms", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        st.error(f"Local analysis failed: {e}")
        return None
    finally:
        if orchestrator is not None:
            orchestrator.close()


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level"""
    colors = {
        "HIGH": "#ff6b6b",
        "MEDIUM": "#feca57",
        "LOW": "#5f27cd"
    }
    return colors.get(risk_level, "#6c757d")


def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Create a gauge chart for fraud probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 20}},
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#5f27cd"},
                {'range': [40, 70], 'color': "#feca57"},
                {'range': [70, 100], 'color': "#ff6b6b"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50))
    return fig


def create_agent_scores_chart(scores: dict) -> go.Figure:
    """Create bar chart for agent scores"""
    agents = list(scores.keys())
    values = [scores[a] * 100 for a in agents]

    palette = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#17becf']
    colors = [palette[i % len(palette)] for i in range(len(agents))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=agents,
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Model and Agent Score Breakdown",
        xaxis_title="Agent",
        yaxis_title="Risk Score (%)",
        yaxis=dict(range=[0, 100]),
        height=300
    )
    
    return fig


@st.cache_data
def load_gnn_archive_status() -> dict:
    status = {
        "runtime_enabled": False,
        "roc_auc": 0.62,
    }
    if GNN_METRICS_PATH.exists():
        try:
            with GNN_METRICS_PATH.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            status["runtime_enabled"] = bool(payload.get("is_used_in_runtime", False))
            status["roc_auc"] = float(payload.get("validation_roc_auc", status["roc_auc"]))
        except Exception:
            pass
    return status


@st.cache_data
def load_lightgbm_status() -> dict:
    status = {
        "roc_auc": 0.0,
        "test_predicted_fraud_rate": 0.0,
        "best_threshold": 0.5,
    }
    if LIGHTGBM_METRICS_PATH.exists():
        try:
            with LIGHTGBM_METRICS_PATH.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)

            runtime_snapshot = payload.get("runtime_eval_snapshot", {})
            test_snapshot = payload.get("test_inference_snapshot", {})

            status["roc_auc"] = float(runtime_snapshot.get("roc_auc", status["roc_auc"]))
            status["test_predicted_fraud_rate"] = float(
                test_snapshot.get("predicted_fraud_rate", status["test_predicted_fraud_rate"])
            )
            status["best_threshold"] = float(payload.get("best_threshold", status["best_threshold"]))
        except Exception:
            pass
    return status


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    gnn_status = load_gnn_archive_status()
    lightgbm_status = load_lightgbm_status()

    # Header
    st.markdown('<h1 class="main-header">🔍 Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #6c757d;">Ensemble fraud scoring with Vibe Checker, Era Tracker, OG Check, and The Yapper</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Transaction", "Batch Upload", "Demo Mode"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Primary Model", "LightGBM", delta=f"ROC-AUC {lightgbm_status['roc_auc']:.3f}")
    st.sidebar.metric("GNN (Archived)", f"AUC {gnn_status['roc_auc']:.3f}")
    st.sidebar.metric("Avg Latency", "<100ms")
    st.sidebar.metric("Predicted Test Fraud", f"{lightgbm_status['test_predicted_fraud_rate']:.2%}")
    st.sidebar.metric("Decision Threshold", f"{lightgbm_status['best_threshold']:.3f}")
    st.sidebar.caption(f"GNN runtime enabled: {gnn_status['runtime_enabled']}")
    
    # Main content based on mode
    if mode == "Single Transaction":
        single_transaction_view()
    elif mode == "Batch Upload":
        batch_upload_view()
    else:
        demo_mode_view()


def single_transaction_view():
    """Single transaction analysis view"""
    st.header("📝 Transaction Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Transaction Details")

        reference_payload = load_reference_payload()
        if reference_payload:
            st.caption(f"Full-feature mode active: {len(reference_payload)} base fields loaded from data/*.csv")
        else:
            st.warning("Reference dataset row not found. Requests will include only fields entered here.")
        
        with st.form("transaction_form"):
            base = reference_payload or {}

            # Basic info
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(_get_default(base, "TransactionAmt", 150.0)),
                step=10.0
            )

            transaction_dt = st.number_input(
                "TransactionDT (timedelta seconds)",
                min_value=0.0,
                value=float(_get_default(base, "TransactionDT", 0.0)),
                step=3600.0
            )
            
            product = st.selectbox(
                "Product Code",
                ["W", "H", "C", "S", "R"],
                index=max(0, ["W", "H", "C", "S", "R"].index(str(_get_default(base, "ProductCD", "W")))) if str(_get_default(base, "ProductCD", "W")) in ["W", "H", "C", "S", "R"] else 0
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                card1 = st.number_input("card1", value=int(float(_get_default(base, "card1", 12345))), step=1)
                card2 = st.number_input("card2", value=float(_get_default(base, "card2", 321.0)), step=1.0)
                card3 = st.number_input("card3", value=float(_get_default(base, "card3", 150.0)), step=1.0)
                card5 = st.number_input("card5", value=float(_get_default(base, "card5", 226.0)), step=1.0)

                card_type_options = ["visa", "mastercard", "discover", "american express", "amex", "debit", "credit", "charge card"]
                card4_default = str(_get_default(base, "card4", "visa"))
                card_type = st.selectbox("card4", card_type_options, index=card_type_options.index(card4_default) if card4_default in card_type_options else 0)

                card6_default = str(_get_default(base, "card6", "debit"))
                card6 = st.text_input("card6", card6_default)
            
            with col_b:
                hour_default = int(_get_default(base, "hour", int((transaction_dt // 3600) % 24)))
                hour = st.slider("Hour of Day", 0, 23, hour_default)
                day = st.selectbox("Day of Week", 
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=int(_get_default(base, "day", 0)) % 7
                )

                addr1 = st.number_input("addr1", value=float(_get_default(base, "addr1", 325.0)), step=1.0)
                addr2 = st.number_input("addr2", value=float(_get_default(base, "addr2", 87.0)), step=1.0)
            
            with st.expander("Transaction categorical fields (M1-M9, emails)", expanded=True):
                p_email_default = str(_get_default(base, "P_emaildomain", "gmail.com"))
                r_email_default = str(_get_default(base, "R_emaildomain", "gmail.com"))
                email_domain = st.text_input("P_emaildomain", p_email_default)
                r_email_domain = st.text_input("R_emaildomain", r_email_default)

                m_options = ["", "T", "F"]
                m_values = {}
                for m_key in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]:
                    default_value = str(_get_default(base, m_key, ""))
                    m_values[m_key] = st.selectbox(
                        m_key,
                        m_options,
                        index=m_options.index(default_value) if default_value in m_options else 0,
                        key=f"single_{m_key}",
                    )

            with st.expander("Identity categorical fields (Device + id_12..id_38)", expanded=False):
                device_type = st.text_input("DeviceType", str(_get_default(base, "DeviceType", "desktop")))
                device_info = st.text_input("DeviceInfo", str(_get_default(base, "DeviceInfo", "Windows")))

                identity_inputs = {}
                for id_col in [
                    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19",
                    "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27",
                    "id_28", "id_29", "id_30", "id_31", "id_32", "id_33", "id_34", "id_35",
                    "id_36", "id_37", "id_38",
                ]:
                    identity_inputs[id_col] = st.text_input(
                        id_col,
                        str(_get_default(base, id_col, "")),
                        key=f"single_{id_col}",
                    )

            with st.expander("Velocity fields", expanded=False):
                txn_count_1h = st.number_input("txn_count_1h", min_value=0, value=int(float(_get_default(base, "txn_count_1h", 3))), step=1)
                amount_1h = st.number_input("amount_1h", min_value=0.0, value=float(_get_default(base, "amount_1h", 200.0)), step=10.0)
            
            submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
    
    with col2:
        st.subheader("Analysis Result")
        
        if submitted:
            # Overlay entered values on top of full-schema reference payload.
            overrides = {
                "TransactionID": int(float(_get_default(base, "TransactionID", 9999999))),
                "TransactionDT": float(transaction_dt),
                "TransactionAmt": amount,
                "ProductCD": product,
                "card1": card1,
                "card2": card2,
                "card3": card3,
                "card4": card_type,
                "card5": card5,
                "card6": card6,
                "addr1": addr1,
                "addr2": addr2,
                "hour": hour,
                "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day),
                "P_emaildomain": email_domain,
                "R_emaildomain": r_email_domain,
                "DeviceType": device_type,
                "DeviceInfo": device_info,
                "txn_count_1h": txn_count_1h,
                "amount_1h": amount_1h,
            }
            overrides.update(m_values)
            overrides.update(identity_inputs)
            transaction_data = build_full_payload(overrides)

            non_null_fields = sum(1 for value in transaction_data.values() if value is not None)
            st.caption(f"Sending payload fields: {non_null_fields}/{len(transaction_data)}")
            
            with st.spinner("🔄 Analyzing transaction..."):
                result = call_api(transaction_data)
            
            if result:
                display_results(result)
        else:
            st.info("👆 Fill in the transaction details and click 'Analyze Transaction'")


def display_results(result: dict):
    """Display analysis results"""
    # Risk level banner
    risk_level = result.get("risk_level", "MEDIUM")
    risk_color = get_risk_color(risk_level)
    
    st.markdown(
        f'<div style="background-color: {risk_color}; padding: 1rem; border-radius: 10px; '
        f'text-align: center; color: white; font-size: 1.5rem; font-weight: bold;">'
        f'Risk Level: {risk_level} | Decision: {result.get("decision", "REVIEW")}</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fraud Probability", f"{result.get('fraud_probability', 0)*100:.1f}%")
    
    with col2:
        st.metric("Processing Time", f"{result.get('processing_time_ms', 0):.1f}ms")
    
    with col3:
        violations = result.get('rule_violations', [])
        st.metric("Rule Violations", len(violations))
    
    # Gauge chart
    st.plotly_chart(
        create_gauge_chart(result.get('fraud_probability', 0.5), "Fraud Probability"),
        use_container_width=True
    )
    
    # Agent scores
    if result.get('agent_scores'):
        st.plotly_chart(
            create_agent_scores_chart(result['agent_scores']),
            use_container_width=True
        )
    
    # Explanation
    st.subheader("📝 Explanation")
    st.info(result.get('explanation', 'No explanation available'))
    
    # Rule violations
    if violations:
        st.subheader("⚠️ Rule Violations")
        for v in violations:
            st.warning(f"• {v}")


def batch_upload_view():
    """Batch upload view"""
    st.header("📁 Batch Analysis")

    st.caption("Upload test-style files: transaction CSV is required; identity CSV is optional.")
    uploaded_tx_file = st.file_uploader(
        "Upload transaction CSV (test_transaction-like)",
        type=['csv'],
        key="batch_tx_file"
    )
    uploaded_id_file = st.file_uploader(
        "Upload identity CSV (test_identity-like, optional)",
        type=['csv'],
        key="batch_id_file"
    )

    if uploaded_tx_file:
        tx_df = pd.read_csv(uploaded_tx_file)
        tx_valid, tx_missing = _validate_required_columns(tx_df, TRANSACTION_REQUIRED_COLUMNS)
        if not tx_valid:
            st.error(f"Transaction CSV missing required columns: {tx_missing}")
            return

        if uploaded_id_file:
            id_df = pd.read_csv(uploaded_id_file)
            id_valid, id_missing = _validate_required_columns(id_df, IDENTITY_REQUIRED_COLUMNS)
            if not id_valid:
                st.error(f"Identity CSV missing required columns: {id_missing}")
                return
            df = tx_df.merge(id_df, on="TransactionID", how="left")
        else:
            st.warning("Identity CSV not provided; prediction will run with transaction-only columns.")
            df = tx_df

        st.write(f"Loaded {len(df)} merged transactions")
        st.dataframe(df.head())

        reference_payload = load_reference_payload()
        if reference_payload:
            st.caption(f"Batch full-feature expansion enabled using {len(reference_payload)} base fields.")
        else:
            st.warning("Reference dataset row not found. Uploaded rows will be sent as-is.")
        
        if st.button("🔍 Analyze All"):
            with st.spinner(f"Analyzing {len(df)} transactions..."):
                # Process each transaction
                results = []
                progress = st.progress(0)
                
                for i, row in df.iterrows():
                    payload = build_full_payload(row.to_dict())
                    result = call_api(payload)
                    if result:
                        results.append({
                            'TransactionID': row.get('TransactionID', i),
                            'fraud_prob': result.get('fraud_probability', 0.5),
                            'decision': result.get('decision', 'REVIEW'),
                            'risk_level': result.get('risk_level', 'MEDIUM')
                        })
                    progress.progress((i + 1) / len(df))
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Summary
                st.subheader("📊 Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_risk = len(results_df[results_df['risk_level'] == 'HIGH'])
                    st.metric("High Risk", high_risk)
                
                with col2:
                    blocked = len(results_df[results_df['decision'] == 'BLOCK'])
                    st.metric("Blocked", blocked)
                
                with col3:
                    avg_prob = results_df['fraud_prob'].mean()
                    st.metric("Avg Fraud Prob", f"{avg_prob*100:.1f}%")


def demo_mode_view():
    """Demo mode with pre-configured examples"""
    st.header("🎮 Demo Mode")
    
    st.markdown("Select a demo scenario to see the system in action:")

    @st.cache_data
    def load_fraud_demo_scenarios() -> dict:
        transaction_ids = [2987203, 2987243, 2987245]

        if not TRAIN_TRANSACTION_PATH.exists():
            return {}

        tx_df = pd.read_csv(TRAIN_TRANSACTION_PATH)
        tx_df = tx_df[tx_df["TransactionID"].isin(transaction_ids)].copy()

        if tx_df.empty:
            return {}

        if TRAIN_IDENTITY_PATH.exists():
            id_df = pd.read_csv(TRAIN_IDENTITY_PATH)
            id_df = id_df[id_df["TransactionID"].isin(transaction_ids)].copy()
            merged = tx_df.merge(id_df, on="TransactionID", how="left")
        else:
            merged = tx_df

        scenarios_local = {}
        for txid in transaction_ids:
            row = merged[merged["TransactionID"] == txid]
            if row.empty:
                continue

            payload = {k: _sanitize_for_json(v) for k, v in row.iloc[0].to_dict().items()}

            # Add lightweight velocity placeholders used by the frontend/orchestrator.
            payload.setdefault("txn_count_1h", 0)
            payload.setdefault("amount_1h", 0.0)

            scenarios_local[f"Fraud Train Sample {txid}"] = payload

        return scenarios_local

    scenarios = load_fraud_demo_scenarios()

    if not scenarios:
        st.error("Could not load fraud demo scenarios from train CSV files.")
        return
    
    selected = st.selectbox("Choose Scenario", list(scenarios.keys()))
    
    st.json(scenarios[selected])
    
    if st.button("🔍 Run Demo Analysis", use_container_width=True):
        with st.spinner("Analyzing..."):
            payload = build_full_payload(scenarios[selected])
            result = call_api(payload)
        
        if result:
            display_results(result)


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
