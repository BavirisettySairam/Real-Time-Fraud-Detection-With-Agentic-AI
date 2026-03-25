# =============================================================================
# PAGE 3 — SYSTEM OVERVIEW
# =============================================================================

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="System Overview", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .info-card {
        background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
        padding: 1.2rem; margin-bottom: 1rem;
    }
    .info-card h4 { margin: 0 0 0.5rem 0; color: #212529; }
    .info-card p, .info-card li { color: #495057; font-size: 0.95rem; }
    .step-card {
        background: #f8f9fa; border-left: 4px solid #1976d2; border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .step-card h4 { margin: 0 0 0.3rem 0; color: #1976d2; font-size: 1rem; }
    .step-card p { color: #495057; font-size: 0.9rem; margin: 0; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 8px; padding: 0.8rem;
    }
    div[data-testid="stMetric"] label { color: #6c757d !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #212529 !important; }
    .tech-tag {
        display: inline-block; background: #e3f2fd; color: #1565c0;
        border-radius: 4px; padding: 0.2rem 0.6rem; margin: 0.15rem;
        font-size: 0.82rem; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data
def load_metrics():
    vibe = _load_json(MODELS_DIR / "vibe_metrics.json")
    era = _load_json(MODELS_DIR / "era_tracker_metrics.json")
    og = _load_json(MODELS_DIR / "og_check_params.json")
    return vibe, era, og


vibe_raw, era_raw, og_raw = load_metrics()

st.title("System Overview")
st.caption("Architecture, agents, explainability pipeline, and deployment details")

# =============================================================================
# ARCHITECTURE FLOW
# =============================================================================

st.markdown("### Architecture")
st.markdown(
    "```\n"
    "Transaction Input  (TransactionID + raw features)\n"
    "       │\n"
    "       ▼\n"
    "  FeaturePipeline (175 cleaned features)\n"
    "       │\n"
    "       ├──▶ Vibe Checker  (LGB + XGB ensemble)  ──┐\n"
    "       ├──▶ Era Tracker   (CatBoost + 24 window) ──┤\n"
    "       └──▶ OG Check      (LGB + 19 rules)      ──┤\n"
    "                                                   ▼\n"
    "                                Weighted Fusion (60/25/15)\n"
    "                                                   │\n"
    "                                SHAP TreeExplainer (top-10 features)\n"
    "                                                   │\n"
    "                                Gemini LLM Summary (3-model cascade)\n"
    "                                                   │\n"
    "                                              API Response\n"
    "                                 (score, decision, explanation, SHAP)\n"
    "```"
)

st.markdown("---")
st.markdown("### Request Lifecycle")
st.markdown(
    '<div class="step-card">'
    "<h4>1. Input Parsing</h4>"
    "<p>FastAPI validates the incoming JSON payload. Supports single transaction "
    "or batch (up to 10). TransactionID is looked up in the dataset for demo mode.</p></div>"
    '<div class="step-card">'
    "<h4>2. Feature Engineering</h4>"
    "<p>FeaturePipeline transforms 394 raw columns → 175 features via missing removal, "
    "correlation filtering, info-gain selection, class-specific median imputation, and "
    "missingness indicators. Runs in ~260 ms.</p></div>"
    '<div class="step-card">'
    "<h4>3. Parallel Agent Scoring</h4>"
    "<p>All 3 agents run concurrently (asyncio.gather). Each produces a fraud probability "
    "from its own model plus agent-specific supplementary features.</p></div>"
    '<div class="step-card">'
    "<h4>4. Weighted Fusion</h4>"
    "<p>Default: 60% Vibe · 25% Era · 15% OG. High-confidence override: if Vibe ≥ 0.8, "
    "it gets 100% weight. Final score mapped to BLOCK / REVIEW / APPROVE.</p></div>"
    '<div class="step-card">'
    "<h4>5. Explainability</h4>"
    "<p>SHAP TreeExplainer computes per-feature attributions. Top-10 sent to Gemini "
    "for plain-English narrative. Template fallback if LLM unavailable.</p></div>"
    '<div class="step-card">'
    "<h4>6. Response</h4>"
    "<p>JSON with fraud_score, decision (BLOCK/REVIEW/APPROVE), per-agent scores, "
    "SHAP chart data, LLM explanation, and latency breakdown.</p></div>",
    unsafe_allow_html=True,
)

# =============================================================================
# FUSION LOGIC
# =============================================================================

st.markdown("---")
st.markdown("### Fusion & Decision Logic")

fl1, fl2 = st.columns(2)
with fl1:
    st.markdown(
        '<div class="info-card">'
        "<h4>Fusion Weights</h4>"
        "<table style='width:100%;color:#495057;font-size:0.95rem;'>"
        "<tr><td><b>Agent</b></td><td><b>Default</b></td><td><b>High-Conf</b></td></tr>"
        "<tr><td>Vibe Checker</td><td>60%</td><td>100%</td></tr>"
        "<tr><td>Era Tracker</td><td>25%</td><td>0%</td></tr>"
        "<tr><td>OG Check</td><td>15%</td><td>0%</td></tr>"
        "</table>"
        "<p style='margin-top:0.8rem;font-size:0.85rem;'>High-confidence override triggers "
        "when Vibe Checker score ≥ 0.80.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with fl2:
    st.markdown(
        '<div class="info-card">'
        "<h4>Decision Thresholds</h4>"
        "<table style='width:100%;color:#495057;font-size:0.95rem;'>"
        "<tr><td><b>Score Range</b></td><td><b>Decision</b></td><td><b>Action</b></td></tr>"
        "<tr><td>≥ 0.70</td><td style='color:#d32f2f;font-weight:600;'>BLOCK</td>"
        "<td>Auto-decline</td></tr>"
        "<tr><td>0.40 – 0.69</td><td style='color:#f57c00;font-weight:600;'>REVIEW</td>"
        "<td>Manual queue</td></tr>"
        "<tr><td>< 0.40</td><td style='color:#388e3c;font-weight:600;'>APPROVE</td>"
        "<td>Pass-through</td></tr>"
        "</table></div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# MODEL DETAILS
# =============================================================================

st.markdown("---")
st.markdown("### Agent Details")

vibe_ens = vibe_raw.get("ensemble", {})
og_m = og_raw.get("metrics", {})

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        '<div class="info-card">'
        "<h4>Vibe Checker</h4>"
        "<p><b>LightGBM + XGBoost</b> ensemble (70/30 blend)</p>"
        "<ul>"
        f"<li>ROC-AUC: <b>{vibe_ens.get('roc_auc', 0):.4f}</b></li>"
        f"<li>PR-AUC: {vibe_ens.get('pr_auc', 0):.4f}</li>"
        f"<li>F1: {vibe_ens.get('f1', 0):.4f}</li>"
        f"<li>Threshold: {vibe_ens.get('threshold', 0):.4f}</li>"
        "<li>175 pipeline features</li>"
        "<li>5-fold cross-validation</li>"
        "</ul>"
        "<p style='font-size:0.85rem;color:#6c757d;'>Primary agent — gets highest fusion "
        "weight and high-confidence override.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="info-card">'
        "<h4>Era Tracker</h4>"
        "<p><b>CatBoost</b> with 24 sliding-window behavioural features</p>"
        "<ul>"
        f"<li>ROC-AUC: <b>{era_raw.get('roc_auc', 0):.4f}</b></li>"
        f"<li>PR-AUC: {era_raw.get('pr_auc', 0):.4f}</li>"
        f"<li>F1: {era_raw.get('f1', 0):.4f}</li>"
        f"<li>Threshold: {era_raw.get('threshold', 0):.4f}</li>"
        f"<li>{era_raw.get('num_features', 199)} total features (175 + 24 window)</li>"
        f"<li>Best iteration: {era_raw.get('best_iteration', 'N/A')}</li>"
        "</ul>"
        "<p style='font-size:0.85rem;color:#6c757d;'>Temporal specialist — detects velocity "
        "spikes, burst transactions, and behavioural shifts.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="info-card">'
        "<h4>OG Check</h4>"
        "<p><b>LightGBM + hand-crafted rules</b></p>"
        "<ul>"
        f"<li>ROC-AUC: <b>{og_m.get('roc_auc', 0):.4f}</b></li>"
        f"<li>PR-AUC: {og_m.get('pr_auc', 0):.4f}</li>"
        f"<li>F1: {og_m.get('f1', 0):.4f}</li>"
        f"<li>Threshold: {og_m.get('threshold', 0):.4f}</li>"
        "<li>175 + 19 rule features</li>"
        "<li>3-fold cross-validation</li>"
        "</ul>"
        "<p style='font-size:0.85rem;color:#6c757d;'>Rule-based specialist — encodes domain "
        "knowledge (night-owl, high-amount, email mismatch, etc.).</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# EXPLAINABILITY
# =============================================================================

st.markdown("---")
st.markdown("### Explainability Pipeline")

ex1, ex2 = st.columns([1, 1])
with ex1:
    st.markdown(
        '<div class="step-card">'
        "<h4>SHAP TreeExplainer</h4>"
        "<p>Computes exact Shapley values for tree-based models. Top-10 features "
        "by absolute SHAP value are extracted for each prediction.</p></div>"
        '<div class="step-card">'
        "<h4>Gemini LLM Narrative</h4>"
        "<p>Top features + SHAP values are sent to Google Gemini with a structured "
        "prompt including transaction context (amount, product, time).</p></div>"
        '<div class="step-card">'
        "<h4>Template Fallback</h4>"
        "<p>If LLM is unavailable (quota, timeout), a rule-based template generates "
        "the explanation from SHAP values directly.</p></div>",
        unsafe_allow_html=True,
    )
with ex2:
    st.markdown(
        '<div class="info-card">'
        "<h4>LLM Cascade</h4>"
        "<p>3-model failover for reliability:</p>"
        "<ol>"
        "<li><b>gemini-2.5-flash-lite</b> — fastest, primary</li>"
        "<li><b>gemini-flash-lite-latest</b> — backup</li>"
        "<li><b>gemini-2.5-flash</b> — final fallback</li>"
        "</ol>"
        "<p>Each model gets 8-second timeout. If all fail, "
        "template explanation is used.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-card">'
        "<h4>Output Format</h4>"
        "<p>The explanation includes:</p>"
        "<ul>"
        "<li>Risk assessment sentence</li>"
        "<li>Top contributing features with direction (increases/decreases risk)</li>"
        "<li>Transaction context interpretation</li>"
        "<li>Waterfall SHAP chart (interactive Plotly)</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# TECH STACK
# =============================================================================

st.markdown("---")
st.markdown("### Technology Stack")

stack = {
    "ML / AI": ["LightGBM", "XGBoost", "CatBoost", "SHAP", "scikit-learn", "NumPy", "Pandas"],
    "Backend": ["FastAPI", "Uvicorn", "Pydantic", "LangGraph", "asyncio"],
    "Frontend": ["Streamlit", "Plotly", "Custom CSS"],
    "LLM": ["Google Gemini API", "3-model cascade", "Template fallback"],
    "Infra": ["Docker Compose", "Nginx", "Redis", "Prometheus metrics"],
    "Testing": ["pytest", "Locust", "103 tests", "78% coverage"],
}

for category, techs in stack.items():
    tags = "".join(f'<span class="tech-tag">{t}</span>' for t in techs)
    st.markdown(f"**{category}:** {tags}", unsafe_allow_html=True)

# =============================================================================
# API ENDPOINTS
# =============================================================================

st.markdown("---")
st.markdown("### API Endpoints")

endpoints = [
    ("GET", "/health", "Health check — version, uptime, model status"),
    ("GET", "/ready", "Readiness probe — all 3 agents loaded check"),
    ("POST", "/api/v1/predict", "Single transaction fraud prediction"),
    ("POST", "/api/v1/predict/batch", "Batch prediction (up to 10 transactions)"),
    ("POST", "/api/v1/explain", "Predict + SHAP values + LLM explanation"),
    ("GET", "/api/v1/agents", "List registered agents and fusion weights"),
    ("GET", "/api/v1/workflow", "LangGraph workflow diagram (Mermaid)"),
    ("GET", "/metrics", "Prometheus metrics (request count, latency histograms)"),
]

header = "| Method | Path | Description |\n|--------|------|-------------|\n"
rows = "\n".join(f"| `{m}` | `{p}` | {d} |" for m, p, d in endpoints)
st.markdown(header + rows)

st.markdown(
    '<div class="info-card" style="margin-top:1rem;">'
    "<h4>Request/Response Example</h4>"
    "<p><b>POST /api/v1/predict</b></p>"
    '<pre style="background:#f1f3f5;padding:0.8rem;border-radius:6px;font-size:0.85rem;color:#495057;">'
    '// Request\n'
    '{"TransactionID": 2987016}\n\n'
    '// Response\n'
    '{\n'
    '  "fraud_score": 0.034,\n'
    '  "decision": "APPROVE",\n'
    '  "agents": {\n'
    '    "vibe_checker": 0.028,\n'
    '    "era_tracker": 0.041,\n'
    '    "og_check": 0.039\n'
    '  }\n'
    "}</pre></div>",
    unsafe_allow_html=True,
)

# =============================================================================
# DEPLOYMENT
# =============================================================================

st.markdown("---")
st.markdown("### Deployment")

dc1, dc2 = st.columns(2)
with dc1:
    st.markdown(
        '<div class="info-card">'
        "<h4>Docker Compose Stack</h4>"
        "<ul>"
        "<li><b>api-gateway</b> — FastAPI + Uvicorn (port 8000)</li>"
        "<li><b>frontend</b> — Streamlit (port 8501)</li>"
        "<li><b>nginx</b> — Reverse proxy + rate limiting (port 80)</li>"
        "<li><b>redis</b> — Optional caching layer (port 6379)</li>"
        "</ul>"
        "<p style='font-size:0.85rem;color:#6c757d;'>All containers share a bridge network. "
        "Health-check probes configured on both services.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with dc2:
    st.markdown(
        '<div class="info-card">'
        "<h4>Production Targets</h4>"
        "<ul>"
        "<li><b>Workers:</b> gunicorn --workers 4 for multi-core</li>"
        "<li><b>Caching:</b> Redis for repeated TransactionID lookups</li>"
        "<li><b>Latency:</b> p50 target 200–300 ms (currently ~880 ms single-worker)</li>"
        "<li><b>Rate Limit:</b> 100 req/min per IP via Nginx</li>"
        "<li><b>Monitoring:</b> Prometheus /metrics endpoint</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# LINKS
# =============================================================================

st.markdown("---")
st.markdown("### Resources")

rc1, rc2, rc3 = st.columns(3)
with rc1:
    st.link_button("GitHub Repository", "https://github.com/BavirisettySairam/Real-Time-Fraud-Detection-With-Agentic-AI", use_container_width=True)
with rc2:
    st.link_button("API Documentation", "https://github.com/BavirisettySairam/Real-Time-Fraud-Detection-With-Agentic-AI/blob/main/docs/api.md", use_container_width=True)
with rc3:
    st.link_button("Architecture Docs", "https://github.com/BavirisettySairam/Real-Time-Fraud-Detection-With-Agentic-AI/blob/main/docs/architecture.md", use_container_width=True)
