# =============================================================================
# PAGE 4 — PRESENTATION
# =============================================================================

import streamlit as st

st.set_page_config(page_title="Presentation", layout="wide", page_icon="📑")

st.markdown("""
<style>
    .doc-card {
        background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px;
        padding: 1.5rem; text-align: center;
    }
    .doc-card h4 { color: #212529; margin-bottom: 0.5rem; }
    .doc-card p { color: #6c757d; font-size: 0.95rem; }
    .doc-card a {
        display: inline-block; margin-top: 0.8rem; padding: 0.5rem 1.4rem;
        border-radius: 6px; background: #1976d2; color: #fff !important;
        text-decoration: none; font-weight: 500; font-size: 0.9rem;
    }
    .doc-card a:hover { background: #1565c0; }
    .embed-frame {
        border: 1px solid #dee2e6; border-radius: 8px;
        margin-top: 1rem; background: #fff;
    }
    .info-card {
        background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
        padding: 1.2rem; margin-bottom: 1rem;
    }
    .info-card h4 { margin: 0 0 0.5rem 0; color: #212529; }
    .info-card p, .info-card li { color: #495057; font-size: 0.95rem; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 8px; padding: 0.8rem;
    }
    div[data-testid="stMetric"] label { color: #6c757d !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #212529 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Presentation")
st.caption("Project documentation and presentation materials")

# =============================================================================
# PROJECT SUMMARY
# =============================================================================

st.markdown("### Project Summary")
st.markdown(
    "**Real-Time Fraud Detection with Agentic AI** is an end-to-end machine learning "
    "system that scores e-commerce transactions for fraud in real time. It combines "
    "three specialised ML agents (Vibe Checker, Era Tracker, OG Check) with SHAP "
    "explainability and an LLM-powered narrative generator to produce actionable, "
    "human-readable fraud decisions."
)

sm1, sm2, sm3, sm4 = st.columns(4)
with sm1:
    st.metric("Agents", "3")
    st.caption("Specialised ML models")
with sm2:
    st.metric("Features", "175")
    st.caption("Engineered per transaction")
with sm3:
    st.metric("Best ROC-AUC", "0.8991")
    st.caption("Vibe Checker ensemble")
with sm4:
    st.metric("Dataset", "590K")
    st.caption("IEEE-CIS transactions")

# =============================================================================
# KEY HIGHLIGHTS
# =============================================================================

st.markdown("---")
st.markdown("### Key Highlights")

hl1, hl2 = st.columns(2)
with hl1:
    st.markdown(
        '<div class="info-card">'
        "<h4>Multi-Agent Architecture</h4>"
        "<ul>"
        "<li><b>Vibe Checker</b> — LightGBM + XGBoost ensemble for broad pattern detection</li>"
        "<li><b>Era Tracker</b> — CatBoost with 24 sliding-window behavioural features</li>"
        "<li><b>OG Check</b> — LightGBM with 19 hand-crafted domain rules</li>"
        "<li>Weighted fusion with high-confidence override</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-card">'
        "<h4>Explainability</h4>"
        "<ul>"
        "<li>SHAP TreeExplainer for per-feature attributions</li>"
        "<li>Google Gemini LLM for natural-language explanations</li>"
        "<li>3-model cascade with template fallback</li>"
        "<li>Interactive waterfall charts in the UI</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
with hl2:
    st.markdown(
        '<div class="info-card">'
        "<h4>Feature Engineering</h4>"
        "<ul>"
        "<li>394 raw columns → 175 cleaned features via multi-step pipeline</li>"
        "<li>Missing removal, correlation filtering, info-gain selection</li>"
        "<li>Class-specific median imputation, missingness indicators</li>"
        "<li>Agent-specific additions: 24 temporal + 19 rule features</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-card">'
        "<h4>Production Stack</h4>"
        "<ul>"
        "<li>FastAPI backend with async agent execution</li>"
        "<li>Streamlit frontend with real-time detection</li>"
        "<li>Docker Compose deployment (API + Frontend + Nginx + Redis)</li>"
        "<li>103 tests, 78% coverage, Locust load testing</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# DOCUMENTS
# =============================================================================

st.markdown("---")
st.markdown("### Documents")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="doc-card">'
        "<h4>Project Document</h4>"
        "<p>Full project report covering problem statement, methodology, "
        "architecture design, training pipeline, evaluation results, "
        "and deployment strategy.</p>"
        '<a href="https://docs.google.com/document/d/1Ite0lvuRiGSQAEU1Y5V_3MMdxkn4baFIQx9XuccRbJ8/edit?usp=sharing" '
        'target="_blank">Open Document</a>'
        "</div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        '<div class="doc-card">'
        "<h4>Slide Deck</h4>"
        "<p>Presentation slides summarising the fraud detection system — "
        "architecture diagrams, model comparisons, performance metrics, "
        "and live demo walkthrough.</p>"
        '<a href="https://docs.google.com/presentation/d/16z_GUXKDf58VeVRYs0NdcIDF70Lsgp0M/edit?usp=sharing&ouid=101672989201710559146&rtpof=true&sd=true" '
        'target="_blank">Open Slides</a>'
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# EMBEDDED SLIDES
# =============================================================================

st.markdown("---")
st.markdown("### Slide Preview")

SLIDES_EMBED_URL = (
    "https://docs.google.com/presentation/d/16z_GUXKDf58VeVRYs0NdcIDF70Lsgp0M/embed"
    "?start=false&loop=false&delayms=3000"
)

st.markdown(
    f'<div class="embed-frame">'
    f'<iframe src="{SLIDES_EMBED_URL}" width="100%" height="500" '
    f'frameborder="0" allowfullscreen="true"></iframe>'
    f"</div>",
    unsafe_allow_html=True,
)

# =============================================================================
# SOURCE CODE & DOCS
# =============================================================================

st.markdown("---")
st.markdown("### Source Code & Documentation")

rc1, rc2, rc3 = st.columns(3)
with rc1:
    st.markdown(
        '<div class="doc-card">'
        "<h4>GitHub Repository</h4>"
        "<p>Full source code, training scripts, model configs, and CI setup.</p>"
        '<a href="https://github.com/BavirisettySairam/Real-Time-Fraud-Detection-With-Agentic-AI" '
        'target="_blank">View on GitHub</a>'
        "</div>",
        unsafe_allow_html=True,
    )
with rc2:
    st.markdown(
        '<div class="doc-card">'
        "<h4>API Documentation</h4>"
        "<p>Endpoint specs, request/response schemas, and authentication details.</p>"
        '<a href="https://github.com/BavirisettySairam/Real-Time-Fraud-Detection-With-Agentic-AI/blob/main/docs/api.md" '
        'target="_blank">View API Docs</a>'
        "</div>",
        unsafe_allow_html=True,
    )
with rc3:
    st.markdown(
        '<div class="doc-card">'
        "<h4>Architecture Guide</h4>"
        "<p>System design, data flow, agent orchestration, and deployment topology.</p>"
        '<a href="https://github.com/BavirisettySairam/Real-Time-Fraud-Detection-With-Agentic-AI/blob/main/docs/architecture.md" '
        'target="_blank">View Architecture</a>'
        "</div>",
        unsafe_allow_html=True,
    )
