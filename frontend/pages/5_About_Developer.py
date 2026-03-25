# =============================================================================
# PAGE 5 — ABOUT DEVELOPER
# =============================================================================

import streamlit as st

st.set_page_config(page_title="About Developer", layout="wide", page_icon="👤")

st.markdown("""
<style>
    .profile-card {
        background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 12px;
        padding: 2rem; text-align: center; max-width: 520px; margin: 2rem auto;
    }
    .profile-card h2 { color: #212529; margin-bottom: 0.3rem; }
    .profile-card p { color: #6c757d; font-size: 1rem; margin: 0.3rem 0; }
    .link-row { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 1.2rem; }
    .link-row a {
        display: inline-block; padding: 0.5rem 1.2rem; border-radius: 6px;
        background: #1976d2; color: #fff !important; text-decoration: none;
        font-size: 0.9rem; font-weight: 500;
    }
    .link-row a:hover { background: #1565c0; }
    .resume-frame {
        border: 1px solid #dee2e6; border-radius: 8px;
        margin-top: 2rem; background: #fff;
    }
    .info-card {
        background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
        padding: 1.2rem; margin-bottom: 1rem;
    }
    .info-card h4 { margin: 0 0 0.5rem 0; color: #212529; }
    .info-card p, .info-card li { color: #495057; font-size: 0.95rem; }
    .tech-tag {
        display: inline-block; background: #e3f2fd; color: #1565c0;
        border-radius: 4px; padding: 0.2rem 0.6rem; margin: 0.15rem;
        font-size: 0.82rem; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.title("About Developer")

st.markdown(
    '<div class="profile-card">'
    "<h2>Bavirisetty Sairam</h2>"
    "<p>ML & Data Engineering</p>"
    "<p style='font-size:0.85rem;color:#adb5bd;'>Building real-time fraud detection systems with agentic AI.</p>"
    '<div class="link-row">'
    '<a href="https://github.com/BavirisettySairam" target="_blank">GitHub</a>'
    '<a href="https://linkedin.com/in/bavirisetty-sairam" target="_blank">LinkedIn</a>'
    '<a href="mailto:message2sairam@gmail.com">Email</a>'
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

# =============================================================================
# ABOUT THIS PROJECT
# =============================================================================

st.markdown("---")
st.markdown("### About This Project")
st.markdown(
    "This real-time fraud detection system was designed and built as a capstone project "
    "demonstrating the integration of multiple ML models, explainable AI, and "
    "production-grade deployment practices. The system processes IEEE-CIS e-commerce "
    "transactions through three specialised agents, each contributing a unique detection "
    "perspective, fused into a single actionable decision."
)

# =============================================================================
# SKILLS & TECH
# =============================================================================

st.markdown("---")
st.markdown("### Skills & Technologies Used")

skills = {
    "Machine Learning": ["LightGBM", "XGBoost", "CatBoost", "scikit-learn", "SHAP"],
    "Data Engineering": ["Pandas", "NumPy", "Feature Engineering", "Pipeline Design"],
    "AI / LLM": ["Google Gemini API", "LangGraph", "Prompt Engineering", "Agentic AI"],
    "Backend": ["Python", "FastAPI", "Uvicorn", "asyncio", "Pydantic"],
    "Frontend": ["Streamlit", "Plotly", "HTML/CSS", "Responsive Design"],
    "DevOps": ["Docker", "Docker Compose", "Nginx", "Redis", "Prometheus"],
    "Testing": ["pytest", "Locust", "Integration Testing", "Load Testing"],
}

for category, items in skills.items():
    tags = "".join(f'<span class="tech-tag">{t}</span>' for t in items)
    st.markdown(f"**{category}:** {tags}", unsafe_allow_html=True)

# =============================================================================
# PROJECT CONTRIBUTIONS
# =============================================================================

st.markdown("---")
st.markdown("### What I Built")

bc1, bc2 = st.columns(2)
with bc1:
    st.markdown(
        '<div class="info-card">'
        "<h4>ML Pipeline</h4>"
        "<ul>"
        "<li>FeaturePipeline: 394 raw columns → 175 cleaned features</li>"
        "<li>3-agent architecture with weighted ensemble fusion</li>"
        "<li>Era Tracker: 24 sliding-window behavioural features</li>"
        "<li>OG Check: 19 hand-crafted domain rules</li>"
        "<li>Cross-validated training with F1-optimal thresholds</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-card">'
        "<h4>Explainability</h4>"
        "<ul>"
        "<li>SHAP TreeExplainer integration for all agents</li>"
        "<li>Gemini LLM narrative with 3-model cascade</li>"
        "<li>Template fallback for robustness</li>"
        "<li>Interactive waterfall charts</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
with bc2:
    st.markdown(
        '<div class="info-card">'
        "<h4>Full-Stack Application</h4>"
        "<ul>"
        "<li>FastAPI backend with async agent orchestration</li>"
        "<li>Streamlit frontend with real-time detection UI</li>"
        "<li>Demo mode with real TransactionIDs from dataset</li>"
        "<li>Batch processing support (up to 10 transactions)</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-card">'
        "<h4>Production Engineering</h4>"
        "<ul>"
        "<li>Docker Compose deployment (4 services)</li>"
        "<li>Nginx reverse proxy with rate limiting</li>"
        "<li>Prometheus metrics endpoint</li>"
        "<li>103 tests with 78% code coverage</li>"
        "<li>Locust load testing (7.6 RPS)</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# RESUME
# =============================================================================

st.markdown("---")
st.markdown("### Resume")

RESUME_EMBED_URL = "https://drive.google.com/file/d/19FOyLTa60l7m5RI5ZtbvJBNSWin7aquv/preview"

st.markdown(
    f'<div class="resume-frame">'
    f'<iframe src="{RESUME_EMBED_URL}" width="100%" height="800" frameborder="0" '
    f'allow="autoplay"></iframe>'
    f"</div>",
    unsafe_allow_html=True,
)
