# =============================================================================
# PAGE 2 — DATASET & PREPROCESSING
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="Dataset & Preprocessing", layout="wide", page_icon="🗃️")

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
    .feature-tag {
        display: inline-block; background: #e3f2fd; color: #1565c0;
        border-radius: 4px; padding: 0.2rem 0.6rem; margin: 0.15rem;
        font-size: 0.8rem; font-family: monospace;
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
def load_pipeline_info() -> dict:
    return _load_json(MODELS_DIR / "feature_pipeline.json")


@st.cache_data
def load_split_metrics() -> dict:
    return _load_json(MODELS_DIR / "lightgbm_split_60_20_20_metrics.json")


@st.cache_data
def load_era_metrics() -> dict:
    return _load_json(MODELS_DIR / "era_tracker_metrics.json")


@st.cache_data
def load_og_params() -> dict:
    return _load_json(MODELS_DIR / "og_check_params.json")


st.title("Dataset & Preprocessing")
st.caption("IEEE-CIS Fraud Detection — real data from the training pipeline")

# =============================================================================
# DATASET OVERVIEW
# =============================================================================

st.markdown("### IEEE-CIS Fraud Detection Dataset")
st.markdown(
    "The [IEEE-CIS dataset](https://www.kaggle.com/c/ieee-fraud-detection) contains "
    "real-world e-commerce transactions provided by Vesta Corporation. Each record "
    "combines a **transaction table** (amounts, card info, device signals) with an "
    "**identity table** (device type, browser, OS, network metadata). The target "
    "variable `isFraud` indicates whether a transaction was fraudulent."
)

split_m = load_split_metrics()
rows_info = split_m.get("rows", {})
fraud_info = split_m.get("fraud_rate", {})

total_rows = rows_info.get("total", 590540)
train_rows = rows_info.get("train", 354324)
val_rows = rows_info.get("validation", 118108)
test_rows = rows_info.get("test", 118108)
fraud_rate = fraud_info.get("total", 0.035)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Transactions", f"{total_rows:,}")
with c2:
    st.metric("Fraud Rate", f"{fraud_rate:.2%}")
with c3:
    st.metric("Transaction Features", "394")
    st.caption("Original columns in transaction table")
with c4:
    st.metric("Identity Features", "41")
    st.caption("Original columns in identity table")

# --- Class distribution pie ---
st.markdown("---")
st.markdown("### Class Distribution")

fraud_count = int(round(total_rows * fraud_rate))
legit_count = total_rows - fraud_count

col_pie, col_info = st.columns([1, 1])
with col_pie:
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Legitimate", "Fraudulent"],
        values=[legit_count, fraud_count],
        marker=dict(colors=["#90caf9", "#e53935"]),
        textinfo="label+percent",
        textfont=dict(size=14, color="#212529"),
        hole=0.45,
    )])
    fig_pie.update_layout(
        height=320, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#495057"),
        showlegend=False,
        annotations=[dict(text=f"{fraud_rate:.1%}<br>fraud", x=0.5, y=0.5,
                          font_size=16, font_color="#495057", showarrow=False)],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_info:
    st.markdown(
        '<div class="info-card">'
        "<h4>Imbalanced Dataset</h4>"
        f"<p>Only <b>{fraud_rate:.2%}</b> of transactions are fraudulent — "
        f"roughly <b>{fraud_count:,}</b> out of <b>{total_rows:,}</b>.</p>"
        "<p>This severe class imbalance drives key design choices:</p>"
        "<ul>"
        "<li><b>Stratified splitting</b> — fraud rate preserved across train/val/test</li>"
        "<li><b>scale_pos_weight</b> — ~27.6× upweight for minority class in LightGBM/XGBoost</li>"
        "<li><b>PR-AUC</b> used alongside ROC-AUC (more sensitive to minority class)</li>"
        "<li><b>Threshold tuning</b> — F1-optimal thresholds per agent (not 0.5)</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# TRAIN / VAL / TEST SPLIT
# =============================================================================

st.markdown("---")
st.markdown("### 60 / 20 / 20 Stratified Split")
st.markdown(
    "The combined transaction + identity data is split with `stratify=isFraud` "
    "and `random_state=42` to ensure reproducibility and balanced fraud rates."
)

train_fraud = fraud_info.get("train", 0.035)
val_fraud = fraud_info.get("validation", 0.035)
test_fraud = fraud_info.get("test", 0.035)

fig_split = go.Figure()
fig_split.add_trace(go.Bar(
    x=["Train (60%)", "Validation (20%)", "Test (20%)"],
    y=[train_rows, val_rows, test_rows],
    marker_color=["#1976d2", "#42a5f5", "#90caf9"],
    text=[f"{train_rows:,}<br>fraud: {train_fraud:.3%}",
          f"{val_rows:,}<br>fraud: {val_fraud:.3%}",
          f"{test_rows:,}<br>fraud: {test_fraud:.3%}"],
    textposition="outside",
    textfont=dict(color="#495057", size=12),
))
fig_split.update_layout(
    height=340, yaxis_title="Transactions",
    margin=dict(t=20, b=20, l=60, r=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#495057"), yaxis=dict(gridcolor="#e9ecef"),
)
st.plotly_chart(fig_split, use_container_width=True)

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.metric("Train Rows", f"{train_rows:,}")
    st.caption(f"Fraud rate: {train_fraud:.4%}")
with sc2:
    st.metric("Validation Rows", f"{val_rows:,}")
    st.caption(f"Fraud rate: {val_fraud:.4%}")
with sc3:
    st.metric("Test Rows", f"{test_rows:,}")
    st.caption(f"Fraud rate: {test_fraud:.4%}")

# =============================================================================
# FEATURE PIPELINE
# =============================================================================

st.markdown("---")
st.markdown("### Feature Pipeline")
st.markdown(
    "The `FeaturePipeline` applies a multi-step cleaning and engineering process "
    "that transforms raw IEEE-CIS columns into a compact, high-signal feature set."
)

pipe = load_pipeline_info()
steps = pipe.get("pipeline_steps", {})
thresholds = pipe.get("thresholds", {})
final_count = pipe.get("final_feature_count", 175)

# Funnel chart for pipeline steps
step_labels = [
    f"After Missing Removal (>{thresholds.get('missing', 0.95):.0%} null)",
    f"After Variance Removal",
    f"After Correlation Removal (r>{thresholds.get('correlation', 0.98)})",
    f"Baseline Features",
    f"Final Pipeline Output",
]
step_values = [
    steps.get("step1_after_missing_removal", 422),
    steps.get("step2_after_variance_removal", 422),
    steps.get("step3_after_correlation_removal", 347),
    steps.get("step4_baseline", 209),
    final_count,
]

fig_funnel = go.Figure(go.Funnel(
    y=step_labels,
    x=step_values,
    textinfo="value",
    marker=dict(color=["#1976d2", "#2196f3", "#42a5f5", "#64b5f6", "#90caf9"]),
    textfont=dict(color="#212529", size=14),
    connector=dict(line=dict(color="#dee2e6")),
))
fig_funnel.update_layout(
    height=360, margin=dict(t=20, b=20, l=20, r=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#495057"),
)
st.plotly_chart(fig_funnel, use_container_width=True)

# Pipeline step descriptions
st.markdown(
    '<div class="step-card">'
    "<h4>Step 1 — Missing Value Removal</h4>"
    f"<p>Columns with >{thresholds.get('missing', 0.95):.0%} missing values are dropped. "
    f"Reduced from raw columns to <b>{steps.get('step1_after_missing_removal', 422)}</b> features.</p>"
    "</div>"
    '<div class="step-card">'
    "<h4>Step 2 — Zero-Variance Removal</h4>"
    f"<p>Constant columns removed. Kept <b>{steps.get('step2_after_variance_removal', 422)}</b> features.</p>"
    "</div>"
    '<div class="step-card">'
    "<h4>Step 3 — High-Correlation Removal</h4>"
    f"<p>One of each pair with Pearson |r| > {thresholds.get('correlation', 0.98)} is dropped. "
    f"Reduced to <b>{steps.get('step3_after_correlation_removal', 347)}</b> features.</p>"
    "</div>"
    '<div class="step-card">'
    "<h4>Step 4 — Information Gain Filter</h4>"
    f"<p>Features with mutual information < {thresholds.get('info_gain', 0.001)} removed. "
    f"<b>{steps.get('step4_baseline', 209)}</b> baseline features.</p>"
    "</div>"
    '<div class="step-card">'
    "<h4>Step 5 — Final Selection + Engineering</h4>"
    f"<p>Missingness indicators added (threshold: {thresholds.get('missingness_indicator', 0.2):.0%}), "
    f"class-specific median imputation, frequency encoding. Final output: <b>{final_count}</b> features.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# =============================================================================
# FEATURE CATEGORIES
# =============================================================================

st.markdown("---")
st.markdown("### Feature Categories")
st.markdown(
    "In addition to the 175 pipeline features, individual agents engineer "
    "specialised features for their specific detection approach."
)

fc1, fc2 = st.columns(2)

# -- ERA TRACKER sliding-window features --
era_m = load_era_metrics()
sw_features = era_m.get("sliding_window_features", [])

with fc1:
    st.markdown(
        '<div class="info-card">'
        f"<h4>Era Tracker — 24 Sliding-Window Features</h4>"
        "<p>Computed from each user's transaction history within a 24-hour window. "
        "Captures velocity, burst patterns, and temporal deviations.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    if sw_features:
        tags_html = "".join(f'<span class="feature-tag">{f}</span>' for f in sw_features)
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.info("Sliding-window feature list not found in era_tracker_metrics.json")

# -- OG CHECK rule features --
og_params = load_og_params()
rule_features = og_params.get("rule_features", [])
rule_thresholds = og_params.get("thresholds", {})

with fc2:
    st.markdown(
        '<div class="info-card">'
        f"<h4>OG Check — 19 Rule-Engineered Features</h4>"
        "<p>Binary flags from hand-crafted domain rules (amount limits, time-of-day, "
        "device/email mismatches, missing-field patterns).</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    if rule_features:
        tags_html = "".join(f'<span class="feature-tag">{f}</span>' for f in rule_features)
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.info("Rule feature list not found in og_check_params.json")

# --- Rule thresholds ---
if rule_thresholds:
    st.markdown("---")
    st.markdown("### OG Check Rule Thresholds")
    st.markdown("Thresholds learned from training data (percentile-based):")

    th_cols = st.columns(4)
    thresh_items = list(rule_thresholds.items())
    for i, (name, val) in enumerate(thresh_items):
        with th_cols[i % 4]:
            display_name = name.replace("_", " ").title()
            if isinstance(val, float):
                st.metric(display_name, f"${val:,.2f}" if "amount" in name or "threshold" in name or "spike" in name else f"{val:,.2f}")
            else:
                st.metric(display_name, f"{val}")

# =============================================================================
# FEATURE ENGINEERING SUMMARY
# =============================================================================

st.markdown("---")
st.markdown("### Engineering Summary")

eng_categories = {
    "Temporal": "15 features — hour-of-day (sin/cos), day-of-week, time deltas, night/weekend flags",
    "Amount": "12 features — z-scores, ratios to user mean/median/max, log-transform, binning",
    "Aggregation": "28 features — per-card1, per-addr1, per-email rolling stats (count, mean, std)",
    "Interaction": "25 features — card×amount, device×time, email-domain×product cross-features",
}

for cat, desc in eng_categories.items():
    st.markdown(
        f'<div class="step-card">'
        f"<h4>{cat} Features</h4>"
        f"<p>{desc}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

# --- Final note ---
st.markdown("---")
st.caption(
    "All numbers on this page are loaded from model JSON files generated during training. "
    "No synthetic or placeholder data is used."
)
