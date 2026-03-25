# =============================================================================
# LANGGRAPH PAGE - Dynamic workflow visualization
# =============================================================================

import os
import time
from typing import Any, Dict, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="LangGraph Workflow", layout="wide")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #2c2c2c; border: 1px solid #424242;
        border-radius: 8px; padding: 0.8rem;
    }
    div[data-testid="stMetric"] label { color: #9e9e9e !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def fetch_workflow(base_url: str) -> Tuple[Dict[str, Any], str]:
    """Fetch workflow payload from API and return data or error message."""
    try:
        response = requests.get(f"{base_url}/api/v1/workflow", timeout=8)
        if response.status_code != 200:
            return {}, f"API returned status {response.status_code}: {response.text[:200]}"
        return response.json(), ""
    except requests.exceptions.ConnectionError:
        return {}, "API not reachable — is the server running?"
    except requests.exceptions.Timeout:
        return {}, "Request timed out after 8 s"
    except requests.RequestException as exc:
        return {}, str(exc)


def normalize_mermaid(diagram_text: str) -> str:
    """Accept either fenced markdown or raw Mermaid and normalize to raw Mermaid."""
    if not diagram_text:
        return "graph TD\n    A[Unavailable] --> B[No diagram returned]"

    lines = diagram_text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

    normalized = "\n".join(lines).strip()
    return normalized or "graph TD\n    A[Unavailable] --> B[Empty diagram]"


def render_mermaid(diagram_text: str) -> None:
    """Render Mermaid graph via embedded Mermaid JS runtime."""
    mermaid_html = f"""
    <div style=\"padding: 8px; background: #ffffff; border-radius: 8px; border: 1px solid #e5e7eb;\">
      <pre class=\"mermaid\">{diagram_text}</pre>
    </div>
    <script type=\"module\">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose', theme: 'default' }});
    </script>
    """
    components.html(mermaid_html, height=700, scrolling=True)


st.title("LangGraph Workflow")
st.caption("Live architecture graph from `/api/v1/workflow`.")

with st.sidebar:
    st.subheader("Live Settings")
    api_url = st.text_input("API URL", value=API_URL)
    auto_refresh = st.toggle("Auto Refresh", value=True)
    refresh_seconds = st.slider("Refresh Every (seconds)", min_value=3, max_value=60, value=10)
    show_raw = st.toggle("Show Raw Mermaid", value=False)

workflow_data, error = fetch_workflow(api_url)

if error:
    st.error(f"Could not fetch workflow: {error}")
else:
    mode = workflow_data.get("mode", "unknown")
    active = bool(workflow_data.get("langgraph_active", False))
    generated_at = workflow_data.get("generated_at", "unknown")

    col1, col2, col3 = st.columns(3)
    col1.metric("Runtime", workflow_data.get("workflow", "LangGraph"))
    col2.metric("Execution Mode", mode)
    col3.metric("LangGraph Active", "Yes" if active else "No")
    st.caption(f"Last generated: {generated_at}")

    diagram = normalize_mermaid(workflow_data.get("mermaid_diagram", ""))
    render_mermaid(diagram)

    if show_raw:
        st.subheader("Raw Mermaid")
        st.code(diagram, language="mermaid")

if auto_refresh:
    st.caption(f"Auto-refreshing every {refresh_seconds} seconds")
    time.sleep(refresh_seconds)
    st.rerun()
