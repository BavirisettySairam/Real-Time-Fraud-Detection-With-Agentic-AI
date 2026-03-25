#!/usr/bin/env python
# =============================================================================
# QUICK START SCRIPT - Run API and Streamlit locally
# =============================================================================

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests


def wait_for_url(url: str, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def spawn_process(command: list[str], cwd: Path, env: dict[str, str]):
    if sys.platform == "win32":
        return subprocess.Popen(command, cwd=str(cwd), env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
    return subprocess.Popen(command, cwd=str(cwd), env=env)


def main() -> None:
    print("=" * 70)
    print("Fraud Detection System - Local Production Quick Start")
    print("=" * 70)

    base_dir = Path(__file__).parent

    print(f"Python: {sys.version.split()[0]}")
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=False)

    api_env = os.environ.copy()
    api_env.setdefault("API_HOST", "0.0.0.0")
    api_env.setdefault("API_PORT", "8000")
    api_env.setdefault("USE_LANGGRAPH", "true")
    api_env.setdefault("STRICT_LANGGRAPH", "true")
    api_env.setdefault("ORCHESTRATOR_PARALLEL", "true")
    api_env.setdefault("MODEL_PRIMARY_WEIGHT", "0.75")

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "services.api_gateway.main:app",
        "--host",
        api_env["API_HOST"],
        "--port",
        api_env["API_PORT"],
    ]

    print("Starting API server on http://localhost:8000 ...")
    api_proc = spawn_process(api_cmd, base_dir, api_env)

    if not wait_for_url("http://localhost:8000/ready", timeout_seconds=45):
        print("API failed readiness check. Stop and inspect logs.")
        api_proc.terminate()
        return

    frontend_env = os.environ.copy()
    frontend_env.setdefault("API_URL", "http://localhost:8000")

    print("Starting Streamlit frontend on http://localhost:8501 ...")
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "Home.py",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
    frontend_dir = base_dir / "frontend"
    streamlit_proc = spawn_process(streamlit_cmd, frontend_dir, frontend_env)

    time.sleep(2)
    webbrowser.open("http://localhost:8501")

    print("\nSystem is running")
    print("- API docs: http://localhost:8000/docs")
    print("- Frontend: http://localhost:8501")
    print("Press Ctrl+C to stop both services")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        streamlit_proc.terminate()
        api_proc.terminate()


if __name__ == "__main__":
    main()
