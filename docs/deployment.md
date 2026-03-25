# Deployment Guide

Instructions for running the Fraud Detection System in local, Docker, and production-adjacent environments.

**Author:** Bavirisetty Sairam
**Last Updated:** March 2026

---

## Option 1: Native (Local Development)

### Prerequisites

- Python 3.10+ (tested on 3.13)
- Conda or virtualenv
- ~4 GB RAM minimum (model loading)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY for LLM explanations
```

### Run

```bash
python run.py
```

Starts both services:
- **API** — `http://localhost:8000` (Swagger at `/docs`)
- **Streamlit** — `http://localhost:8501`

### API Only

```bash
uvicorn services.api_gateway.main:app --host 0.0.0.0 --port 8000
```

---

## Option 2: Docker Compose

### Prerequisites

- Docker Engine 20+
- Docker Compose v2+

### Run

```bash
docker compose up --build
```

Services:

| Service | Port | Description |
|---|---|---|
| `fraud-api` | 8000 | FastAPI backend with health checks |
| `fraud-frontend` | 8501 | Streamlit dashboard |
| `redis` | 6379 | Redis 7 (Alpine) for caching |

The API container uses `requirements.api-docker.txt` and the frontend uses `requirements.frontend-docker.txt` — slim dependency sets that exclude training-only packages to keep images small and builds fast.

### Stopping

```bash
docker compose down
```

---

## Environment Variables

### Required

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Google Gemini API key for LLM explanations |

### API Gateway

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated allowed origins |
| `BATCH_MAX_SIZE` | `256` | Maximum transactions per batch request |
| `BATCH_MAX_WORKERS` | `8` | Concurrent batch processing workers |

### Orchestrator

| Variable | Default | Description |
|---|---|---|
| `USE_LANGGRAPH` | `true` | Enable LangGraph runtime |
| `STRICT_LANGGRAPH` | `true` | Fail on startup if LangGraph is missing |
| `ORCHESTRATOR_PARALLEL` | `true` | Parallel agent execution |
| `ORCHESTRATOR_TIMEOUT_MS` | `5000` | Per-agent timeout in milliseconds |
| `MODEL_PRIMARY_WEIGHT` | `0.75` | Vibe Checker weight in legacy fusion |

### LLM (The Yapper)

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `GOOGLE_MODEL` | `gemini-2.5-flash-lite` | Primary model (cascade fallback to gemini-flash-lite-latest, gemini-2.5-flash) |

### Frontend

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | Backend API URL |

---

## Model Artifacts

The following files must be present in the `models/` directory at startup:

| File | Agent | Format |
|---|---|---|
| `vibe_lgb.txt` | Vibe Checker | LightGBM Booster text |
| `vibe_xgb.json` | Vibe Checker | XGBoost Booster JSON |
| `vibe_metrics.json` | Vibe Checker | Threshold + ensemble weight |
| `era_tracker_catboost.cbm` | Era Tracker | CatBoost binary model |
| `og_check_lgb.txt` | OG Check | LightGBM Booster text |
| `og_check_params.json` | OG Check | Feature list + threshold |

If any model file is missing, the corresponding agent defaults to a fallback score (0.5) and logs a warning. The system remains operational.

---

## Health Monitoring

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check — returns component status |
| `GET /ready` | Readiness check — returns 503 until orchestrator is initialised |
| `GET /metrics` | Prometheus-compatible metrics (requires `prometheus-client`) |

Docker Compose includes a health check on the `/health` endpoint with 10-second intervals.

---

## Resource Requirements

| Environment | CPU | RAM | Disk |
|---|---|---|---|
| Local development | 2+ cores | 4 GB | 2 GB (models + data) |
| Docker (API only) | 2+ cores | 2 GB | 500 MB (models) |
| Docker (full stack) | 4+ cores | 4 GB | 1 GB |

The primary memory consumer is the LightGBM model (445 features). CatBoost and SHAP add approximately 200 MB.
