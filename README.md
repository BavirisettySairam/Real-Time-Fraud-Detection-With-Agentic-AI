# Fraud Detection System

Real-time transaction fraud detection platform built on the IEEE-CIS dataset. Multi-agent ML pipeline with SHAP-based explainability and LLM-powered natural language explanations.

**Author:** Bavirisetty Sairam
**Contact:** [message2sairam@gmail.com](mailto:message2sairam@gmail.com) · +91 9513377365
**LinkedIn:** [linkedin.com/in/bavirisetty-sairam](https://linkedin.com/in/bavirisetty-sairam)
**GitHub:** [github.com/BavirisettySairam](https://github.com/BavirisettySairam)

---

## Overview

This system scores financial transactions for fraud using a four-agent architecture orchestrated by LangGraph. Each agent contributes a specialised signal, which is fused into a final risk score using dynamic weighting. The Yapper agent generates human-readable explanations by running SHAP on the primary model and feeding the results to an LLM.

| Component | Role |
|---|---|
| **Vibe Checker** | Primary ML ensemble — LightGBM + XGBoost (90/10 blend, 175 pipeline features) |
| **Era Tracker** | 24-hour behavioural window analysis — CatBoost on 199 features (175 pipeline + 24 sliding-window) |
| **OG Check** | Rule + ML hybrid — LightGBM on 194 features (175 pipeline + 19 rule-engineered) |
| **The Yapper** | Explainability — SHAP TreeExplainer → OpenRouter LLM |

---

## Production Metrics (Validation Set — 118,108 transactions)

| Agent | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Vibe Checker (Ensemble) | 0.8991 | 0.6001 | 0.7714 | 0.4794 | 0.5913 |
| OG Check | 0.8903 | 0.5352 | 0.6678 | 0.4417 | 0.5317 |
| Era Tracker (CatBoost) | 0.8773 | 0.5162 | 0.6572 | 0.4240 | 0.5154 |

Dataset split: 60% train (354,324) / 20% validation (118,108) / 20% test (118,108), stratified on `isFraud` (~3.5% fraud rate).

### Metric Progression

Development iterations that led to the current performance:

| Phase | Agent | ROC-AUC | PR-AUC | F1 | Change |
|---|---|---:|---:|---:|---|
| v1 — LightGBM baseline | Era Tracker | 0.7077 | 0.0893 | 0.0985 | Baseline |
| v2 — LogReg baseline | OG Check | 0.7039 | 0.0924 | 0.0086 | Baseline |
| v3 — Rule features + threshold tuning | OG Check | 0.7833 | 0.2311 | 0.2479 | PR-AUC +150% |
| v4 — CatBoost + behavioural features | Era Tracker | 0.7813 | 0.1612 | 0.2404 | PR-AUC +80%, Recall +335% |
| v5 — FeaturePipeline (175 features) | Vibe Checker | 0.8991 | 0.6001 | 0.5913 | Pipeline retrain |
| v5 — FeaturePipeline (175+24 features) | Era Tracker | 0.8773 | 0.5162 | 0.5154 | PR-AUC +220%, F1 +114% |
| v5 — FeaturePipeline (175+19 features) | OG Check | 0.8903 | 0.5352 | 0.5317 | PR-AUC +132%, F1 +114% |

---

## Architecture

```
Client Request
       │
       ▼
┌─────────────┐
│  FastAPI     │  POST /api/v1/predict
│  Gateway     │  POST /api/v1/explain
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│     FeaturePipeline          │
│  NaN/Inf → impute → encode  │
│  → engineer → select (175)  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│     LangGraph Orchestrator   │
│  (parallel fan-out/fan-in)   │
└──────┬───────┬───────┬──────┘
       │       │       │
       ▼       ▼       ▼
   ┌──────┐ ┌──────┐ ┌──────┐
   │ Vibe │ │ Era  │ │ OG   │
   │Checker│ │Tracker│ │Check │
   └──┬───┘ └──┬───┘ └──┬───┘
      │        │        │
      ▼        ▼        ▼
┌──────────────────────────────┐
│      Decision Aggregator     │
│  Dynamic fusion:             │
│  vibe > 0.8 → trust alone   │
│  else 0.60V + 0.25E + 0.15O │
└──────────┬───────────────────┘
           │
           ▼
    ┌─────────────┐
    │ The Yapper   │
    │ SHAP → LLM  │
    └──────┬──────┘
           │
           ▼
      Final Response
```

See [docs/architecture.md](docs/architecture.md) for full technical details.

### Latency

Measured p50 latency is **~880ms** on a single uvicorn worker (local dev). This includes FeaturePipeline preprocessing (~260ms) plus 4 agents (3 ML inference + SHAP) running in parallel via ThreadPoolExecutor. With 4 gunicorn workers + Redis warm cache + production hardware, expect **200–300ms** — well within tolerance for a fraud system where the alternative is manual review taking hours. See [docs/load_test_report.md](docs/load_test_report.md) for full benchmark data.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Conda or virtualenv

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your OpenRouter API key for LLM explanations
```

### Run

```bash
python run.py
```

This starts:
- **API** at `http://localhost:8000` (Swagger UI at `/docs`)
- **Streamlit Dashboard** at `http://localhost:8501`

### API Only

```bash
uvicorn services.api_gateway.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Single transaction fraud scoring |
| `POST` | `/api/v1/predict/batch` | Batch scoring (up to 256 transactions) |
| `POST` | `/api/v1/explain` | Full SHAP + LLM explanation |
| `GET` | `/api/v1/agents` | Agent configuration and weights |
| `GET` | `/api/v1/workflow` | LangGraph workflow + Mermaid diagram |
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

### Example — Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 450, "ProductCD": "W", "card4": "visa", "hour": 3}'
```

### Example — Full Explanation

```bash
curl -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 450, "ProductCD": "W", "card4": "visa", "hour": 3}'
```

Returns SHAP feature contributions, risk factors, LLM-generated plain-English explanation, confidence assessment, and recommended action.

See [docs/api.md](docs/api.md) for full endpoint reference.

---

## Training

```bash
# Train all agents
python train_agents.py --agent all --folds 5

# Train individually
python train_agents.py --agent vibe --folds 5
python train_agents.py --agent era --folds 3
python train_agents.py --agent og --folds 5
```

Metrics are logged to `logs/training_metrics.log` (append-only with timestamps and confusion matrices).

See [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) for full training documentation.

---

## Project Structure

```
├── ml/
│   └── preprocessing/        # Feature pipeline
│       └── feature_pipeline.py
├── services/
│   ├── api_gateway/          # FastAPI application
│   │   └── main.py
│   ├── orchestrator/         # LangGraph workflow
│   │   └── agent_orchestrator.py
│   └── agents/               # ML agents
│       ├── vibe_checker.py   # XGBoost + LightGBM ensemble
│       ├── era_tracker.py    # CatBoost behavioural analysis
│       ├── og_check.py       # Rule + LightGBM hybrid
│       └── the_yapper.py     # SHAP + LLM explainability
├── models/                   # Trained model artifacts + feature_pipeline.pkl
├── data/                     # Dataset (not committed)
├── frontend/                 # Streamlit dashboard
├── tests/                    # Unit + integration + load tests
├── docs/                     # Technical documentation
├── train_agents.py           # Training pipeline
├── run.py                    # Quick-start launcher
└── docker-compose.yml        # Container deployment
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML Models | LightGBM, XGBoost, CatBoost |
| Explainability | SHAP, OpenRouter (LLM) |
| Orchestration | LangGraph |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, Plotly |
| Observability | Prometheus, structured logging |
| Containerisation | Docker Compose |
| Cache | Redis |

---

## Documentation

- [Architecture](docs/architecture.md) — System design, agent pipeline, fusion logic
- [API Reference](docs/api.md) — Full endpoint specification with request/response schemas
- [Deployment](docs/deployment.md) — Local, Docker, and environment configuration
- [Testing](docs/testing.md) — Unit, integration, and load testing
- [Training Workflow](TRAINING_WORKFLOW.md) — Dataset preparation, training pipeline, metrics

---

## License

Internal use. All rights reserved.

**Bavirisetty Sairam** — [message2sairam@gmail.com](mailto:message2sairam@gmail.com)
