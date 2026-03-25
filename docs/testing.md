# Testing & Validation

Test strategy, test execution, and quality gates for the Fraud Detection System.

**Author:** Bavirisetty Sairam
**Last Updated:** March 2026

---

## Test Structure

```
tests/
├── unit/
│   └── test_orchestrator.py    # Orchestrator logic, fusion, error handling
├── integration/
│   └── test_api.py             # API endpoint smoke tests
└── load_test.py                # Locust-based load testing
```

---

## Unit Tests

Tests cover orchestrator scoring logic, dynamic fusion, parallel error handling, and LangGraph integration.

### Current Test Suite

| Test | What It Validates |
|---|---|
| `test_orchestrator_weighted_decision_and_output_shape` | Score fusion, dynamic weighting (vibe > 0.8 pass-through), output schema |
| `test_orchestrator_parallel_path_handles_agent_errors` | Agent failure fallback (score defaults to 0.5), system stability |
| `test_orchestrator_langgraph_path_when_available` | LangGraph workflow compilation and execution |
| `test_orchestrator_strict_langgraph_raises_when_missing` | Strict mode raises `RuntimeError` when LangGraph is absent |

### Run

```bash
python -m pytest tests/unit -v --tb=short
```

Expected: 4 passed.

---

## Integration Tests

API-level tests that verify endpoint contracts, response schemas, and error handling.

### Run

```bash
# Start the API first
uvicorn services.api_gateway.main:app --port 8000 &

# Run integration tests
python -m pytest tests/integration -v --tb=short
```

---

## Load Testing

Locust-based load test for the `/api/v1/predict` endpoint.

### Run

```bash
# Start the API first
uvicorn services.api_gateway.main:app --port 8000

# In another terminal
locust -f tests/load_test.py --host http://localhost:8000
```

Open `http://localhost:8089` for the Locust dashboard.

### Key Metrics to Monitor

| Metric | Target |
|---|---|
| p50 latency (`/api/v1/predict`) | ~620ms measured (single uvicorn worker, local dev) |
| p95 latency (`/api/v1/predict`) | ~780ms measured (target <2000ms) |
| Error rate | 0% under normal load |
| Batch endpoint near `BATCH_MAX_SIZE` | Graceful 413 rejection |

---

## Observability Checks

| Check | How |
|---|---|
| Prometheus metrics | `GET /metrics` — verify counters and histograms are present |
| Request tracing | Verify `X-Request-ID` and `X-Process-Time-ms` headers in responses |
| SHAP availability | `POST /api/v1/explain` — verify `shap_available: true` in response |
| LLM integration | `POST /api/v1/explain` — verify `llm_used: true` when API key is configured |

---

## Validation Workflow

Before any deployment or merge:

1. **Unit tests pass** — `pytest tests/unit -v`
2. **API starts cleanly** — no errors in startup logs, `/ready` returns 200
3. **SHAP produces non-zero values** — `/api/v1/explain` returns `top_features` with non-zero `shap_value`
4. **All three agents load models** — check startup logs for `Loaded LightGBM model`, `CatBoost model loaded`, `OG Check LightGBM model loaded`
5. **LLM responds** (if configured) — `/api/v1/explain` returns `llm_used: true`

---

## Training Validation

After retraining any agent:

1. Check `logs/training_metrics.log` for the new entry (timestamped with confusion matrix).
2. Compare ROC-AUC and PR-AUC against the previous run.
3. Run unit tests to ensure inference compatibility.
4. Hit `/api/v1/explain` to verify SHAP still works with the new model.
