# API Reference

Full specification for the Fraud Detection System REST API. All endpoints are served by FastAPI with auto-generated OpenAPI documentation available at `/docs`.

**Base URL:** `http://localhost:8000`
**Content-Type:** `application/json`

---

## Endpoints

### Health & Readiness

#### `GET /health`

Returns service health status, version, uptime, and component states.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.1.0",
  "timestamp": "2026-03-25T12:00:00+00:00",
  "uptime_seconds": 3600.5,
  "components": {
    "api": "healthy",
    "orchestrator": "healthy",
    "agents": "healthy"
  }
}
```

#### `GET /ready`

Kubernetes-compatible readiness probe. Returns `200` when the orchestrator is initialised, `503` otherwise.

#### `GET /metrics`

Prometheus-compatible metrics output. Requires `prometheus-client` to be installed; returns a JSON hint otherwise.

Exposed metrics:
- `fraud_api_requests_total` — counter by method, path, status
- `fraud_api_request_latency_seconds` — histogram by method, path
- `fraud_predictions_total` — counter by endpoint, decision
- `fraud_prediction_latency_seconds` — histogram by endpoint

---

### Prediction

#### `POST /api/v1/predict`

Score a single transaction for fraud.

**Request Body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `TransactionAmt` | float | **Yes** | Transaction amount (≥ 0) |
| `TransactionID` | int | No | Unique identifier |
| `TransactionDT` | float | No | Seconds from dataset reference start |
| `ProductCD` | string | No | Product code (W, C, H, R, S) |
| `card1` | int | No | Card identifier |
| `card4` | string | No | Card network (visa, mastercard, etc.) |
| `card6` | string | No | Card type (debit, credit) |
| `addr1` | float | No | Billing address region |
| `addr2` | float | No | Billing country code |
| `P_emaildomain` | string | No | Purchaser email domain |
| `R_emaildomain` | string | No | Recipient email domain |
| `DeviceType` | string | No | Device type |
| `DeviceInfo` | string | No | Device information |
| `hour` | int (0–23) | No | Transaction hour (derived from current time if omitted) |
| `day` | int (0–6) | No | Day of week (derived from current time if omitted) |
| `C1`–`C14` | float | No | Count variables |
| `D1`–`D15` | float | No | Time delta variables |
| `V1`–`V339` | float | No | Vesta enrichment features |
| `M1`–`M9` | string | No | Match flags |

Additional fields accepted via Pydantic `extra="allow"`.

**Response:**
```json
{
  "transaction_id": 999,
  "fraud_probability": 0.4846,
  "decision": "REVIEW",
  "risk_level": "MEDIUM",
  "explanation": "This transaction shows MODERATE RISK (48.5% fraud probability)...",
  "agent_scores": {
    "vibe_checker": 0.4122,
    "agent_ensemble": 0.6241,
    "era_tracker": 0.5000,
    "og_check": 0.7484
  },
  "rule_violations": ["HIGH_AMOUNT", "LATE_NIGHT"],
  "processing_time_ms": 426.26,
  "timestamp": "2026-03-25T12:00:00+00:00"
}
```

| Response Field | Type | Description |
|---|---|---|
| `fraud_probability` | float (0–1) | Final fused fraud score |
| `decision` | string | `APPROVE`, `REVIEW`, or `BLOCK` |
| `risk_level` | string | `LOW`, `MEDIUM`, or `HIGH` |
| `explanation` | string | SHAP-enriched explanation text |
| `agent_scores` | object | Per-agent scores |
| `rule_violations` | array | Triggered OG Check rules |
| `processing_time_ms` | float | End-to-end latency |

---

#### `POST /api/v1/predict/batch`

Score multiple transactions in a single request. Bounded by `BATCH_MAX_SIZE` (default 256).

**Request Body:**
```json
{
  "transactions": [
    {"TransactionAmt": 100, "ProductCD": "W"},
    {"TransactionAmt": 7000, "ProductCD": "W", "hour": 2}
  ]
}
```

**Response:**
```json
{
  "predictions": [ ... ],
  "total_processed": 2,
  "total_time_ms": 850.5
}
```

Returns HTTP `413` if the batch exceeds the configured maximum size.

---

#### `POST /api/v1/explain`

Score a transaction and return a detailed SHAP + LLM explanation. This is the primary explainability endpoint.

**Request Body:** Same as `/api/v1/predict`.

**Response:**
```json
{
  "transaction_id": 999,
  "fraud_probability": 0.4846,
  "decision": "REVIEW",
  "risk_level": "MEDIUM",
  "summary": "LOW RISK — Fraud probability 48.5%, 2 rule violation(s)",
  "natural_language_explanation": "This transaction was flagged for review primarily because...",
  "top_risk_factors": [
    "Payment Count (C14) pushes risk up (SHAP +1.4911)",
    "Transaction Amount ($) pushes risk up (SHAP +0.9125)",
    "High OG Check score (75%)"
  ],
  "top_features": [
    {
      "feature_name": "Payment Count (C14)",
      "raw_name": "C14",
      "feature_value": 0.0,
      "shap_value": 1.491071,
      "direction": "increases_risk"
    },
    {
      "feature_name": "Transaction Amount ($)",
      "raw_name": "TransactionAmt",
      "feature_value": 450.0,
      "shap_value": 0.912484,
      "direction": "increases_risk"
    }
  ],
  "shap_available": true,
  "llm_used": true,
  "confidence_factors": [
    "1 high-severity rule(s) support the prediction",
    "Prediction grounded in SHAP feature attributions"
  ],
  "recommended_action": "REVIEW — Queue for manual analyst review",
  "agent_scores": { ... },
  "rule_violations": ["HIGH_AMOUNT", "LATE_NIGHT"],
  "processing_time_ms": 1250.3,
  "timestamp": "2026-03-25T12:00:00+00:00"
}
```

| Additional Field | Type | Description |
|---|---|---|
| `summary` | string | One-line risk summary |
| `natural_language_explanation` | string | LLM-generated plain-English explanation (or template fallback) |
| `top_risk_factors` | array | Human-readable risk drivers |
| `top_features` | array | SHAP feature attributions (up to 10) |
| `shap_available` | boolean | Whether SHAP values were computed |
| `llm_used` | boolean | Whether the LLM was called successfully |
| `confidence_factors` | array | Agent agreement / divergence signals |
| `recommended_action` | string | Suggested next step (BLOCK/CHALLENGE/REVIEW/MONITOR/APPROVE) |

---

### Metadata

#### `GET /api/v1/agents`

Returns the current agent configuration, weights, thresholds, and execution parameters.

#### `GET /api/v1/workflow`

Returns the LangGraph workflow description and a Mermaid diagram for visualisation.

---

## Request Tracing

Every response includes:

| Header | Description |
|---|---|
| `X-Request-ID` | Unique request identifier (pass your own via request header or auto-generated) |
| `X-Process-Time-ms` | Total request processing time in milliseconds |

---

## Error Responses

| Status | Condition |
|---|---|
| `422` | Request validation failure (missing `TransactionAmt`, invalid types) |
| `413` | Batch size exceeds `BATCH_MAX_SIZE` |
| `503` | Orchestrator not initialised (service starting up) |
| `500` | Unhandled internal error (logged with request ID) |
