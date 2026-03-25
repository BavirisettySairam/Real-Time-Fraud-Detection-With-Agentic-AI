# System Architecture

Technical reference for the fraud detection platform's internal design, agent pipeline, data flow, and score fusion logic.

**Author:** Bavirisetty Sairam
**Last Updated:** March 2026

---

## High-Level Design

The system follows a multi-agent scoring architecture. A single API gateway accepts transaction payloads. A LangGraph-based orchestrator fans the request out to three scoring agents in parallel, aggregates their outputs through dynamic fusion, then passes the result to an explainability agent before returning the response.

```
                          ┌──────────────────┐
                          │   Client / UI    │
                          └────────┬─────────┘
                                   │  HTTP POST
                                   ▼
                          ┌──────────────────┐
                          │  FastAPI Gateway  │
                          │  (main.py)        │
                          └────────┬─────────┘
                                   │  validate + enrich
                                   ▼
                    ┌──────────────────────────────┐
                    │    LangGraph Orchestrator     │
                    │    (agent_orchestrator.py)    │
                    └───┬──────────┬──────────┬────┘
                        │          │          │
              parallel  │          │          │  parallel
                        ▼          ▼          ▼
                  ┌──────────┐ ┌──────────┐ ┌──────────┐
                  │   Vibe   │ │   Era    │ │    OG    │
                  │ Checker  │ │ Tracker  │ │  Check   │
                  │ (LGB+XGB)│ │(CatBoost)│ │ (LGB)   │
                  └────┬─────┘ └────┬─────┘ └────┬─────┘
                       │            │             │
                       ▼            ▼             ▼
                  ┌─────────────────────────────────────┐
                  │        Decision Aggregator           │
                  │   dynamic weighting (see below)      │
                  └──────────────┬──────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   The Yapper    │
                        │  SHAP → LLM    │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  JSON Response  │
                        └─────────────────┘
```

---

## Agent Details

### Vibe Checker — Primary ML Ensemble

| Property | Value |
|---|---|
| Models | LightGBM Booster + XGBoost Booster |
| Blend | 90% LightGBM / 10% XGBoost |
| Features | 445 (raw + engineered) |
| Artifacts | `models/vibe_lgb.txt`, `models/vibe_xgb.json` |
| ROC-AUC | 0.9706 |

The ensemble uses both gradient-boosted tree models trained on the same feature set. Engineered features include `TransactionAmt_log`, `TransactionAmt_sqrt`, `TransactionAmt_cents`, time features (`hour`, `day`, `is_weekend`, `is_night`), `card_addr_combo`, `email_domain_pair`, and `missing_count`.

The `vectorize_transaction()` method is exposed publicly for SHAP consumption by The Yapper.

### Era Tracker — Behavioural Analysis

| Property | Value |
|---|---|
| Model | CatBoost Classifier |
| Features | 25 numeric + 3 categorical = 28 |
| Artifacts | `models/era_tracker_catboost.cbm` |
| ROC-AUC | 0.7813 |

Numeric features include per-user velocity metrics (`user_txn_count_24h`, `user_txn_count_1h`, `burst_amt_10min`), deviation measures (`amt_zscore_user`, `hour_deviation`), behavioural flags (`rapid_succession`, `night_first_time`, `increasing_amounts`), and circular time encoding (`hour_sin`, `hour_cos`).

Categorical features: `ProductCD`, `card4`, `card6` — handled natively by CatBoost without encoding.

Maintains an in-memory per-user transaction history and updates it on each call.

### OG Check — Rule + ML Hybrid

| Property | Value |
|---|---|
| Model | LightGBM on 20 rule-derived features |
| Artifacts | `models/og_check_lgb.txt`, `models/og_check_params.json` |
| ROC-AUC | 0.7833 |

12 original rule features (binary indicators for violations like `HIGH_AMOUNT`, `LATE_NIGHT`, `MISSING_EMAIL`, etc.) plus 8 engineered features: `addr2_missing`, `D1_missing`, `D1_high`, `C1_high`, `C13_high`, `M_mismatch_count`, `id_missing`, `moderate_spike`.

Falls back to heuristic rule-sum scoring if the LightGBM model is unavailable.

### The Yapper — Explainability

| Property | Value |
|---|---|
| SHAP | TreeExplainer on Vibe Checker's LightGBM model |
| LLM | OpenRouter API (configurable model, default `gpt-4o-mini`) |
| Fallback | Template-based explanation when LLM is unavailable |

Pipeline:
1. Run `shap.TreeExplainer.shap_values()` on the vectorised transaction.
2. Map raw feature names to human-readable labels (100+ mapped).
3. Sort by absolute SHAP value → top 8 features.
4. Build a structured prompt with transaction context, SHAP drivers, agent scores, and risk factors.
5. Send to LLM via OpenRouter with a system prompt that enforces plain English, no jargon.
6. Return the LLM explanation alongside SHAP feature breakdown, risk factors, confidence assessment, and recommended action.

---

## Score Fusion Logic

The decision aggregator uses dynamic weighting based on the Vibe Checker's confidence:

```
if vibe_score > 0.8:
    final_score = vibe_score                          # high-confidence pass-through
    score_source = "vibe_high_confidence"
else:
    final_score = 0.60 * vibe + 0.25 * era + 0.15 * og   # blended
    score_source = "dynamic_blend"
```

When no ML model is loaded (fallback mode):

```
final_score = era_weight * era + og_weight * og       # agents-only
score_source = "agents_fallback"
```

Decision thresholds:

| Score Range | Decision |
|---|---|
| ≥ 0.7 | **BLOCK** |
| 0.4 – 0.7 | **REVIEW** |
| < 0.4 | **APPROVE** |

---

## LangGraph Workflow

The orchestrator supports two execution modes:

**Parallel (default):**
- `START` fans out to `vibe_check`, `era_track`, `og_check` concurrently.
- All three converge at `decision`.
- `decision` → `yapper` → `END`.

**Sequential:**
- `START` → `vibe_check` → `era_track` → `og_check` → `decision` → `yapper` → `END`.

Configuration: `OrchestratorConfig.enable_parallel` (default `True`).

When LangGraph is not installed, the orchestrator falls back to a `_fast_analyze()` path that uses `ThreadPoolExecutor` for parallel agent calls without the LangGraph runtime.

---

## Timeout and Error Handling

- Per-agent timeout: configurable via `ORCHESTRATOR_TIMEOUT_MS` (default 5000ms).
- If any agent times out or throws, its score defaults to 0.5 (neutral).
- The orchestrator logs failures but never propagates agent exceptions to the API response.
- The Yapper catches LLM failures and falls back to template explanations.

---

## Data Flow

```
IEEE-CIS CSV files
       │
       ▼
train_agents.py           # 60/20/20 stratified split
       │
       ├──► models/vibe_lgb.txt + vibe_xgb.json
       ├──► models/era_tracker_catboost.cbm
       └──► models/og_check_lgb.txt + og_check_params.json
              │
              ▼
       Agent __init__()    # loads on startup
              │
              ▼
       FastAPI lifespan    # orchestrator + agents ready
              │
              ▼
       /predict or /explain requests
```

---

## Latency Profile

Measured under load (10 concurrent users, 60s, Locust):

| Metric | Single Predict | Batch (3 txns) |
|---|---|---|
| p50 | ~620ms | ~740ms |
| p95 | ~760ms | ~880ms |
| p99 | ~810ms | ~940ms |
| Throughput | ~8.4 RPS | ~1.9 RPS |

**Why ~620ms, not sub-100ms:** Each request runs 4 agents (3 ML models + SHAP explainability) in parallel via `ThreadPoolExecutor` on a single uvicorn worker on a local development machine. The bottleneck is CPU-bound tree inference across LightGBM, XGBoost, and CatBoost models, each operating on hundreds of features.

**Production path to 200–300ms:** Deploy with `gunicorn --workers 4 -k uvicorn.workers.UvicornWorker`, add Redis warm-cache for repeat cards/devices, and run on production-grade hardware (e.g., 4-core VM with dedicated CPU). For a fraud detection system where the alternative is manual review taking hours, 200–300ms per transaction is more than adequate.

---

## File Reference

| File | Purpose |
|---|---|
| `services/api_gateway/main.py` | FastAPI application, endpoints, Pydantic models |
| `services/orchestrator/agent_orchestrator.py` | LangGraph workflow, fusion logic, parallel execution |
| `services/agents/vibe_checker.py` | LightGBM + XGBoost ensemble inference |
| `services/agents/era_tracker.py` | CatBoost behavioural scoring |
| `services/agents/og_check.py` | Rule + LightGBM hybrid scoring |
| `services/agents/the_yapper.py` | SHAP + LLM explainability |
| `train_agents.py` | Training pipeline for all three trainable agents |
| `run.py` | Quick-start launcher (API + Streamlit) |
| `docker-compose.yml` | Container deployment (API + Frontend + Redis) |
