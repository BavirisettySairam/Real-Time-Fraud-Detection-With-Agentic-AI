# Load Test Report

**Date**: 2025-03-25  
**Tool**: Locust 2.43.3 (headless mode)  
**Target**: `http://127.0.0.1:8000` (single-worker uvicorn)  
**Duration**: 60 seconds  
**Users**: 10 concurrent (ramp: 2 users/sec)

---

## Summary

| Metric | predict_single | predict_batch (3 txns) | Aggregated |
|--------|---------------|----------------------|------------|
| Requests | 491 | 113 | 604 |
| Failures | 0 | 0 | 0 |
| p50 (ms) | 620 | 740 | 650 |
| p95 (ms) | 760 | 880 | 780 |
| p99 (ms) | 810 | 940 | 880 |
| Min (ms) | 411 | 480 | 411 |
| Max (ms) | 1,030 | 1,032 | 1,032 |
| Avg (ms) | 605 | 741 | 631 |
| RPS | 8.4 | 1.9 | 10.3 |

## Configuration

- **API**: FastAPI + uvicorn (1 worker, no gunicorn)
- **Orchestrator**: LangGraph workflow, parallel agent execution enabled
- **Agents**: Vibe Checker (LGB+XGB), Era Tracker (CatBoost), OG Check (LGB), The Yapper (SHAP)
- **Rate limiting**: Disabled for this test (set to 10,000/min)
- **Hardware**: Local development machine

## Key Findings

1. **Zero failures** across 604 requests — the API handled 10 concurrent users without errors.
2. **p50 latency ~620ms** for single predictions — dominated by ML inference (4 agents in parallel).
3. **Batch overhead is modest**: 3-transaction batch at ~740ms p50 vs 620ms single (only ~19% slower, not 3x).
4. **Throughput**: ~10.3 requests/second sustained on a single uvicorn worker.
5. **Tail latency is well-controlled**: p99 (880ms) is only ~35% above p50 (650ms).

## Recommendations

- Add gunicorn with multiple uvicorn workers for production (`--workers 4`).
- Consider async agent execution to improve per-request latency.
- The batch endpoint amortises overhead well and should be preferred for bulk scoring.

## How to Reproduce

### Load Test
```bash
# Start API (with high rate limits for load testing)
RATE_LIMIT_PREDICT=10000/minute RATE_LIMIT_DEFAULT=10000/minute \
  python -m uvicorn services.api_gateway.main:create_app --factory --host 0.0.0.0 --port 8000

# Run Locust
python -m locust -f tests/load_test.py --host http://127.0.0.1:8000 \
  --headless -u 10 -r 2 -t 60s --csv locust_results --only-summary
```

---

## Batch vs Sequential Benchmark

5 transactions per round, 5 repeats averaged.

| Method | Mean (s) | Median (s) | Min (s) | Max (s) |
|--------|----------|-----------|---------|---------|
| Sequential (5 × single) | 2.560 | 2.613 | 2.309 | 2.935 |
| Batch (1 call, 5 txns) | 0.753 | 0.746 | 0.710 | 0.825 |

**Speedup: 3.40x** — the batch endpoint is ~3.4x faster than making 5 sequential single calls.

### Why Batch Is Faster

- Eliminates 4 HTTP round-trips (connection overhead, JSON parsing, response serialization).
- The API uses `ThreadPoolExecutor` (up to `batch_max_workers=8`) to score transactions concurrently within a single batch request.
- Shared agent initialisation cost is amortised across all transactions in the batch.

### Reproduce
```bash
python tests/benchmark_batch.py --host http://127.0.0.1:8000 --n 5 --repeats 5
```
