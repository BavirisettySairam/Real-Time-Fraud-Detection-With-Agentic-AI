# Load Test Report

**Date**: 2026-03-25  
**Tool**: Locust 2.43.3 (headless mode)  
**Target**: `http://127.0.0.1:8001` (single-worker uvicorn)  
**Duration**: 60 seconds  
**Users**: 10 concurrent (ramp: 2 users/sec)

---

## Summary

| Metric | predict_single | predict_batch (3 txns) | Aggregated |
|--------|---------------|----------------------|------------|
| Requests | 334 | 115 | 449 |
| Failures | 0 | 0 | 0 |
| p50 (ms) | 880 | 1,200 | 940 |
| p95 (ms) | 1,500 | 1,900 | 1,600 |
| p99 (ms) | 1,700 | 2,000 | 1,900 |
| Min (ms) | 274 | 406 | 274 |
| Max (ms) | 1,793 | 2,257 | 2,257 |
| Avg (ms) | 912 | 1,203 | 987 |
| RPS | 5.6 | 1.9 | 7.6 |

## Configuration

- **API**: FastAPI + uvicorn (1 worker, no gunicorn)
- **Orchestrator**: LangGraph workflow, parallel agent execution enabled
- **Preprocessing**: FeaturePipeline (175 features) transforms raw input before agent inference
- **Agents**: Vibe Checker (LGB+XGB), Era Tracker (CatBoost), OG Check (LGB), The Yapper (SHAP)
- **Rate limiting**: Disabled for this test (set to 9,999/min)
- **Hardware**: Local development machine

## Key Findings

1. **Zero failures** across 449 requests — the API handled 10 concurrent users without errors.
2. **p50 latency ~880ms** for single predictions — includes FeaturePipeline preprocessing + 4 agents in parallel.
3. **Preprocessing overhead**: ~260ms added vs pre-pipeline baseline (p50 was 620ms, now 880ms).
4. **Batch overhead is proportional**: batch p50 ~1,200ms vs single 880ms (~36% increase for 3 transactions).
5. **Throughput**: ~7.6 requests/second sustained on a single uvicorn worker.
6. **Tail latency**: p99 (1,900ms) is ~2x p50 (940ms) — higher tail due to pipeline transform variability.

## Recommendations

- Add gunicorn with multiple uvicorn workers for production (`--workers 4`).
- Cache FeaturePipeline transform results for repeated transactions to reduce overhead.
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
