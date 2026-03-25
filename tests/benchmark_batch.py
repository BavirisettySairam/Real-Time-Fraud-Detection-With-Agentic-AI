"""Benchmark: batch endpoint vs N sequential single calls.

Usage:
    python tests/benchmark_batch.py [--host HOST] [--n N]

Requires the API to be running.
"""

import argparse
import time
import json
import statistics
import requests

SAMPLE_TRANSACTIONS = [
    {"TransactionAmt": 120.0, "ProductCD": "W", "card1": 10001, "hour": 13},
    {"TransactionAmt": 6500.0, "ProductCD": "W", "card1": 222, "hour": 3},
    {"TransactionAmt": 20.0, "ProductCD": "H", "card1": 777, "hour": 18},
    {"TransactionAmt": 999.0, "ProductCD": "W", "card1": 555, "hour": 0},
    {"TransactionAmt": 50.0, "ProductCD": "C", "card1": 888, "hour": 10},
]


def benchmark_sequential(host: str, transactions: list, repeats: int = 5) -> dict:
    """Send each transaction individually and accumulate wall-clock time."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for txn in transactions:
            r = requests.post(f"{host}/api/v1/predict", json=txn, timeout=30)
            r.raise_for_status()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
    return {
        "method": "sequential",
        "txn_count": len(transactions),
        "repeats": repeats,
        "mean_s": statistics.mean(timings),
        "median_s": statistics.median(timings),
        "min_s": min(timings),
        "max_s": max(timings),
    }


def benchmark_batch(host: str, transactions: list, repeats: int = 5) -> dict:
    """Send all transactions in one batch call."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        r = requests.post(
            f"{host}/api/v1/predict/batch",
            json={"transactions": transactions},
            timeout=30,
        )
        r.raise_for_status()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
    return {
        "method": "batch",
        "txn_count": len(transactions),
        "repeats": repeats,
        "mean_s": statistics.mean(timings),
        "median_s": statistics.median(timings),
        "min_s": min(timings),
        "max_s": max(timings),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch vs sequential benchmark")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=5, help="Number of transactions per round")
    parser.add_argument("--repeats", type=int, default=5, help="Rounds to average over")
    args = parser.parse_args()

    transactions = SAMPLE_TRANSACTIONS[: args.n]

    # Warm up
    requests.post(f"{args.host}/api/v1/predict", json=transactions[0], timeout=30)

    seq = benchmark_sequential(args.host, transactions, args.repeats)
    batch = benchmark_batch(args.host, transactions, args.repeats)

    speedup = seq["mean_s"] / batch["mean_s"] if batch["mean_s"] > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"Batch vs Sequential Benchmark  ({args.n} transactions, {args.repeats} repeats)")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Mean (s)':<12} {'Median (s)':<12} {'Min (s)':<10} {'Max (s)':<10}")
    print(f"{'-'*60}")
    print(f"{'Sequential':<15} {seq['mean_s']:<12.3f} {seq['median_s']:<12.3f} {seq['min_s']:<10.3f} {seq['max_s']:<10.3f}")
    print(f"{'Batch':<15} {batch['mean_s']:<12.3f} {batch['median_s']:<12.3f} {batch['min_s']:<10.3f} {batch['max_s']:<10.3f}")
    print(f"{'-'*60}")
    print(f"Speedup: {speedup:.2f}x faster with batch")
    print(f"{'='*60}\n")

    return seq, batch, speedup


if __name__ == "__main__":
    seq, batch, speedup = main()
