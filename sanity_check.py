"""Sanity check: run 3 transactions through the orchestrator directly."""
import time
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from ml.preprocessing import FeaturePipeline
from services.orchestrator.agent_orchestrator import AgentOrchestrator, OrchestratorConfig

# --- Load pipeline once (mirrors what the API gateway does) ---
pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")

config = OrchestratorConfig(use_langgraph=False, enable_parallel=False)
orch = AgentOrchestrator(config=config)

transactions = {
    "Txn1 (low-risk)": {
        "TransactionAmt": 50,
        "card1": 1000,
        "card4": "visa",
        "card6": "debit",
    },
    "Txn2 (medium)": {
        "TransactionAmt": 500,
        "card1": 2000,
        "card4": "mastercard",
        "card6": "credit",
    },
    "Txn3 (high-risk)": {
        "TransactionAmt": 5000,
        "card1": 9999,
        "card4": "visa",
        "card6": "credit",
    },
}

scores = []
for label, txn in transactions.items():
    # Preprocess exactly like the API gateway does
    features_df = pipeline.transform(pd.DataFrame([txn]))
    pipeline_features = features_df.values[0].astype(np.float32)

    t0 = time.perf_counter()
    result = orch.analyze(txn, pipeline_features=pipeline_features)
    elapsed = (time.perf_counter() - t0) * 1000
    score = result["final_score"]
    decision = result["final_decision"]
    scores.append(score)
    print(f"{label} score: {score:.4f}  decision: {decision}  "
          f"[vibe={result['vibe_score']:.4f} era={result['era_score']:.4f} og={result['og_score']:.4f}]  "
          f"({elapsed:.0f}ms)")

print()
if len(set(f"{s:.4f}" for s in scores)) == 1:
    print("WARNING: all scores identical — something may be wrong")
elif all(abs(s - 0.5) < 0.02 for s in scores):
    print("WARNING: all scores ~0.5 — models may not be loaded")
else:
    print("OK: scores vary across inputs")

orch.close()
