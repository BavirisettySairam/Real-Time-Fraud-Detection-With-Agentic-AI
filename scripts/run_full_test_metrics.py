from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_lightgbm import LightGBMTrainer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    models_dir = root / "models"
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "lightgbm_model.txt"
    metrics_path = models_dir / "lightgbm_metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = lgb.Booster(model_file=str(model_path))
    num_features = int(model.num_feature())

    threshold = 0.5
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            threshold = float(json.load(fp).get("best_threshold", 0.5))

    trainer = LightGBMTrainer()
    print("Building category mappings from sampled train data...")
    df_train = trainer.load_and_prepare_data(data_dir, max_samples=120000)
    trainer.extract_features(df_train, fit=True)
    del df_train
    gc.collect()

    print("Loading full test data...")
    df_test = trainer.load_test_data(data_dir)
    X_test, _ = trainer.extract_features(df_test, fit=False)

    if X_test.shape[1] < num_features:
        for idx in range(X_test.shape[1], num_features):
            X_test[f"pad_{idx}"] = 0.0
    if X_test.shape[1] > num_features:
        X_test = X_test.iloc[:, :num_features]

    print("Running full test prediction...")
    scores = model.predict(X_test.values.astype(np.float32)).astype(float)
    scores = np.clip(scores, 0.0, 1.0)
    pred = (scores >= threshold).astype(int)

    txids = df_test["TransactionID"].astype(int).values
    ranked_idx = np.argsort(scores)[::-1][:20]
    top20 = [
        {
            "TransactionID": int(txids[idx]),
            "fraud_score": float(scores[idx]),
            "TransactionAmt": float(df_test.iloc[idx].get("TransactionAmt", 0.0) or 0.0),
            "ProductCD": str(df_test.iloc[idx].get("ProductCD", "")),
        }
        for idx in ranked_idx
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "test_transaction + test_identity (full rows)",
        "rows": int(len(scores)),
        "model_num_features": num_features,
        "processed_feature_count": int(X_test.shape[1]),
        "used_threshold": threshold,
        "predicted_fraud_count": int(pred.sum()),
        "predicted_fraud_rate": float(pred.mean()),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_quantiles": {
            "p01": float(np.quantile(scores, 0.01)),
            "p05": float(np.quantile(scores, 0.05)),
            "p10": float(np.quantile(scores, 0.10)),
            "p25": float(np.quantile(scores, 0.25)),
            "p50": float(np.quantile(scores, 0.50)),
            "p75": float(np.quantile(scores, 0.75)),
            "p90": float(np.quantile(scores, 0.90)),
            "p95": float(np.quantile(scores, 0.95)),
            "p99": float(np.quantile(scores, 0.99)),
            "p999": float(np.quantile(scores, 0.999)),
        },
        "threshold_counts": {
            str(t): {
                "count": int((scores >= t).sum()),
                "rate": float((scores >= t).mean()),
            }
            for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        "top20_test_predictions": top20,
        "note": "Test set has no labels; supervised metrics (ROC-AUC/PR-AUC/F1/precision/recall/log-loss) are not computable.",
    }

    out_json = models_dir / "lightgbm_test_full_metrics.json"
    out_log = logs_dir / "lightgbm_test_full_metrics.log"

    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    with out_log.open("a", encoding="utf-8") as fp:
        fp.write("=" * 100 + "\n")
        fp.write(payload["generated_at"] + "\n")
        fp.write(json.dumps(payload) + "\n")

    print(f"Wrote {out_json}")
    print(f"Logged {out_log}")
    print(f"rows={payload['rows']}")
    print(f"predicted_fraud_count={payload['predicted_fraud_count']}")
    print(f"predicted_fraud_rate={payload['predicted_fraud_rate']:.6f}")
    print(f"score_mean={payload['score_mean']:.6f}")
    print(f"score_p95={payload['score_quantiles']['p95']:.6f}")
    print(f"score_p99={payload['score_quantiles']['p99']:.6f}")


if __name__ == "__main__":
    main()
