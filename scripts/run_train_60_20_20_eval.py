from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_lightgbm import LightGBMTrainer


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob.astype(float), 1e-12, 1 - 1e-12)
    y_true = y_true.astype(float)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def summarize_split(name: str, y: np.ndarray) -> str:
    return f"{name}: rows={len(y)}, fraud_rate={float(np.mean(y)):.4%}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    models_dir = root / "models"
    logs_dir = root / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    max_samples_env = os.getenv("MAX_TRAIN_SAMPLES", "").strip()
    max_samples = int(max_samples_env) if max_samples_env else None

    trainer = LightGBMTrainer()

    print("Loading train_transaction + train_identity only...")
    df = trainer.load_and_prepare_data(data_dir, max_samples=max_samples)
    X, y = trainer.extract_features(df, fit=True)

    # 60/20/20 split using stratification.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,  # 0.25 of 80% = 20%
        random_state=42,
        stratify=y_train_val,
    )

    print(summarize_split("Train", y_train))
    print(summarize_split("Validation", y_val))
    print(summarize_split("Test", y_test))

    # LightGBM config aligned with main trainer.
    base_weight = (1 - y_train.mean()) / max(y_train.mean(), 0.01)
    params = {
        "objective": "binary",
        "metric": ["auc", "aucpr"],
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": 42,
        "verbose": -1,
        "scale_pos_weight": float(base_weight),
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "min_data_in_leaf": 40,
        "num_threads": -1,
    }

    train_data = lgb.Dataset(X_train.values.astype(np.float32), label=y_train.astype(np.int32))
    val_data = lgb.Dataset(X_val.values.astype(np.float32), label=y_val.astype(np.int32), reference=train_data)

    print("Training LightGBM on 60% split...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(period=100),
        ],
    )

    val_scores = model.predict(X_val.values.astype(np.float32)).astype(float)
    val_threshold, val_best_f1 = trainer.find_optimal_threshold(y_val, val_scores)
    val_metrics = trainer.compute_metrics(y_val, val_scores, val_threshold)
    val_metrics["rows"] = int(len(y_val))
    val_metrics["fraud_rate"] = float(np.mean(y_val))
    val_metrics["log_loss"] = compute_log_loss(y_val, val_scores)

    test_scores = model.predict(X_test.values.astype(np.float32)).astype(float)
    test_metrics = trainer.compute_metrics(y_test, test_scores, val_threshold)
    test_metrics["rows"] = int(len(y_test))
    test_metrics["fraud_rate"] = float(np.mean(y_test))
    test_metrics["log_loss"] = compute_log_loss(y_test, test_scores)

    # Optional: also report test-optimal threshold for diagnostics only.
    test_opt_threshold, test_opt_f1 = trainer.find_optimal_threshold(y_test, test_scores)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "train_transaction + train_identity only (test_* ignored)",
        "split_strategy": {
            "train": 0.60,
            "validation": 0.20,
            "test": 0.20,
            "random_state": 42,
            "stratified": True,
        },
        "rows": {
            "total": int(len(y)),
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "fraud_rate": {
            "total": float(np.mean(y)),
            "train": float(np.mean(y_train)),
            "validation": float(np.mean(y_val)),
            "test": float(np.mean(y_test)),
        },
        "model": {
            "num_features": int(X_train.shape[1]),
            "best_iteration": int(model.best_iteration or 0),
            "scale_pos_weight": float(base_weight),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "threshold_selection": {
            "selected_from_validation": float(val_threshold),
            "validation_best_f1": float(val_best_f1),
            "test_optimal_threshold_diagnostic": float(test_opt_threshold),
            "test_optimal_f1_diagnostic": float(test_opt_f1),
        },
    }

    model_out = models_dir / "lightgbm_model_60_20_20.txt"
    metrics_out = models_dir / "lightgbm_split_60_20_20_metrics.json"
    log_out = logs_dir / "lightgbm_split_60_20_20.log"

    model.save_model(str(model_out))
    with metrics_out.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    with log_out.open("a", encoding="utf-8") as fp:
        fp.write("=" * 100 + "\n")
        fp.write(payload["generated_at"] + "\n")
        fp.write(json.dumps(payload) + "\n")

    print(f"Wrote model: {model_out}")
    print(f"Wrote metrics: {metrics_out}")
    print(f"Appended log: {log_out}")
    print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.6f}")
    print(f"Validation PR-AUC: {val_metrics['pr_auc']:.6f}")
    print(f"Validation F1: {val_metrics['f1']:.6f} @ threshold {val_threshold:.6f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.6f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.6f}")
    print(f"Test F1: {test_metrics['f1']:.6f} @ validation-threshold {val_threshold:.6f}")


if __name__ == "__main__":
    main()
