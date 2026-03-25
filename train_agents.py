#!/usr/bin/env python
"""
Train all fraud-detection agents from the IEEE-CIS 60/20/20 split.

Agents trained:
  1. Vibe Checker  — XGBoost + LightGBM ensemble (+ interaction features)
  2. Era Tracker   — LightGBM on 26 per-user sliding-window features
  3. OG Check      — LightGBM on 20 rule + data-signal features

Logs all metrics with timestamp to logs/training_metrics.log (append mode).

Usage:
    python train_agents.py                     # train all
    python train_agents.py --agent vibe        # train only Vibe Checker
    python train_agents.py --agent era         # train only Era Tracker
    python train_agents.py --agent og          # train only OG Check
"""

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

METRICS_LOG_PATH = LOG_DIR / "training_metrics.log"


# ============================================================================
# METRICS LOGGING  (append with timestamp — never overwrite)
# ============================================================================

def log_metrics_to_file(agent_name: str, metrics: Dict[str, Any],
                        y_true: np.ndarray = None, y_proba: np.ndarray = None,
                        threshold: float = None) -> None:
    """Append full metrics block to the shared training log file."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "",
        "=" * 72,
        f"  {agent_name}  —  {ts}",
        "=" * 72,
    ]

    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"):
        val = metrics.get(key)
        if val is not None:
            lines.append(f"  {key:<12s}: {val:.6f}")

    # Confusion matrix (if raw arrays provided)
    if y_true is not None and y_proba is not None and threshold is not None:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        lines.append(f"  Confusion Matrix:")
        lines.append(f"    TN={tn:>7d}  FP={fp:>7d}")
        lines.append(f"    FN={fn:>7d}  TP={tp:>7d}")
        metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp),
                                        "fn": int(fn), "tp": int(tp)}
    elif "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        lines.append(f"  Confusion Matrix:")
        lines.append(f"    TN={cm['tn']:>7d}  FP={cm['fp']:>7d}")
        lines.append(f"    FN={cm['fn']:>7d}  TP={cm['tp']:>7d}")

    # Extra keys
    for key in sorted(metrics):
        if key not in ("roc_auc", "pr_auc", "precision", "recall", "f1",
                        "threshold", "confusion_matrix"):
            lines.append(f"  {key}: {metrics[key]}")

    lines.append("")

    text = "\n".join(lines)
    logger.info(text)
    with open(METRICS_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


# ============================================================================
# DATA HELPERS
# ============================================================================

def load_split(split: str) -> pd.DataFrame:
    """Load a split ('train', 'test', 'val') from the 60/20/20 files."""
    txn_path = DATA_DIR / f"{split}_transaction.csv"
    idn_path = DATA_DIR / f"{split}_identity.csv"
    logger.info("Loading %s transactions from %s …", split, txn_path)
    df = pd.read_csv(txn_path)
    if idn_path.exists():
        idn = pd.read_csv(idn_path)
        df = df.merge(idn, on="TransactionID", how="left")
    logger.info("  %s: %d rows, fraud rate %.2f%%", split, len(df), df["isFraud"].mean() * 100)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionDT" not in df.columns:
        return df
    tdt = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0)
    df["hour"] = ((tdt // 3600) % 24).astype(np.int16)
    df["day"] = ((tdt // 86400) % 7).astype(np.int16)
    df["is_weekend"] = df["day"].isin([5, 6]).astype(np.int8)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(np.int8)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionAmt" in df.columns:
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0.0)
        df["TransactionAmt_log"] = np.log1p(np.clip(amt, 0, None))
        df["TransactionAmt_sqrt"] = np.sqrt(np.clip(amt, 0, None))
        df["TransactionAmt_cents"] = np.mod(np.round(amt * 100), 100).astype(np.float32)
    if {"card1", "addr1"}.issubset(df.columns):
        df["card_addr_combo"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
    if {"P_emaildomain", "R_emaildomain"}.issubset(df.columns):
        df["email_domain_pair"] = df["P_emaildomain"].astype(str) + "_" + df["R_emaildomain"].astype(str)
    df["missing_count"] = df.isna().sum(axis=1).astype(np.float32)

    # --- Interaction features ---
    if "TransactionAmt" in df.columns:
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0.0)
        is_night = df["is_night"].fillna(0) if "is_night" in df.columns else 0
        addr_missing = df["addr1"].isna().astype(np.int8) if "addr1" in df.columns else 0
        device_missing = df.get("DeviceInfo", pd.Series(dtype=str)).fillna("").eq("").astype(np.int8)
        df["night_x_amount"] = is_night * amt
        df["addr_missing_x_amount"] = addr_missing * amt
        df["device_missing_x_amount"] = device_missing * amt

    return df


def optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    valid = prec >= 0.15
    f1_masked = np.where(valid, f1, -1)
    idx = int(np.argmax(f1_masked))
    if f1_masked[idx] < 0:
        idx = int(np.argmax(f1))
    thr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return thr, float(f1[idx])


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                    threshold: float) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }


# ============================================================================
# 1. VIBE CHECKER  (XGBoost + LightGBM ensemble + interaction features)
# ============================================================================

class VibeCheckerTrainer:
    def __init__(self):
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.feature_columns: List[str] = []

    def extract_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        X = add_time_features(df.copy())
        X = add_engineered_features(X)
        drop_cols = ["TransactionID", "isFraud"]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        X = X.fillna(0)

        cat_cols = X.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            txt = X[col].astype(str)
            if fit:
                mapping = {v: i for i, v in enumerate(txt.unique())}
                self.category_mappings[col] = mapping
            else:
                mapping = self.category_mappings.get(col, {})
            X[col] = txt.map(mapping).fillna(-1).astype(np.int32)

        if fit:
            self.feature_columns = list(X.columns)
        else:
            for name in self.feature_columns:
                if name not in X.columns:
                    X[name] = 0
            X = X[self.feature_columns]

        X = X.astype(np.float32)
        y = df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df))
        return X, y

    @staticmethod
    def _train_lgb(X_train, y_train, X_val, y_val, n_folds: int = 3):
        logger.info("--- LightGBM Training ---")
        scale_pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        params = {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
            "boosting_type": "gbdt",
            "num_leaves": 127,
            "learning_rate": 0.03,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "scale_pos_weight": scale_pw,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "min_data_in_leaf": 40,
            "num_threads": -1,
            "verbose": -1,
            "seed": 42,
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_auc = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
            d_tr = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
            d_va = lgb.Dataset(X_train[va_idx], label=y_train[va_idx], reference=d_tr)
            m = lgb.train(params, d_tr, num_boost_round=2500,
                          valid_sets=[d_va],
                          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
            cv_auc.append(roc_auc_score(y_train[va_idx], m.predict(X_train[va_idx])))
            logger.info("  Fold %d ROC-AUC: %.4f", fold, cv_auc[-1])
        logger.info("  CV ROC-AUC: %.4f ± %.4f", np.mean(cv_auc), np.std(cv_auc))

        d_tr = lgb.Dataset(X_train, label=y_train)
        d_va = lgb.Dataset(X_val, label=y_val, reference=d_tr)
        model = lgb.train(params, d_tr, num_boost_round=2500,
                          valid_sets=[d_tr, d_va],
                          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])
        return model, cv_auc

    @staticmethod
    def _train_xgb(X_train, y_train, X_val, y_val, n_folds: int = 3):
        import xgboost as xgb
        logger.info("--- XGBoost Training ---")
        scale_pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": scale_pw,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "min_child_weight": 40,
            "seed": 42,
            "verbosity": 0,
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_auc = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
            dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx])
            dval = xgb.DMatrix(X_train[va_idx], label=y_train[va_idx])
            m = xgb.train(params, dtrain, num_boost_round=2500,
                          evals=[(dval, "val")],
                          early_stopping_rounds=200, verbose_eval=0)
            preds = m.predict(dval)
            cv_auc.append(roc_auc_score(y_train[va_idx], preds))
            logger.info("  Fold %d ROC-AUC: %.4f", fold, cv_auc[-1])
        logger.info("  CV ROC-AUC: %.4f ± %.4f", np.mean(cv_auc), np.std(cv_auc))

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(params, dtrain, num_boost_round=2500,
                          evals=[(dtrain, "train"), (dval, "val")],
                          early_stopping_rounds=200, verbose_eval=100)
        return model, cv_auc

    @staticmethod
    def _find_best_lgb_weight(lgb_scores, xgb_scores, y_true):
        best_w, best_auc = 0.5, 0.0
        for w in np.arange(0.0, 1.01, 0.05):
            ensemble = w * lgb_scores + (1 - w) * xgb_scores
            auc = roc_auc_score(y_true, ensemble)
            if auc > best_auc:
                best_w, best_auc = w, auc
        logger.info("  Best lgb_weight=%.2f  ROC-AUC=%.6f", best_w, best_auc)
        return float(round(best_w, 2))

    def train(self, df_train, df_val, n_folds=3):
        logger.info("=" * 70)
        logger.info("VIBE CHECKER — XGBoost + LightGBM Ensemble")
        logger.info("=" * 70)

        X_train, y_train = self.extract_features(df_train, fit=True)
        X_val, y_val = self.extract_features(df_val, fit=False)
        X_tr = X_train.values.astype(np.float32)
        X_va = X_val.values.astype(np.float32)
        y_tr = y_train.astype(np.int32)
        y_va = y_val.astype(np.int32)

        lgb_model, lgb_cv = self._train_lgb(X_tr, y_tr, X_va, y_va, n_folds)
        try:
            xgb_model, xgb_cv = self._train_xgb(X_tr, y_tr, X_va, y_va, n_folds)
            xgb_available = True
        except ImportError:
            logger.warning("xgboost not installed — LightGBM only")
            xgb_model, xgb_cv = None, []
            xgb_available = False

        lgb_val = np.clip(lgb_model.predict(X_va), 0, 1)
        if xgb_available:
            import xgboost as xgb
            xgb_val = np.clip(xgb_model.predict(xgb.DMatrix(X_va)), 0, 1)
            lgb_weight = self._find_best_lgb_weight(lgb_val, xgb_val, y_va)
            ensemble_val = lgb_weight * lgb_val + (1 - lgb_weight) * xgb_val
        else:
            lgb_weight = 1.0
            xgb_val = lgb_val
            ensemble_val = lgb_val

        thr, _ = optimal_threshold(y_va, ensemble_val)
        lgb_met = compute_metrics(y_va, lgb_val, thr)
        xgb_met = compute_metrics(y_va, xgb_val, thr) if xgb_available else {}
        ens_met = compute_metrics(y_va, ensemble_val, thr)

        logger.info("LightGBM  — ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f",
                     lgb_met["roc_auc"], lgb_met["pr_auc"], lgb_met["f1"])
        if xgb_available:
            logger.info("XGBoost   — ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f",
                         xgb_met["roc_auc"], xgb_met["pr_auc"], xgb_met["f1"])
        logger.info("Ensemble  — ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f  (w=%.2f, thr=%.4f)",
                     ens_met["roc_auc"], ens_met["pr_auc"], ens_met["f1"], lgb_weight, thr)

        # Log to file
        log_metrics_to_file("Vibe Checker (Ensemble)", ens_met, y_va, ensemble_val, thr)
        log_metrics_to_file("Vibe Checker (LightGBM)", lgb_met, y_va, lgb_val, thr)
        if xgb_available:
            log_metrics_to_file("Vibe Checker (XGBoost)", xgb_met, y_va, xgb_val, thr)

        # Save
        lgb_path = MODEL_DIR / "vibe_lgb.txt"
        lgb_model.save_model(str(lgb_path))
        if xgb_available:
            xgb_path = MODEL_DIR / "vibe_xgb.json"
            xgb_model.save_model(str(xgb_path))

        metrics_data = {
            "agent": "vibe_checker",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "best_threshold": round(thr, 6),
            "lgb_weight": lgb_weight,
            "ensemble": {k: round(v, 6) if isinstance(v, float) else v for k, v in ens_met.items()},
            "lightgbm": {k: round(v, 6) if isinstance(v, float) else v for k, v in lgb_met.items()},
            "xgboost": {k: round(v, 6) if isinstance(v, float) else v for k, v in xgb_met.items()},
            "lgb_cv_roc_auc": [round(x, 6) for x in lgb_cv],
            "xgb_cv_roc_auc": [round(x, 6) for x in xgb_cv],
            "num_features": int(X_train.shape[1]),
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
        }
        (MODEL_DIR / "vibe_metrics.json").write_text(json.dumps(metrics_data, indent=2))
        return metrics_data


# ============================================================================
# 2. ERA TRACKER  (CatBoost on ~25 behavioral + temporal features)
# ============================================================================

# Numeric behavioral features engineered from per-user history
ERA_NUM_FEATURES = [
    # Amount deviation signals (current vs user baseline)
    "amt_zscore_user",        # z-score vs user's historical mean
    "amt_ratio_user_mean",    # ratio to user's mean amount
    "amt_ratio_user_median",  # ratio to user's median amount
    "amt_ratio_user_max",     # ratio to user's max amount
    "amt_log",                # log1p of current amount
    # Velocity features (user's recent activity)
    "user_txn_count_24h",     # number of user txns in 24h window
    "user_txn_count_1h",      # number of user txns in 1h window
    "user_total_amt_24h",     # total amount in 24h
    "user_mean_amt_24h",      # mean amount in 24h
    "user_max_amt_24h",       # max amount in 24h
    "user_std_amt_24h",       # std of amounts in 24h
    # Time-gap signals
    "time_since_last",        # seconds since user's last txn
    "avg_gap_24h",            # mean gap between txns in 24h
    "min_gap_24h",            # smallest gap in 24h (burst)
    # Temporal context
    "hour_sin",               # sin(2π·h/24) for circular hour encoding
    "hour_cos",               # cos(2π·h/24)
    "is_night",               # 1 if hour in [22-5]
    "is_weekend",             # 1 if day in [5,6]
    # Behavioral pattern flags
    "rapid_succession",       # 1 if <60s since last
    "is_new_user",            # 1 if first txn for this card
    "night_first_time",       # 1 if night txn but user never had one
    "increasing_amounts",     # 1 if amt > last > second-last
    "hour_deviation",         # circular distance from user's mean hour
    "product_diversity_24h",  # distinct ProductCD in 24h window
    "burst_amt_10min",        # total amount in last 10 minutes
]

# Categorical features (CatBoost handles natively)
ERA_CAT_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
]

ERA_ALL_FEATURES = ERA_NUM_FEATURES + ERA_CAT_FEATURES


def _engineer_era_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer per-user behavioral features for the Era Tracker.

    Simulates a sliding-window view: for each transaction, looks back at the
    same user's (card1) transactions in the preceding 24h / 1h / 10min to build
    velocity, deviation, and consistency signals.
    """
    logger.info("Engineering Era Tracker CatBoost features (per-user sliding window) …")
    df = df.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)

    amt_arr = df["TransactionAmt"].fillna(0).astype(np.float64).values
    card_arr = df["card1"].fillna(-1).values
    tdt_arr = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).values.astype(np.float64)
    hour_arr = df["hour"].fillna(12).astype(int).values if "hour" in df.columns else np.full(len(df), 12, dtype=int)
    day_arr = df["day"].fillna(0).astype(int).values if "day" in df.columns else np.zeros(len(df), dtype=int)
    prod_arr = df["ProductCD"].fillna("unknown").astype(str).values if "ProductCD" in df.columns else np.full(len(df), "unknown", dtype=object)

    n = len(df)
    num_result = np.zeros((n, len(ERA_NUM_FEATURES)), dtype=np.float64)

    card_change = np.concatenate(([True], card_arr[1:] != card_arr[:-1]))
    group_starts = np.where(card_change)[0]

    WINDOW_24H = 86400
    WINDOW_1H = 3600
    WINDOW_10M = 600

    for g in range(len(group_starts)):
        start = group_starts[g]
        end = group_starts[g + 1] if g + 1 < len(group_starts) else n

        user_hours_seen = []

        for i in range(start, end):
            cur_t = tdt_arr[i]
            cur_amt = amt_arr[i]
            cur_hour = hour_arr[i]

            # Collect indices within windows
            cutoff_24h = cur_t - WINDOW_24H
            cutoff_1h = cur_t - WINDOW_1H
            cutoff_10m = cur_t - WINDOW_10M

            hist_24h = [j for j in range(start, i) if tdt_arr[j] >= cutoff_24h]
            hist_1h = [j for j in hist_24h if tdt_arr[j] >= cutoff_1h]
            hist_10m = [j for j in hist_24h if tdt_arr[j] >= cutoff_10m]

            h_amts = amt_arr[hist_24h] if hist_24h else np.array([])
            wc = len(hist_24h)

            # Amount deviation signals
            w_mean = float(h_amts.mean()) if wc > 0 else 0.0
            w_median = float(np.median(h_amts)) if wc > 0 else 0.0
            w_max = float(h_amts.max()) if wc > 0 else 0.0
            w_std = float(h_amts.std()) if wc > 0 else 1.0
            w_total = float(h_amts.sum()) if wc > 0 else 0.0

            safe_std = w_std if w_std > 0 else 1.0
            zscore = max(-10.0, min((cur_amt - w_mean) / safe_std, 10.0)) if wc > 0 else 0.0
            ratio_mean = min(cur_amt / (w_mean or 1.0), 100.0)
            ratio_median = min(cur_amt / (w_median or 1.0), 100.0)
            ratio_max = min(cur_amt / (w_max or 1.0), 100.0)
            amt_log = float(np.log1p(max(cur_amt, 0)))

            # Velocity
            tc_1h = len(hist_1h)

            # Time gaps
            if hist_24h:
                tsl = max(0.0, min(cur_t - tdt_arr[hist_24h[-1]], 999999.0))
                hist_dts = tdt_arr[hist_24h]
                if len(hist_dts) >= 2:
                    gaps = np.diff(hist_dts)
                    a_gap = float(gaps.mean())
                    mn_gap = float(gaps.min())
                else:
                    a_gap = float(WINDOW_24H)
                    mn_gap = float(WINDOW_24H)
            else:
                tsl = 999999.0
                a_gap = float(WINDOW_24H)
                mn_gap = float(WINDOW_24H)

            # Temporal
            h_rad = 2 * np.pi * cur_hour / 24.0
            hour_sin = float(np.sin(h_rad))
            hour_cos = float(np.cos(h_rad))
            is_night = 1 if cur_hour in (0, 1, 2, 3, 4, 5, 22, 23) else 0
            is_wknd = 1 if day_arr[i] in (5, 6) else 0

            # Behavioral flags
            rapid = 1 if tsl < 60 else 0
            is_new = 1 if i == start else 0

            prior_nights = sum(1 for j in hist_24h if hour_arr[j] in (0, 1, 2, 3, 4, 5, 22, 23))
            night_first = 1 if (is_night and prior_nights == 0 and wc > 0) else 0

            if len(h_amts) >= 2:
                incr = 1 if (cur_amt > h_amts[-1] > h_amts[-2]) else 0
            else:
                incr = 0

            # Hour deviation (circular)
            user_hours_seen.append(cur_hour)
            if len(user_hours_seen) > 1:
                h_dev = abs(cur_hour - float(np.mean(user_hours_seen[:-1])))
                h_dev = min(h_dev, 24 - h_dev)
            else:
                h_dev = 0.0

            # Product diversity in 24h
            prod_div = len(set(prod_arr[j] for j in hist_24h)) if hist_24h else 0

            # Burst amount in 10 min
            burst_10m = float(amt_arr[hist_10m].sum()) if hist_10m else 0.0

            num_result[i] = [
                zscore, ratio_mean, ratio_median, ratio_max, amt_log,
                wc, tc_1h, w_total, w_mean, w_max, w_std,
                tsl, a_gap, mn_gap,
                hour_sin, hour_cos, is_night, is_wknd,
                rapid, is_new, night_first, incr,
                h_dev, prod_div, burst_10m,
            ]

    out = pd.DataFrame(num_result, columns=ERA_NUM_FEATURES)

    # Add categorical columns
    if "ProductCD" in df.columns:
        out["ProductCD"] = df["ProductCD"].fillna("unknown").astype(str).values
    else:
        out["ProductCD"] = "unknown"
    if "card4" in df.columns:
        out["card4"] = df["card4"].fillna("unknown").astype(str).values
    else:
        out["card4"] = "unknown"
    if "card6" in df.columns:
        out["card6"] = df["card6"].fillna("unknown").astype(str).values
    else:
        out["card6"] = "unknown"

    out["isFraud"] = df["isFraud"].values
    logger.info("  Generated %d rows × %d features (%d num + %d cat)",
                len(out), len(ERA_ALL_FEATURES), len(ERA_NUM_FEATURES), len(ERA_CAT_FEATURES))
    return out


def train_era_tracker(df_train, df_val, n_folds=3):
    from catboost import CatBoostClassifier, Pool

    logger.info("=" * 70)
    logger.info("ERA TRACKER (CatBoost) — %d behavioral features", len(ERA_ALL_FEATURES))
    logger.info("=" * 70)

    era_train = _engineer_era_features(df_train)
    era_val = _engineer_era_features(df_val)

    X_train = era_train[ERA_ALL_FEATURES].copy()
    y_train = era_train["isFraud"].values.astype(np.int32)
    X_val = era_val[ERA_ALL_FEATURES].copy()
    y_val = era_val["isFraud"].values.astype(np.int32)

    cat_indices = [ERA_ALL_FEATURES.index(c) for c in ERA_CAT_FEATURES]

    logger.info("  Train: %d  Val: %d  Fraud rate: %.2f%%  Cat indices: %s",
                len(X_train), len(X_val), y_train.mean() * 100, cat_indices)

    scale_pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        cat_features=cat_indices,
        verbose=200,
        early_stopping_rounds=200,
        use_best_model=True,
        task_type="CPU",
    )

    train_pool = Pool(X_train, label=y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_indices)

    model.fit(train_pool, eval_set=val_pool)

    val_proba = model.predict_proba(val_pool)[:, 1]
    val_scores = np.clip(val_proba, 0, 1)

    thr, _ = optimal_threshold(y_val, val_scores)
    met = compute_metrics(y_val, val_scores, thr)

    logger.info("Era Tracker (CatBoost) — ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f  (thr=%.4f)",
                met["roc_auc"], met["pr_auc"], met["f1"], thr)

    # Feature importance
    fi = model.get_feature_importance()
    fi_sorted = sorted(zip(ERA_ALL_FEATURES, fi), key=lambda x: -x[1])
    logger.info("  Feature importance (top 15):")
    for fname, imp in fi_sorted[:15]:
        logger.info("    %-30s  %.1f", fname, imp)

    log_metrics_to_file("Era Tracker (CatBoost)", met, y_val, val_scores, thr)

    model_path = MODEL_DIR / "era_tracker_catboost.cbm"
    model.save_model(str(model_path))
    logger.info("Saved Era Tracker CatBoost → %s", model_path)

    metrics_data = {
        "agent": "era_tracker",
        "model_type": "catboost",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_threshold": round(thr, 6),
        "num_features": len(ERA_NUM_FEATURES),
        "cat_features": ERA_CAT_FEATURES,
        "all_features": ERA_ALL_FEATURES,
        "cat_indices": cat_indices,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in met.items()},
        "best_iteration": model.get_best_iteration(),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
    }
    (MODEL_DIR / "era_tracker_metrics.json").write_text(json.dumps(metrics_data, indent=2))
    return metrics_data


# ============================================================================
# 3. OG CHECK  (LightGBM on 20 rule + data-signal features)
# ============================================================================

OG_FEATURES = [
    # Original binary rules
    "rule_max_amount",
    "rule_high_amount",
    "rule_micro_amount",
    "rule_late_night",
    "rule_weekend_high",
    "rule_round_amount",
    "rule_email_mismatch",
    "rule_no_device_high_value",
    "rule_is_night",
    "rule_amount_log",
    "rule_addr_missing",
    "rule_card_not_visa",
    # --- NEW high-signal features ---
    "rule_addr2_missing",
    "rule_D1_missing",
    "rule_D1_high",
    "rule_C1_high",
    "rule_C13_high",
    "rule_M_mismatch_count",
    "rule_id_missing",
    "rule_moderate_spike",
]


def _learn_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    amt = df["TransactionAmt"].fillna(0)
    y = df["isFraud"]
    fraud_amts = amt[y == 1]
    c1 = df.get("C1", pd.Series(dtype=float)).fillna(0)
    c13 = df.get("C13", pd.Series(dtype=float)).fillna(0)
    return {
        "max_single": round(float(np.percentile(amt.dropna(), 99)), 2),
        "high_risk_threshold": round(float(np.percentile(fraud_amts.dropna(), 75)), 2),
        "micro_threshold": 1.0,
        "moderate_spike": round(float(2 * amt.median()), 2),
        "C1_p95": round(float(np.percentile(c1.dropna(), 95)), 2),
        "C13_p95": round(float(np.percentile(c13.dropna(), 95)), 2),
        "D1_p95": round(float(np.percentile(df.get("D1", pd.Series(dtype=float)).dropna(), 95)), 2),
    }


def _engineer_og_features(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    amt = df["TransactionAmt"].fillna(0)
    hour = df["hour"].fillna(12).astype(int) if "hour" in df.columns else pd.Series(12, index=df.index)
    is_weekend = df["is_weekend"].fillna(0).astype(int) if "is_weekend" in df.columns else pd.Series(0, index=df.index)

    f = pd.DataFrame(index=df.index)

    # Original features
    f["rule_max_amount"] = (amt > thresholds["max_single"]).astype(np.int8)
    f["rule_high_amount"] = (amt > thresholds["high_risk_threshold"]).astype(np.int8)
    f["rule_micro_amount"] = ((amt > 0) & (amt < thresholds["micro_threshold"])).astype(np.int8)
    f["rule_late_night"] = hour.isin([2, 3, 4, 5]).astype(np.int8)
    f["rule_weekend_high"] = ((is_weekend == 1) & (amt > 1000)).astype(np.int8)
    f["rule_round_amount"] = amt.isin({100, 200, 500, 1000, 2000, 5000}).astype(np.int8)

    p_email = df.get("P_emaildomain", pd.Series(dtype=str)).fillna("")
    r_email = df.get("R_emaildomain", pd.Series(dtype=str)).fillna("")
    f["rule_email_mismatch"] = ((p_email != "") & (r_email != "") & (p_email != r_email)).astype(np.int8)

    device_info = df.get("DeviceInfo", pd.Series(dtype=str)).fillna("")
    f["rule_no_device_high_value"] = ((device_info == "") & (amt > 500)).astype(np.int8)
    f["rule_is_night"] = df["is_night"].fillna(0).astype(np.int8) if "is_night" in df.columns else pd.Series(0, index=df.index)
    f["rule_amount_log"] = np.log1p(amt.clip(lower=0))
    f["rule_addr_missing"] = df.get("addr1", pd.Series(dtype=float)).isna().astype(np.int8)

    card4 = df.get("card4", pd.Series(dtype=str)).fillna("unknown")
    f["rule_card_not_visa"] = (card4 != "visa").astype(np.int8)

    # --- NEW features ---
    f["rule_addr2_missing"] = df.get("addr2", pd.Series(dtype=float)).isna().astype(np.int8)

    D1 = df.get("D1", pd.Series(dtype=float))
    f["rule_D1_missing"] = D1.isna().astype(np.int8)
    f["rule_D1_high"] = (D1.fillna(0) > thresholds.get("D1_p95", 999999)).astype(np.int8)

    C1 = df.get("C1", pd.Series(dtype=float)).fillna(0)
    f["rule_C1_high"] = (C1 > thresholds.get("C1_p95", 999999)).astype(np.int8)

    C13 = df.get("C13", pd.Series(dtype=float)).fillna(0)
    f["rule_C13_high"] = (C13 > thresholds.get("C13_p95", 999999)).astype(np.int8)

    # M-column mismatches: count of 'F' values in M1-M9
    m_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
    if m_cols:
        m_df = df[m_cols].fillna("")
        f["rule_M_mismatch_count"] = (m_df == "F").sum(axis=1).astype(np.float32)
    else:
        f["rule_M_mismatch_count"] = 0

    # Identity info missing (id_01 is NaN → no identity attached)
    f["rule_id_missing"] = df.get("id_01", pd.Series(dtype=float)).isna().astype(np.int8)

    # Moderate spike: amount > 2× global median
    f["rule_moderate_spike"] = (amt > thresholds.get("moderate_spike", 999999)).astype(np.int8)

    return f


def train_og_check(df_train, df_val):
    logger.info("=" * 70)
    logger.info("OG CHECK — LightGBM on 20 rule + data-signal features")
    logger.info("=" * 70)

    thresholds = _learn_thresholds(df_train)
    logger.info("  Learned thresholds: %s", thresholds)

    feat_train = _engineer_og_features(df_train, thresholds)
    feat_val = _engineer_og_features(df_val, thresholds)

    X_train = feat_train[OG_FEATURES].fillna(0).astype(np.float32).values
    y_train = df_train["isFraud"].values.astype(np.int32)
    X_val = feat_val[OG_FEATURES].fillna(0).astype(np.float32).values
    y_val = df_val["isFraud"].values.astype(np.int32)

    logger.info("  Train: %d  Val: %d  Fraud rate: %.2f%%",
                len(X_train), len(X_val), y_train.mean() * 100)

    scale_pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 3,
        "scale_pos_weight": scale_pw,
        "lambda_l1": 0.3,
        "lambda_l2": 0.3,
        "min_data_in_leaf": 100,
        "num_threads": -1,
        "verbose": -1,
        "seed": 42,
    }

    # CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_auc = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        d_tr = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx], feature_name=OG_FEATURES)
        d_va = lgb.Dataset(X_train[va_idx], label=y_train[va_idx], reference=d_tr, feature_name=OG_FEATURES)
        m = lgb.train(params, d_tr, num_boost_round=500,
                      valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        preds = m.predict(X_train[va_idx])
        auc_val = roc_auc_score(y_train[va_idx], preds)
        cv_auc.append(auc_val)
        logger.info("  Fold %d ROC-AUC: %.4f", fold, auc_val)
    logger.info("  CV ROC-AUC: %.4f ± %.4f", np.mean(cv_auc), np.std(cv_auc))

    # Final model
    d_tr = lgb.Dataset(X_train, label=y_train, feature_name=OG_FEATURES)
    d_va = lgb.Dataset(X_val, label=y_val, reference=d_tr, feature_name=OG_FEATURES)
    model = lgb.train(params, d_tr, num_boost_round=500,
                      valid_sets=[d_tr, d_va],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

    val_scores = np.clip(model.predict(X_val), 0, 1)
    thr, _ = optimal_threshold(y_val, val_scores)
    met = compute_metrics(y_val, val_scores, thr)

    logger.info("OG Check — ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f  (thr=%.4f)",
                met["roc_auc"], met["pr_auc"], met["f1"], thr)

    # Feature importance
    fi = model.feature_importance(importance_type="gain")
    fi_sorted = sorted(zip(OG_FEATURES, fi), key=lambda x: -x[1])
    logger.info("  Feature importance:")
    for fname, imp in fi_sorted:
        logger.info("    %-30s  gain=%.1f", fname, imp)

    log_metrics_to_file("OG Check", met, y_val, val_scores, thr)

    # Save model (OG Check is now LightGBM, not LogReg)
    og_model_path = MODEL_DIR / "og_check_lgb.txt"
    model.save_model(str(og_model_path))
    logger.info("Saved OG model → %s", og_model_path)

    # Also save params for the agent's rule-based path + model path
    params_data = {
        "thresholds": thresholds,
        "og_model_path": "og_check_lgb.txt",
        "features": OG_FEATURES,
        "best_threshold": round(thr, 6),
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in met.items()},
        "cv_roc_auc": [round(x, 6) for x in cv_auc],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "og_check_params.json").write_text(json.dumps(params_data, indent=2))
    return {**met, "agent": "og_check"}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train fraud-detection agents")
    parser.add_argument("--agent", choices=["vibe", "era", "og"], default=None,
                        help="Train only the specified agent (default: all)")
    parser.add_argument("--folds", type=int, default=3, help="CV folds (default: 3)")
    args = parser.parse_args()

    df_train = load_split("train")
    df_train = add_time_features(df_train)
    df_val = load_split("val")
    df_val = add_time_features(df_val)

    results = {}

    if args.agent in (None, "vibe"):
        vt = VibeCheckerTrainer()
        results["vibe_checker"] = vt.train(df_train, df_val, n_folds=args.folds)

    if args.agent in (None, "era"):
        results["era_tracker"] = train_era_tracker(df_train, df_val, n_folds=args.folds)

    if args.agent in (None, "og"):
        results["og_check"] = train_og_check(df_train, df_val)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    for name, met in results.items():
        roc = met.get("ensemble", met).get("roc_auc", met.get("roc_auc", 0))
        pr = met.get("ensemble", met).get("pr_auc", met.get("pr_auc", 0))
        f1v = met.get("ensemble", met).get("f1", met.get("f1", 0))
        logger.info("  %-15s ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f", name, roc, pr, f1v)
    logger.info("All metrics appended to %s", METRICS_LOG_PATH)


if __name__ == "__main__":
    main()
