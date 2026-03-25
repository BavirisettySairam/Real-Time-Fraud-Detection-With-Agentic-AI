# =============================================================================
# ERA TRACKER - 24-hour transaction window behavioral analysis (CatBoost)
# =============================================================================

import json
import time as _time
import logging
import math
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "era_tracker_catboost.cbm"
METRICS_PATH = Path(__file__).resolve().parents[2] / "models" / "era_tracker_metrics.json"

WINDOW_SECONDS = 86400  # 24 hours


@dataclass
class EraTrackerResult:
    """Result from 24-hour window analysis."""
    fraud_score: float
    anomaly_score: float
    pattern_deviations: List[str]
    window_txn_count: int
    explanation: str


class EraTracker:
    """
    Era Tracker — analyses the user's transactions in the last 24 hours.

    Maintains a per-user sliding window of transactions (Redis or in-memory).
    Engineers behavioural features from that window and scores via a trained
    CatBoost model.  Falls back to heuristics when the model is unavailable.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        max_users_in_memory: int = 10000,
        model_path: Optional[str] = None,
        window_seconds: int = WINDOW_SECONDS,
    ):
        self.window_seconds = window_seconds
        self.redis_client = redis_client
        self.max_users_in_memory = max_users_in_memory
        self._history_lock = Lock()
        self.model_loaded = False
        self._cat_indices: List[int] = []
        self._all_features: List[str] = []

        # Bounded in-memory LRU cache: user_id -> list of txn dicts
        self.user_history: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()

        self.model = None
        self.threshold = 0.5
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Optional[str] = None) -> None:
        path = Path(model_path) if model_path else MODEL_PATH
        try:
            from catboost import CatBoostClassifier
            if path.exists():
                self.model = CatBoostClassifier()
                self.model.load_model(str(path))
                self.model_loaded = True
                logger.info("Era Tracker CatBoost model loaded from %s", path)
                metrics_path = Path(model_path).parent / "era_tracker_metrics.json" if model_path else METRICS_PATH
                if metrics_path.exists():
                    meta = json.loads(metrics_path.read_text())
                    self.threshold = meta.get("best_threshold", 0.5)
                    self._cat_indices = meta.get("cat_indices", [])
                    self._all_features = meta.get("all_features", [])
            else:
                logger.warning("Era Tracker model not found at %s — using heuristic fallback", path)
        except ImportError:
            logger.warning("catboost not installed — using heuristic fallback")
        except Exception as e:
            logger.warning("Failed to load Era Tracker model: %s — using heuristic fallback", e)

    # ------------------------------------------------------------------
    # History management (24-hour sliding window)
    # ------------------------------------------------------------------

    def _get_user_history(self, user_id: str, current_dt: float) -> List[Dict[str, Any]]:
        """Return transactions for *user_id* within the 24-hour window."""
        if self.redis_client:
            try:
                import json as _json
                key = f"era:{user_id}"
                raw_list = self.redis_client.lrange(key, 0, -1)
                history = [_json.loads(r) for r in raw_list]
                cutoff = current_dt - self.window_seconds
                return [h for h in history if h.get("TransactionDT", 0) >= cutoff]
            except Exception as e:
                logger.warning("Redis fetch failed: %s", e)

        with self._history_lock:
            if user_id in self.user_history:
                self.user_history.move_to_end(user_id)
                cutoff = current_dt - self.window_seconds
                return [h for h in self.user_history[user_id] if h.get("TransactionDT", 0) >= cutoff]
        return []

    def _update_user_history(self, user_id: str, transaction: Dict[str, Any]) -> None:
        txn_summary = {
            'TransactionAmt': transaction.get('TransactionAmt') or 0,
            'ProductCD': transaction.get('ProductCD') or 'unknown',
            'card4': transaction.get('card4') or 'unknown',
            'card6': transaction.get('card6') or 'unknown',
            'hour': transaction.get('hour') or 0,
            'TransactionDT': transaction.get('TransactionDT') or 0,
        }

        if self.redis_client:
            try:
                import json as _json
                key = f"era:{user_id}"
                self.redis_client.rpush(key, _json.dumps(txn_summary))
                # Keep a generously sized list; window filtering happens on read.
                self.redis_client.ltrim(key, -200, -1)
                self.redis_client.expire(key, self.window_seconds + 3600)
                return
            except Exception as e:
                logger.warning("Redis update failed: %s", e)

        with self._history_lock:
            if user_id not in self.user_history:
                self.user_history[user_id] = []
            self.user_history[user_id].append(txn_summary)
            # Prune entries older than the window.
            cutoff = txn_summary["TransactionDT"] - self.window_seconds
            self.user_history[user_id] = [
                h for h in self.user_history[user_id] if h.get("TransactionDT", 0) >= cutoff
            ]
            self.user_history.move_to_end(user_id)
            if len(self.user_history) > self.max_users_in_memory:
                self.user_history.popitem(last=False)

    # ------------------------------------------------------------------
    # Analysis entry point
    # ------------------------------------------------------------------

    def analyze(self, transaction: Dict[str, Any], pipeline_features: Optional[np.ndarray] = None) -> EraTrackerResult:
        """Analyse a transaction using pipeline features + sliding-window features.

        Args:
            transaction: Raw transaction dict (used for sliding-window computation
                         and heuristic fallback).
            pipeline_features: 1-D array of preprocessed features from the
                               centralised FeaturePipeline (175 features).
        """
        # Defensively replace None values with safe defaults so downstream
        # float()/int() calls never receive NoneType.
        transaction = {
            k: (v if v is not None else 0) for k, v in transaction.items()
        }
        user_id = str(transaction.get('card1') or 'unknown')
        current_dt = float(transaction.get('TransactionDT') or 0)

        history = self._get_user_history(user_id, current_dt)
        self._update_user_history(user_id, transaction)

        pattern_deviations = self._detect_pattern_deviations(transaction, history)

        if self.model is not None and pipeline_features is not None:
            sliding = self._build_sliding_window_features(transaction, history)
            combined = np.concatenate([
                np.asarray(pipeline_features, dtype=np.float64).ravel(),
                np.asarray(sliding, dtype=np.float64),
            ])
            proba = self.model.predict_proba([combined])
            raw = float(proba[0][1]) if len(proba[0]) > 1 else float(proba[0][0])
            fraud_score = max(0.0, min(raw, 1.0))
            anomaly_score = fraud_score
        else:
            anomaly_score = self._heuristic_anomaly(transaction, history)
            seq_risk = self._heuristic_sequence_risk(transaction, history)
            fraud_score = 0.5 * anomaly_score + 0.5 * seq_risk

        explanation = self._generate_explanation(fraud_score, anomaly_score, pattern_deviations, len(history))

        return EraTrackerResult(
            fraud_score=fraud_score,
            anomaly_score=anomaly_score,
            pattern_deviations=pattern_deviations,
            window_txn_count=len(history),
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Feature engineering (must match training)
    # ------------------------------------------------------------------

    def _build_sliding_window_features(
        self, transaction: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> List[float]:
        """Build 24 sliding-window behavioural features (must match training order)."""
        amt = float(transaction.get("TransactionAmt", 0))
        hist_amts = [float(h.get("TransactionAmt", 0)) for h in history]

        wc = len(history)
        w_mean = float(np.mean(hist_amts)) if hist_amts else 0.0
        w_median = float(np.median(hist_amts)) if hist_amts else 0.0
        w_max = float(np.max(hist_amts)) if hist_amts else 0.0
        w_std = float(np.std(hist_amts)) if hist_amts else 1.0
        w_total = float(sum(hist_amts)) if hist_amts else 0.0

        safe_std = w_std if w_std > 0 else 1.0
        zscore = max(-10.0, min((amt - w_mean) / safe_std, 10.0)) if hist_amts else 0.0
        ratio_mean = min(amt / (w_mean or 1.0), 100.0)
        ratio_median = min(amt / (w_median or 1.0), 100.0)
        ratio_max = min(amt / (w_max or 1.0), 100.0)

        current_dt = float(transaction.get("TransactionDT", 0))
        W_1H = 3600
        W_10M = 600

        # 1h window count
        recent_1h = [h for h in history if float(h.get("TransactionDT", 0)) >= current_dt - W_1H]
        tc_1h = len(recent_1h)

        # Time gaps
        if history:
            last_dt = float(history[-1].get("TransactionDT", 0))
            tsl = max(0.0, min(current_dt - last_dt, 999999.0))
            all_dts = [float(h.get("TransactionDT", 0)) for h in history]
            if len(all_dts) >= 2:
                gaps = [all_dts[i+1] - all_dts[i] for i in range(len(all_dts)-1)]
                a_gap = float(np.mean(gaps))
                mn_gap = float(np.min(gaps))
            else:
                a_gap = float(self.window_seconds)
                mn_gap = float(self.window_seconds)
        else:
            tsl = 999999.0
            a_gap = float(self.window_seconds)
            mn_gap = float(self.window_seconds)

        # Temporal
        cur_hour = int(transaction.get("hour", 12))
        h_rad = 2 * math.pi * cur_hour / 24.0
        hour_sin = math.sin(h_rad)
        hour_cos = math.cos(h_rad)
        is_night = 1 if cur_hour in (0,1,2,3,4,5,22,23) else 0
        is_wknd = 1 if int(transaction.get("day", 0)) in (5, 6) else 0

        # Flags
        rapid = 1 if tsl < 60 else 0
        is_new = 1 if wc == 0 else 0

        prior_nights = sum(1 for h in history if int(h.get("hour", 12)) in (0,1,2,3,4,5,22,23))
        night_first = 1 if (is_night and prior_nights == 0 and wc > 0) else 0

        if len(hist_amts) >= 2:
            incr = 1 if (amt > hist_amts[-1] > hist_amts[-2]) else 0
        else:
            incr = 0

        # Hour deviation (circular)
        hist_hours = [int(h.get("hour", 12)) for h in history]
        if hist_hours:
            h_dev = abs(cur_hour - float(np.mean(hist_hours)))
            h_dev = min(h_dev, 24 - h_dev)
        else:
            h_dev = 0.0

        # Product diversity
        products = set(str(h.get("ProductCD", "unknown")) for h in history)
        prod_div = len(products)

        # Burst in 10min
        recent_10m = [h for h in history if float(h.get("TransactionDT", 0)) >= current_dt - W_10M]
        burst_10m = sum(float(h.get("TransactionAmt", 0)) for h in recent_10m)

        # 24 numeric sliding-window features (must match training order)
        return [
            zscore, ratio_mean, ratio_median, ratio_max,
            wc, tc_1h, w_total, w_mean, w_max, w_std,
            tsl, a_gap, mn_gap,
            hour_sin, hour_cos, is_night, is_wknd,
            rapid, is_new, night_first, incr,
            h_dev, prod_div, burst_10m,
        ]

    # ------------------------------------------------------------------
    # Heuristic fallbacks
    # ------------------------------------------------------------------

    def _heuristic_anomaly(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
        if not history:
            return 0.3
        score = 0.0
        amt = transaction.get('TransactionAmt', 0)
        hist_amts = [h.get('TransactionAmt', 0) for h in history]
        mean_a = np.mean(hist_amts)
        std_a = np.std(hist_amts) + 1e-6
        z = abs(amt - mean_a) / std_a
        if z > 3:
            score += 0.4
        elif z > 2:
            score += 0.2

        cur_hour = transaction.get('hour', 12)
        hist_hours = [h.get('hour', 12) for h in history]
        if cur_hour not in set(hist_hours):
            score += 0.1
        if cur_hour in [2, 3, 4, 5] and not any(h in [2, 3, 4, 5] for h in hist_hours):
            score += 0.2
        return min(score, 1.0)

    def _heuristic_sequence_risk(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
        if len(history) < 2:
            return 0.2
        risk = 0.0
        amounts = [h.get('TransactionAmt', 0) for h in history[-5:]] + [transaction.get('TransactionAmt', 0)]
        if len(amounts) >= 3 and all(amounts[i] < amounts[i+1] for i in range(len(amounts)-1)):
            risk += 0.3
        times = [h.get('TransactionDT', 0) for h in history[-3:]]
        if len(times) >= 2:
            diffs = np.diff(times)
            if all(d < 60 for d in diffs):
                risk += 0.3
        return min(risk, 1.0)

    def _detect_pattern_deviations(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> List[str]:
        deviations = []
        if not history:
            deviations.append("No transactions in last 24hrs — new or inactive user")
            return deviations
        amt = transaction.get('TransactionAmt', 0)
        hist_amts = [h.get('TransactionAmt', 0) for h in history]
        if hist_amts and amt > 2 * max(hist_amts):
            deviations.append(f"Amount {amt:.2f} is 2x+ higher than 24hr max {max(hist_amts):.2f}")
        if len(history) >= 10:
            deviations.append(f"High velocity: {len(history)} transactions in 24hrs")
        cur_hour = transaction.get('hour', 12)
        if cur_hour in [2, 3, 4, 5]:
            hist_hours = [h.get('hour', 12) for h in history]
            if not any(h in [2, 3, 4, 5] for h in hist_hours):
                deviations.append(f"Unusual timing — transaction at hour {cur_hour}")
        return deviations

    def _generate_explanation(
        self, fraud_score: float, anomaly_score: float,
        deviations: List[str], window_count: int,
    ) -> str:
        parts = []
        if self.model_loaded:
            parts.append(f"Era Tracker model score: {fraud_score:.2%}")
        parts.append(f"24hr window: {window_count} prior transactions")
        if fraud_score > 0.7:
            parts.append("High-risk pattern in recent activity")
        if anomaly_score > 0.5:
            parts.append("Transaction deviates from 24hr behaviour")
        for d in deviations[:2]:
            parts.append(d)
        if not parts:
            parts.append("Transaction consistent with recent 24hr profile")
        return "; ".join(parts)
