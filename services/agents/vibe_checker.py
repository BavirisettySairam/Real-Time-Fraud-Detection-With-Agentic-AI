# =============================================================================
# VIBE CHECKER - ML Ensemble (XGBoost + LightGBM) for fraud scoring
# =============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VibeCheckerResult:
    """Result from the Vibe Checker ensemble."""
    fraud_score: float
    lightgbm_score: float
    xgboost_score: float
    threshold: float
    models_loaded: Dict[str, bool]
    explanation: str


class VibeChecker:
    """
    Vibe Checker — ML Ensemble agent (XGBoost + LightGBM).

    Loads both a LightGBM Booster and an XGBoost Booster, runs inference
    on both, and returns a weighted average as the ensemble fraud score.
    Falls back to whichever model is available if only one is loaded.
    """

    def __init__(
        self,
        lgb_model_path: Optional[str] = None,
        xgb_model_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        lgb_weight: float = 0.5,
    ):
        default_lgb = ROOT / "models" / "vibe_lgb.txt"
        default_xgb = ROOT / "models" / "vibe_xgb.json"
        default_metrics = ROOT / "models" / "vibe_metrics.json"
        default_data_dir = ROOT / "data"

        self.lgb_path = Path(lgb_model_path) if lgb_model_path else default_lgb
        self.xgb_path = Path(xgb_model_path) if xgb_model_path else default_xgb
        self.metrics_path = Path(metrics_path) if metrics_path else default_metrics
        self.data_dir = Path(data_dir) if data_dir else default_data_dir

        # Ensemble weight: lgb_weight for LightGBM, (1 - lgb_weight) for XGBoost
        self.lgb_weight = min(max(lgb_weight, 0.0), 1.0)

        self.lgb_model: Optional[lgb.Booster] = None
        self.xgb_model: Optional[Any] = None  # xgboost.Booster
        self.threshold: float = 0.5
        self.feature_columns: List[str] = []
        self.num_features: int = 0

        self._load_lgb_model()
        self._load_xgb_model()
        self._load_threshold()
        self._load_feature_columns()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_lgb_model(self) -> None:
        if not self.lgb_path.exists():
            logger.warning("LightGBM model not found at %s", self.lgb_path)
            return
        try:
            self.lgb_model = lgb.Booster(model_file=str(self.lgb_path))
            self.num_features = int(self.lgb_model.num_feature())
            logger.info("Loaded LightGBM model from %s (%d features)", self.lgb_path, self.num_features)
        except Exception as exc:
            logger.error("Failed to load LightGBM model: %s", exc)

    def _load_xgb_model(self) -> None:
        if not self.xgb_path.exists():
            logger.warning("XGBoost model not found at %s", self.xgb_path)
            return
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(self.xgb_path))
            self.xgb_model = booster
            # Sync num_features from XGBoost if LightGBM wasn't loaded.
            if self.num_features == 0:
                self.num_features = int(booster.num_features())
            logger.info("Loaded XGBoost model from %s", self.xgb_path)
        except ImportError:
            logger.warning("xgboost not installed — XGBoost model will not be used")
        except Exception as exc:
            logger.error("Failed to load XGBoost model: %s", exc)

    def _load_threshold(self) -> None:
        if not self.metrics_path.exists():
            return
        try:
            with self.metrics_path.open("r", encoding="utf-8") as fp:
                metrics = json.load(fp)
            value = float(metrics.get("best_threshold", 0.5))
            self.threshold = min(max(value, 0.0), 1.0)
            # Also load ensemble weight if saved.
            if "lgb_weight" in metrics:
                self.lgb_weight = min(max(float(metrics["lgb_weight"]), 0.0), 1.0)
        except Exception as exc:
            logger.warning("Failed to read metrics from %s: %s", self.metrics_path, exc)

    def _load_feature_columns(self) -> None:
        """Build feature column order from dataset headers (mirrors training preprocessing)."""
        trans_path = self.data_dir / "train_transaction.csv"
        identity_path = self.data_dir / "train_identity.csv"

        trans_cols = self._read_csv_header(trans_path) if trans_path.exists() else []
        identity_cols = self._read_csv_header(identity_path) if identity_path.exists() else []

        cols: List[str] = []
        for name in trans_cols + identity_cols:
            if name in {"TransactionID", "isFraud"}:
                continue
            if name not in cols:
                cols.append(name)

        engineered_cols = [
            "TransactionAmt_log",
            "TransactionAmt_sqrt",
            "TransactionAmt_cents",
            "hour",
            "day",
            "is_weekend",
            "is_night",
            "card_addr_combo",
            "email_domain_pair",
            "missing_count",
        ]

        if self.num_features > len(cols):
            for name in engineered_cols:
                if len(cols) >= self.num_features:
                    break
                if name not in cols:
                    cols.append(name)

        if self.num_features > 0:
            cols = cols[: self.num_features]

        # Fallback: try LightGBM model metadata.
        if not cols and self.lgb_model is not None:
            try:
                model_names = list(self.lgb_model.feature_name())
                cols = [n for n in model_names if n]
                if self.num_features > 0:
                    cols = cols[: self.num_features]
            except Exception:
                pass

        self.feature_columns = cols

    # ------------------------------------------------------------------
    # Feature engineering (mirrors training)
    # ------------------------------------------------------------------

    def _prepare_transaction_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(transaction)

        tdt_raw = out.get("TransactionDT")
        hour_raw = out.get("hour")
        day_raw = out.get("day")

        if tdt_raw is not None:
            tdt = self._to_numeric(tdt_raw)
            if hour_raw is None:
                out["hour"] = int((tdt // 3600) % 24)
            if day_raw is None:
                out["day"] = int((tdt // 86400) % 7)
        elif hour_raw is not None or day_raw is not None:
            hour = int(self._to_numeric(hour_raw)) if hour_raw is not None else 0
            day = int(self._to_numeric(day_raw)) if day_raw is not None else 0
            out["TransactionDT"] = float(day * 86400 + hour * 3600)
            out.setdefault("hour", hour)
            out.setdefault("day", day)

        amount = self._to_numeric(out.get("TransactionAmt", 0.0))
        clipped = max(amount, 0.0)
        out["TransactionAmt_log"] = float(np.log1p(clipped))
        out["TransactionAmt_sqrt"] = float(np.sqrt(clipped))
        out["TransactionAmt_cents"] = float((round(clipped * 100) % 100))

        hour = int(self._to_numeric(out.get("hour", 0)))
        day = int(self._to_numeric(out.get("day", 0)))
        out["hour"] = hour
        out["day"] = day
        out["is_weekend"] = 1 if day in {5, 6} else 0
        out["is_night"] = 1 if hour in {0, 1, 2, 3, 4, 5, 22, 23} else 0

        if "card_addr_combo" not in out:
            out["card_addr_combo"] = f"{out.get('card1', '')}_{out.get('addr1', '')}"
        if "email_domain_pair" not in out:
            out["email_domain_pair"] = f"{out.get('P_emaildomain', '')}_{out.get('R_emaildomain', '')}"

        if self.feature_columns:
            missing_count = sum(
                1 for f in self.feature_columns
                if out.get(f) is None or (isinstance(out.get(f), str) and out.get(f).strip() == "")
            )
            out["missing_count"] = float(missing_count)

        return out

    @staticmethod
    def _read_csv_header(path: Path) -> List[str]:
        try:
            with path.open("r", encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp)
                return next(reader)
        except Exception:
            return []

    @staticmethod
    def _to_numeric(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            return 0.0 if np.isnan(value) or np.isinf(value) else value
        text = str(value).strip()
        if not text:
            return 0.0
        try:
            numeric = float(text)
            return 0.0 if np.isnan(numeric) or np.isinf(numeric) else numeric
        except ValueError:
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()
            return float(int(digest[:8], 16) % 100000) / 100000.0

    def _vectorize_transaction(self, transaction: Dict[str, Any]) -> np.ndarray:
        if self.num_features <= 0:
            return np.zeros((1, 1), dtype=np.float32)
        prepared = self._prepare_transaction_features(transaction)
        vector = np.zeros((self.num_features,), dtype=np.float32)
        if self.feature_columns:
            for i, name in enumerate(self.feature_columns):
                vector[i] = np.float32(self._to_numeric(prepared.get(name, 0)))
        return vector.reshape(1, -1)

    # Public helpers for SHAP / explainability
    def vectorize_transaction(self, transaction: Dict[str, Any]) -> np.ndarray:
        return self._vectorize_transaction(transaction)

    def get_feature_names(self) -> List[str]:
        if self.feature_columns:
            return list(self.feature_columns)
        return [f"f_{i}" for i in range(max(self.num_features, 1))]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyze(self, transaction: Dict[str, Any]) -> VibeCheckerResult:
        lgb_loaded = self.lgb_model is not None
        xgb_loaded = self.xgb_model is not None

        if not lgb_loaded and not xgb_loaded:
            return VibeCheckerResult(
                fraud_score=0.5,
                lightgbm_score=0.5,
                xgboost_score=0.5,
                threshold=self.threshold,
                models_loaded={"lightgbm": False, "xgboost": False},
                explanation="No models available; using neutral fallback score",
            )

        x = self._vectorize_transaction(transaction)
        lgb_score = 0.5
        xgb_score = 0.5

        if lgb_loaded:
            try:
                raw = float(self.lgb_model.predict(x)[0])
                lgb_score = min(max(raw, 0.0), 1.0)
            except Exception as exc:
                logger.error("LightGBM inference failed: %s", exc)
                lgb_loaded = False

        if xgb_loaded:
            try:
                import xgboost as xgb
                dmat = xgb.DMatrix(x, feature_names=self.feature_columns[:x.shape[1]] if self.feature_columns else None)
                raw = float(self.xgb_model.predict(dmat)[0])
                xgb_score = min(max(raw, 0.0), 1.0)
            except Exception as exc:
                logger.error("XGBoost inference failed: %s", exc)
                xgb_loaded = False

        # Ensemble
        if lgb_loaded and xgb_loaded:
            score = self.lgb_weight * lgb_score + (1.0 - self.lgb_weight) * xgb_score
        elif lgb_loaded:
            score = lgb_score
        else:
            score = xgb_score

        score = min(max(score, 0.0), 1.0)
        status = "above" if score >= self.threshold else "below"

        return VibeCheckerResult(
            fraud_score=score,
            lightgbm_score=lgb_score,
            xgboost_score=xgb_score,
            threshold=self.threshold,
            models_loaded={"lightgbm": lgb_loaded, "xgboost": xgb_loaded},
            explanation=(
                f"Ensemble score {score:.3f} ({status} threshold {self.threshold:.3f}) "
                f"[LGB={lgb_score:.3f}, XGB={xgb_score:.3f}]"
            ),
        )
