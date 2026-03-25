# =============================================================================
# VIBE CHECKER - ML Ensemble (XGBoost + LightGBM) for fraud scoring
# =============================================================================

from __future__ import annotations

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

    Expects preprocessed features from the centralised FeaturePipeline
    (passed by the orchestrator).  Runs inference on both models and
    returns a weighted average as the ensemble fraud score.
    """

    def __init__(
        self,
        lgb_model_path: Optional[str] = None,
        xgb_model_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        lgb_weight: float = 0.5,
    ):
        default_lgb = ROOT / "models" / "vibe_lgb.txt"
        default_xgb = ROOT / "models" / "vibe_xgb.json"
        default_metrics = ROOT / "models" / "vibe_metrics.json"

        self.lgb_path = Path(lgb_model_path) if lgb_model_path else default_lgb
        self.xgb_path = Path(xgb_model_path) if xgb_model_path else default_xgb
        self.metrics_path = Path(metrics_path) if metrics_path else default_metrics

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
            if "lgb_weight" in metrics:
                self.lgb_weight = min(max(float(metrics["lgb_weight"]), 0.0), 1.0)
        except Exception as exc:
            logger.warning("Failed to read metrics from %s: %s", self.metrics_path, exc)

    def _load_feature_columns(self) -> None:
        """Get feature names from the LightGBM model (source of truth)."""
        if self.lgb_model is not None:
            try:
                self.feature_columns = list(self.lgb_model.feature_name())
            except Exception:
                pass
        if not self.feature_columns:
            self.feature_columns = [f"f_{i}" for i in range(max(self.num_features, 1))]

    # ------------------------------------------------------------------
    # Vectorisation from preprocessed pipeline features
    # ------------------------------------------------------------------

    def _vectorize_from_pipeline(self, features: np.ndarray) -> np.ndarray:
        """Reshape pipeline output to (1, n_features) float32 for model input."""
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    # Public helpers for SHAP / explainability
    def vectorize_transaction(self, features: np.ndarray) -> np.ndarray:
        """Vectorise preprocessed features for SHAP explainer."""
        return self._vectorize_from_pipeline(features)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_columns)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyze(self, features: np.ndarray) -> VibeCheckerResult:
        """Score a transaction using preprocessed pipeline features.

        Args:
            features: 1-D or 2-D array of preprocessed features from
                      FeaturePipeline.transform() (175 features).
        """
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

        x = self._vectorize_from_pipeline(features)
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
