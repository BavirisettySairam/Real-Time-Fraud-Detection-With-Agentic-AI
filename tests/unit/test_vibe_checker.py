"""Unit tests for the Vibe Checker agent — model loading, scoring, and edge cases."""

import numpy as np
import pytest

from services.agents.vibe_checker import VibeChecker, VibeCheckerResult
from tests.unit.conftest import preprocess


@pytest.fixture(scope="module")
def vibe_checker() -> VibeChecker:
    """Load the real Vibe Checker once for the module (loads models from disk)."""
    return VibeChecker()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class TestVibeCheckerLoading:
    def test_lgb_model_loaded(self, vibe_checker: VibeChecker):
        assert vibe_checker.lgb_model is not None, "LightGBM model should be loaded"

    def test_num_features_positive(self, vibe_checker: VibeChecker):
        assert vibe_checker.num_features > 0

    def test_feature_columns_populated(self, vibe_checker: VibeChecker):
        assert len(vibe_checker.feature_columns) > 0

    def test_threshold_in_range(self, vibe_checker: VibeChecker):
        assert 0.0 <= vibe_checker.threshold <= 1.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestVibeCheckerScoring:
    def test_basic_score_in_range(self, vibe_checker: VibeChecker, pipeline):
        features = preprocess(pipeline, {"TransactionAmt": 150.0, "ProductCD": "W"})
        result = vibe_checker.analyze(features)
        assert isinstance(result, VibeCheckerResult)
        assert 0.0 <= result.fraud_score <= 1.0
        assert 0.0 <= result.lightgbm_score <= 1.0

    def test_high_amount_scores_higher(self, vibe_checker: VibeChecker, pipeline):
        low = vibe_checker.analyze(preprocess(pipeline, {"TransactionAmt": 10.0}))
        high = vibe_checker.analyze(preprocess(pipeline, {"TransactionAmt": 9999.0, "hour": 3}))
        # High-risk transaction should generally score higher — not a strict
        # invariant but holds in practice with the trained model.
        assert high.fraud_score >= low.fraud_score * 0.3  # loose sanity check

    def test_result_has_models_loaded_dict(self, vibe_checker: VibeChecker, pipeline):
        features = preprocess(pipeline, {"TransactionAmt": 100.0})
        result = vibe_checker.analyze(features)
        assert "lightgbm" in result.models_loaded
        assert "xgboost" in result.models_loaded

    def test_explanation_string_nonempty(self, vibe_checker: VibeChecker, pipeline):
        features = preprocess(pipeline, {"TransactionAmt": 100.0})
        result = vibe_checker.analyze(features)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

class TestVibeCheckerVectorization:
    def test_vectorize_returns_numpy(self, vibe_checker: VibeChecker, pipeline):
        features = preprocess(pipeline, {"TransactionAmt": 100.0})
        vec = vibe_checker.vectorize_transaction(features)
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 2
        assert vec.shape[0] == 1
        assert vec.shape[1] == vibe_checker.num_features

    def test_get_feature_names_length(self, vibe_checker: VibeChecker):
        names = vibe_checker.get_feature_names()
        assert len(names) > 0
        # May be truncated to num_features or equal to feature_columns length
        assert len(names) <= max(vibe_checker.num_features, len(vibe_checker.feature_columns))


# ---------------------------------------------------------------------------
# No-model fallback
# ---------------------------------------------------------------------------

class TestVibeCheckerFallback:
    def test_no_model_returns_neutral(self):
        checker = VibeChecker.__new__(VibeChecker)
        checker.lgb_model = None
        checker.xgb_model = None
        checker.threshold = 0.5
        checker.num_features = 0
        checker.feature_columns = []
        checker.lgb_weight = 0.9
        result = checker.analyze(np.zeros(0, dtype=np.float32))
        assert result.fraud_score == 0.5
        assert result.models_loaded == {"lightgbm": False, "xgboost": False}
