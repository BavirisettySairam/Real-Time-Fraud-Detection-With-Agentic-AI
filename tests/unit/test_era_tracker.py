"""Unit tests for the Era Tracker agent — model loading, CatBoost scoring, history management."""

import pytest

from services.agents.era_tracker import EraTracker, EraTrackerResult


@pytest.fixture(scope="module")
def era_tracker() -> EraTracker:
    """Load the real Era Tracker once for the module."""
    return EraTracker()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class TestEraTrackerLoading:
    def test_catboost_model_loaded(self, era_tracker: EraTracker):
        assert era_tracker.model is not None, "CatBoost model should be loaded"
        assert era_tracker.model_loaded is True

    def test_threshold_in_range(self, era_tracker: EraTracker):
        assert 0.0 <= era_tracker.threshold <= 1.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestEraTrackerScoring:
    def test_basic_score_in_range(self, era_tracker: EraTracker):
        txn = {"TransactionAmt": 150.0, "ProductCD": "W", "card1": 99999,
               "card4": "visa", "card6": "debit", "hour": 14, "day": 2,
               "TransactionDT": 100000}
        result = era_tracker.analyze(txn)
        assert isinstance(result, EraTrackerResult)
        assert 0.0 <= result.fraud_score <= 1.0
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_result_has_pattern_deviations(self, era_tracker: EraTracker):
        txn = {"TransactionAmt": 500.0, "card1": 88888, "TransactionDT": 200000,
               "ProductCD": "W", "card4": "visa", "card6": "debit", "hour": 10}
        result = era_tracker.analyze(txn)
        assert isinstance(result.pattern_deviations, list)
        assert isinstance(result.window_txn_count, int)

    def test_explanation_nonempty(self, era_tracker: EraTracker):
        txn = {"TransactionAmt": 100.0, "card1": 77777, "TransactionDT": 300000,
               "ProductCD": "W", "card4": "visa", "card6": "debit", "hour": 12}
        result = era_tracker.analyze(txn)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_new_user_detected(self):
        """A brand-new user (no history) should be detected."""
        tracker = EraTracker()
        txn = {"TransactionAmt": 100.0, "card1": 11111, "TransactionDT": 1000000,
               "ProductCD": "W", "card4": "visa", "card6": "debit", "hour": 12}
        result = tracker.analyze(txn)
        assert result.window_txn_count == 0


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

class TestEraTrackerHistory:
    def test_history_accumulates(self):
        tracker = EraTracker()
        user_id = 55555
        for i in range(3):
            tracker.analyze({
                "TransactionAmt": 100.0 + i * 10, "card1": user_id,
                "TransactionDT": 500000 + i * 60, "ProductCD": "W",
                "card4": "visa", "card6": "debit", "hour": 12,
            })
        result = tracker.analyze({
            "TransactionAmt": 200.0, "card1": user_id,
            "TransactionDT": 500300, "ProductCD": "W",
            "card4": "visa", "card6": "debit", "hour": 12,
        })
        assert result.window_txn_count == 3

    def test_old_history_expires(self):
        tracker = EraTracker()
        user_id = 44444
        # Old transaction — beyond 24h window
        tracker.analyze({
            "TransactionAmt": 100.0, "card1": user_id,
            "TransactionDT": 100000, "ProductCD": "W",
            "card4": "visa", "card6": "debit", "hour": 12,
        })
        # New transaction — 24h + 1s later
        result = tracker.analyze({
            "TransactionAmt": 200.0, "card1": user_id,
            "TransactionDT": 100000 + 86401, "ProductCD": "W",
            "card4": "visa", "card6": "debit", "hour": 12,
        })
        # The old transaction should have expired
        assert result.window_txn_count == 0


# ---------------------------------------------------------------------------
# Heuristic fallback (no model)
# ---------------------------------------------------------------------------

class TestEraTrackerFallback:
    def test_heuristic_returns_valid_score(self):
        tracker = EraTracker.__new__(EraTracker)
        tracker.model = None
        tracker.model_loaded = False
        tracker.threshold = 0.5
        tracker.redis_client = None
        tracker.max_users_in_memory = 100
        tracker.window_seconds = 86400
        tracker.user_history = {}
        tracker._history_lock = __import__("threading").Lock()
        tracker._cat_indices = []
        tracker._all_features = []

        txn = {"TransactionAmt": 100.0, "card1": 33333, "TransactionDT": 100000,
               "ProductCD": "W", "card4": "visa", "card6": "debit", "hour": 12}
        result = tracker.analyze(txn)
        assert 0.0 <= result.fraud_score <= 1.0
