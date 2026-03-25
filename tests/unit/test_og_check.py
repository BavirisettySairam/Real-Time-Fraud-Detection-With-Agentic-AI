"""Unit tests for the OG Check agent — model loading, rule firing, and scoring."""

import pytest

from services.agents.og_check import OGCheck, OGCheckResult, RuleViolation
from tests.unit.conftest import preprocess


@pytest.fixture(scope="module")
def og_check() -> OGCheck:
    """Load the real OG Check once for the module."""
    return OGCheck()


# ---------------------------------------------------------------------------
# Model / params loading
# ---------------------------------------------------------------------------

class TestOGCheckLoading:
    def test_params_loaded(self, og_check: OGCheck):
        assert og_check.model_loaded is True

    def test_lgb_model_loaded(self, og_check: OGCheck):
        assert og_check._use_lgb is True
        assert og_check._lgb_model is not None

    def test_threshold_in_range(self, og_check: OGCheck):
        assert 0.0 <= og_check._og_threshold <= 1.0


# ---------------------------------------------------------------------------
# Score output contract
# ---------------------------------------------------------------------------

class TestOGCheckScoring:
    def test_basic_score_in_range(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 100.0, "hour": 14}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        assert isinstance(result, OGCheckResult)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_result_fields(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 100.0, "hour": 14}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        assert isinstance(result.violations, list)
        assert isinstance(result.passed_rules, int)
        assert isinstance(result.failed_rules, int)
        assert isinstance(result.explanation, str)
        assert result.passed_rules + result.failed_rules > 0


# ---------------------------------------------------------------------------
# Individual rule firing
# ---------------------------------------------------------------------------

class TestOGCheckRules:
    def test_high_amount_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 5000.0, "hour": 14}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "HIGH_AMOUNT" in names

    def test_max_amount_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 15000.0, "hour": 14}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "MAX_AMOUNT" in names
        assert "HIGH_AMOUNT" in names  # should also fire

    def test_late_night_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 100.0, "hour": 3}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "LATE_NIGHT" in names

    def test_round_amount_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 1000.0, "hour": 14}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "ROUND_AMOUNT" in names

    def test_micro_amount_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 0.50, "hour": 14}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "MICRO_AMOUNT" in names

    def test_velocity_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 100.0, "hour": 14, "txn_count_1h": 15}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "VELOCITY_TXN_1H" in names

    def test_email_mismatch_fires(self, og_check: OGCheck, pipeline):
        txn = {
            "TransactionAmt": 100.0, "hour": 14,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "yahoo.com",
        }
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "EMAIL_MISMATCH" in names

    def test_disposable_email_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 100.0, "hour": 14, "P_emaildomain": "mailinator.com"}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "DISPOSABLE_EMAIL" in names

    def test_no_device_high_value_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 800.0, "hour": 14, "DeviceInfo": ""}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "NO_DEVICE_HIGH_VALUE" in names

    def test_bot_device_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 100.0, "hour": 14, "DeviceInfo": "headless-chrome v99"}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "BOT_DEVICE" in names

    def test_clean_transaction_no_violations(self, og_check: OGCheck, pipeline):
        txn = {
            "TransactionAmt": 50.0, "hour": 14, "is_weekend": 0,
            "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
            "DeviceInfo": "iPhone 14", "card4": "visa",
        }
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        assert result.failed_rules == 0

    def test_weekend_high_value_fires(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 2000.0, "hour": 14, "is_weekend": 1}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        names = [v.rule_name for v in result.violations]
        assert "WEEKEND_HIGH_VALUE" in names


# ---------------------------------------------------------------------------
# RuleViolation dataclass
# ---------------------------------------------------------------------------

class TestRuleViolation:
    def test_severity_values(self, og_check: OGCheck, pipeline):
        txn = {"TransactionAmt": 15000.0, "hour": 3}
        features = preprocess(pipeline, txn)
        result = og_check.analyze(txn, pipeline_features=features)
        for v in result.violations:
            assert v.severity in ("high", "medium", "low")
            assert v.risk_contribution > 0.0
            assert len(v.description) > 0


# ---------------------------------------------------------------------------
# Fallback (no trained model/params)
# ---------------------------------------------------------------------------

class TestOGCheckFallback:
    def test_fallback_sum_scoring(self):
        """When no LGB model, score comes from sum of risk contributions."""
        og = OGCheck(params_path="nonexistent_path_12345.json")
        assert og._use_lgb is False
        result = og.analyze({"TransactionAmt": 15000.0, "hour": 3})
        assert result.fraud_score > 0.0  # violations should contribute
        assert 0.0 <= result.fraud_score <= 1.0
