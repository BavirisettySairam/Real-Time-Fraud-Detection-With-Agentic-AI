"""Edge-case tests — missing fields, null values, boundary amounts, empty strings."""

import pytest

from services.agents.vibe_checker import VibeChecker, VibeCheckerResult
from services.agents.era_tracker import EraTracker, EraTrackerResult
from services.agents.og_check import OGCheck, OGCheckResult
from tests.unit.conftest import preprocess


@pytest.fixture(scope="module")
def vibe():
    return VibeChecker()


@pytest.fixture(scope="module")
def era():
    return EraTracker()


@pytest.fixture(scope="module")
def og():
    return OGCheck()


# ---- Minimal / empty transactions ----------------------------------------

class TestMissingFields:
    """All agents must still return a valid result when key fields are absent."""

    def test_vibe_empty_transaction(self, vibe, pipeline):
        features = preprocess(pipeline, {})
        result = vibe.analyze(features)
        assert isinstance(result, VibeCheckerResult)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_empty_transaction(self, era, pipeline):
        txn = {}
        features = preprocess(pipeline, txn)
        result = era.analyze(txn, pipeline_features=features)
        assert isinstance(result, EraTrackerResult)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_empty_transaction(self, og, pipeline):
        txn = {}
        features = preprocess(pipeline, txn)
        result = og.analyze(txn, pipeline_features=features)
        assert isinstance(result, OGCheckResult)
        assert 0.0 <= result.fraud_score <= 1.0


class TestNullValues:
    """Fields present but set to None."""

    _TXN = {
        "TransactionAmt": None,
        "ProductCD": None,
        "card1": None,
        "card2": None,
        "addr1": None,
        "P_emaildomain": None,
        "R_emaildomain": None,
        "DeviceType": None,
        "DeviceInfo": None,
        "TransactionDT": None,
    }

    def test_vibe_null_values(self, vibe, pipeline):
        features = preprocess(pipeline, self._TXN)
        result = vibe.analyze(features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_null_values(self, era, pipeline):
        features = preprocess(pipeline, self._TXN)
        result = era.analyze(self._TXN, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_null_values(self, og, pipeline):
        features = preprocess(pipeline, self._TXN)
        result = og.analyze(self._TXN, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0


# ---- Boundary amounts ----------------------------------------------------

class TestBoundaryAmounts:
    """Amount = 0, negative, huge."""

    def test_vibe_amount_zero(self, vibe, pipeline):
        features = preprocess(pipeline, {"TransactionAmt": 0})
        result = vibe.analyze(features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_amount_zero(self, era, pipeline):
        txn = {"TransactionAmt": 0}
        features = preprocess(pipeline, txn)
        result = era.analyze(txn, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_amount_zero(self, og, pipeline):
        txn = {"TransactionAmt": 0}
        features = preprocess(pipeline, txn)
        result = og.analyze(txn, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_vibe_negative_amount(self, vibe, pipeline):
        features = preprocess(pipeline, {"TransactionAmt": -100.0})
        result = vibe.analyze(features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_negative_amount(self, era, pipeline):
        txn = {"TransactionAmt": -100.0}
        features = preprocess(pipeline, txn)
        result = era.analyze(txn, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_negative_amount(self, og, pipeline):
        txn = {"TransactionAmt": -100.0}
        features = preprocess(pipeline, txn)
        result = og.analyze(txn, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_vibe_huge_amount(self, vibe, pipeline):
        features = preprocess(pipeline, {"TransactionAmt": 999_999})
        result = vibe.analyze(features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_huge_amount(self, era, pipeline):
        txn = {"TransactionAmt": 999_999}
        features = preprocess(pipeline, txn)
        result = era.analyze(txn, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_huge_amount(self, og, pipeline):
        txn = {"TransactionAmt": 999_999}
        features = preprocess(pipeline, txn)
        result = og.analyze(txn, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0
        # OG Check should flag a huge amount
        violations = [v.rule_name for v in result.violations]
        assert "MAX_AMOUNT" in violations or "HIGH_AMOUNT" in violations


# ---- Empty string fields --------------------------------------------------

class TestEmptyStringFields:
    """All string-type fields set to empty string."""

    _TXN = {
        "TransactionAmt": 100.0,
        "ProductCD": "",
        "card1": "",
        "card2": "",
        "card4": "",
        "card6": "",
        "addr1": "",
        "P_emaildomain": "",
        "R_emaildomain": "",
        "DeviceType": "",
        "DeviceInfo": "",
    }

    def test_vibe_empty_strings(self, vibe, pipeline):
        features = preprocess(pipeline, self._TXN)
        result = vibe.analyze(features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_empty_strings(self, era, pipeline):
        features = preprocess(pipeline, self._TXN)
        result = era.analyze(self._TXN, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_empty_strings(self, og, pipeline):
        features = preprocess(pipeline, self._TXN)
        result = og.analyze(self._TXN, pipeline_features=features)
        assert 0.0 <= result.fraud_score <= 1.0


# ---- Micro amounts --------------------------------------------------------

class TestMicroAmount:
    """Tiny amount (below $1)."""

    def test_vibe_micro(self, vibe, pipeline):
        features = preprocess(pipeline, {"TransactionAmt": 0.01})
        result = vibe.analyze(features)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_micro_fires_rule(self, og, pipeline):
        txn = {"TransactionAmt": 0.50}
        features = preprocess(pipeline, txn)
        result = og.analyze(txn, pipeline_features=features)
        violations = [v.rule_name for v in result.violations]
        assert "MICRO_AMOUNT" in violations
