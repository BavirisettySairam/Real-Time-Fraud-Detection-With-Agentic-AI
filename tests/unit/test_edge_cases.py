"""Edge-case tests — missing fields, null values, boundary amounts, empty strings."""

import pytest

from services.agents.vibe_checker import VibeChecker, VibeCheckerResult
from services.agents.era_tracker import EraTracker, EraTrackerResult
from services.agents.og_check import OGCheck, OGCheckResult


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

    def test_vibe_empty_transaction(self, vibe):
        result = vibe.analyze({})
        assert isinstance(result, VibeCheckerResult)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_empty_transaction(self, era):
        result = era.analyze({})
        assert isinstance(result, EraTrackerResult)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_empty_transaction(self, og):
        result = og.analyze({})
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

    def test_vibe_null_values(self, vibe):
        result = vibe.analyze(self._TXN)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_null_values(self, era):
        result = era.analyze(self._TXN)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_null_values(self, og):
        result = og.analyze(self._TXN)
        assert 0.0 <= result.fraud_score <= 1.0


# ---- Boundary amounts ----------------------------------------------------

class TestBoundaryAmounts:
    """Amount = 0, negative, huge."""

    def test_vibe_amount_zero(self, vibe):
        result = vibe.analyze({"TransactionAmt": 0})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_amount_zero(self, era):
        result = era.analyze({"TransactionAmt": 0})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_amount_zero(self, og):
        result = og.analyze({"TransactionAmt": 0})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_vibe_negative_amount(self, vibe):
        result = vibe.analyze({"TransactionAmt": -100.0})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_negative_amount(self, era):
        result = era.analyze({"TransactionAmt": -100.0})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_negative_amount(self, og):
        result = og.analyze({"TransactionAmt": -100.0})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_vibe_huge_amount(self, vibe):
        result = vibe.analyze({"TransactionAmt": 999_999})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_huge_amount(self, era):
        result = era.analyze({"TransactionAmt": 999_999})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_huge_amount(self, og):
        result = og.analyze({"TransactionAmt": 999_999})
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

    def test_vibe_empty_strings(self, vibe):
        result = vibe.analyze(self._TXN)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_era_empty_strings(self, era):
        result = era.analyze(self._TXN)
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_empty_strings(self, og):
        result = og.analyze(self._TXN)
        assert 0.0 <= result.fraud_score <= 1.0


# ---- Micro amounts --------------------------------------------------------

class TestMicroAmount:
    """Tiny amount (below $1)."""

    def test_vibe_micro(self, vibe):
        result = vibe.analyze({"TransactionAmt": 0.01})
        assert 0.0 <= result.fraud_score <= 1.0

    def test_og_micro_fires_rule(self, og):
        result = og.analyze({"TransactionAmt": 0.50})
        violations = [v.rule_name for v in result.violations]
        assert "MICRO_AMOUNT" in violations
