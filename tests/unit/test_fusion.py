"""Agent disagreement / fusion tests.

Mocks the four agents to return conflicting scores and verifies the
orchestrator's decision-node fusion logic:

* vibe > 0.8  →  final = vibe  (vibe_high_confidence)
* else        →  final = 0.60*vibe + 0.25*era + 0.15*og  (dynamic_blend)
* no model    →  era_weight*era + og_weight*og  (agents_fallback)
"""

import pytest
from dataclasses import dataclass

from services.orchestrator.agent_orchestrator import (
    AgentOrchestrator,
    OrchestratorConfig,
)


# ---------------------------------------------------------------------------
# Lightweight fakes
# ---------------------------------------------------------------------------

@dataclass
class _DummyVibeResult:
    fraud_score: float
    lightgbm_score: float = 0.5
    xgboost_score: float = 0.5
    models_loaded: dict = None
    explanation: str = "vibe"

    def __post_init__(self):
        if self.models_loaded is None:
            self.models_loaded = {"lightgbm": True, "xgboost": True}


@dataclass
class _DummyResult:
    fraud_score: float
    explanation: str = "ok"


@dataclass
class _DummyRuleViolation:
    rule_name: str


@dataclass
class _DummyOGResult:
    fraud_score: float
    violations: list
    explanation: str = "og"


class _FakeYapper:
    class _Exp:
        natural_language_explanation = "yap"
        def to_dict(self):
            return {"natural_language_explanation": "yap"}
    def explain(self, **_kw):
        return self._Exp()


def _build_orchestrator():
    return AgentOrchestrator(
        OrchestratorConfig(enable_parallel=False, use_langgraph=False)
    )


def _make_vibe(score, loaded=True):
    class V:
        num_features = 175
        feature_columns = [f"f_{i}" for i in range(175)]
        lgb_model = None

        def analyze(self, _features):
            return _DummyVibeResult(
                score, models_loaded={"lightgbm": loaded, "xgboost": loaded}
            )
    return V()


def _make_era(score):
    class E:
        def analyze(self, _txn, pipeline_features=None):
            return _DummyResult(score, "era")
    return E()


def _make_og(score, violations=None):
    class O:
        def analyze(self, _txn, pipeline_features=None):
            viols = [_DummyRuleViolation(v) for v in (violations or [])]
            return _DummyOGResult(score, viols, "og")
    return O()


TXN = {"TransactionAmt": 100}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHighConfidenceVibe:
    """When vibe > 0.8, the final score equals the vibe score alone."""

    def test_vibe_95_overrides_low_agents(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.95)
        orch.era_tracker = _make_era(0.1)           # very low
        orch.og_check = _make_og(0.1)               # very low
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        assert abs(result["final_score"] - 0.95) < 1e-6
        assert result["score_source"] == "vibe_high_confidence"
        orch.close()

    def test_vibe_81_still_overrides(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.81)
        orch.era_tracker = _make_era(0.0)
        orch.og_check = _make_og(0.0)
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        assert abs(result["final_score"] - 0.81) < 1e-6
        assert result["score_source"] == "vibe_high_confidence"
        orch.close()


class TestDynamicBlend:
    """When vibe ≤ 0.8, the formula 0.60*vibe + 0.25*era + 0.15*og applies."""

    def test_blend_math(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.6)
        orch.era_tracker = _make_era(0.8)
        orch.og_check = _make_og(0.4)
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        expected = 0.60 * 0.6 + 0.25 * 0.8 + 0.15 * 0.4  # 0.62
        assert abs(result["final_score"] - expected) < 1e-6
        assert result["score_source"] == "dynamic_blend"
        orch.close()

    def test_vibe_low_era_high_og_high(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.2)
        orch.era_tracker = _make_era(0.9)
        orch.og_check = _make_og(0.9)
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        expected = 0.60 * 0.2 + 0.25 * 0.9 + 0.15 * 0.9  # 0.48
        assert abs(result["final_score"] - expected) < 1e-6
        assert result["final_decision"] == "REVIEW"
        orch.close()

    def test_vibe_exactly_80_uses_blend(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.80)
        orch.era_tracker = _make_era(0.5)
        orch.og_check = _make_og(0.5)
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        expected = 0.60 * 0.8 + 0.25 * 0.5 + 0.15 * 0.5  # 0.68
        assert abs(result["final_score"] - expected) < 1e-6
        assert result["score_source"] == "dynamic_blend"
        orch.close()


class TestAgentsFallback:
    """When no ML model is loaded, only era + og scores are used."""

    def test_fallback_mode(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.5, loaded=False)
        orch.era_tracker = _make_era(0.8)
        orch.og_check = _make_og(0.6)
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        assert result["score_source"] == "agents_fallback"
        # Default OrchestratorConfig: era_weight=0.5, og_weight=0.5 (normalised)
        expected = 0.5 * 0.8 + 0.5 * 0.6  # 0.70
        assert abs(result["final_score"] - expected) < 1e-6
        orch.close()


class TestDecisionThresholds:
    """Verify APPROVE / REVIEW / BLOCK thresholds (0.4, 0.7 by default)."""

    def test_approve(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.1)
        orch.era_tracker = _make_era(0.1)
        orch.og_check = _make_og(0.1)
        orch.the_yapper = _FakeYapper()
        result = orch.analyze(TXN)
        assert result["final_decision"] == "APPROVE"
        orch.close()

    def test_review(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.6)
        orch.era_tracker = _make_era(0.6)
        orch.og_check = _make_og(0.6)
        orch.the_yapper = _FakeYapper()
        result = orch.analyze(TXN)
        # 0.60*0.6 + 0.25*0.6 + 0.15*0.6 = 0.60 → REVIEW
        assert result["final_decision"] == "REVIEW"
        orch.close()

    def test_block(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.95)
        orch.era_tracker = _make_era(0.9)
        orch.og_check = _make_og(0.9, violations=["HIGH_AMOUNT"])
        orch.the_yapper = _FakeYapper()
        result = orch.analyze(TXN)
        assert result["final_decision"] == "BLOCK"
        orch.close()


class TestConflictingScores:
    """Agents wildly disagree — verify fusion still produces a sane number."""

    def test_vibe_high_others_zero(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.99)
        orch.era_tracker = _make_era(0.0)
        orch.og_check = _make_og(0.0)
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        assert 0.0 <= result["final_score"] <= 1.0
        # vibe > 0.8 → trust vibe
        assert abs(result["final_score"] - 0.99) < 1e-6
        orch.close()

    def test_vibe_zero_others_high(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.05)
        orch.era_tracker = _make_era(0.95)
        orch.og_check = _make_og(0.95)
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        # blend: 0.60*0.05 + 0.25*0.95 + 0.15*0.95 = 0.41
        expected = 0.60 * 0.05 + 0.25 * 0.95 + 0.15 * 0.95
        assert abs(result["final_score"] - expected) < 1e-6
        assert result["final_decision"] == "REVIEW"
        orch.close()

    def test_alternating_high_low(self):
        orch = _build_orchestrator()
        orch.vibe_checker = _make_vibe(0.3)
        orch.era_tracker = _make_era(0.9)
        orch.og_check = _make_og(0.1)
        orch.the_yapper = _FakeYapper()

        result = orch.analyze(TXN)
        expected = 0.60 * 0.3 + 0.25 * 0.9 + 0.15 * 0.1  # 0.42
        assert abs(result["final_score"] - expected) < 1e-6
        orch.close()
