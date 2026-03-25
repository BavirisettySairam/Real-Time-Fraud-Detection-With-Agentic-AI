import pytest

from dataclasses import dataclass

from services.orchestrator import agent_orchestrator as orchestrator_module
from services.orchestrator.agent_orchestrator import AgentOrchestrator, OrchestratorConfig


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
    explanation: str = "rules"


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


class _EraTracker:
    def analyze(self, _txn):
        return _DummyResult(0.2, "era")


class _OGCheck:
    def analyze(self, _txn):
        return _DummyOGResult(0.5, [_DummyRuleViolation("HIGH_AMOUNT")], "og")


class _TheYapper:
    class _Exp:
        natural_language_explanation = "combined explanation"

    def explain(self, **_kwargs):
        return self._Exp()


class _VibeChecker:
    def __init__(self, score: float = 0.9, loaded: bool = True):
        self._score = score
        self._loaded = loaded

    def analyze(self, _txn):
        return _DummyVibeResult(
            self._score,
            models_loaded={"lightgbm": self._loaded, "xgboost": self._loaded},
        )


def test_orchestrator_weighted_decision_and_output_shape():
    orchestrator = AgentOrchestrator(
        OrchestratorConfig(
            model_primary_weight=0.75,
            era_weight=0.4,
            og_weight=0.6,
            enable_parallel=False,
            use_langgraph=False,
        )
    )

    orchestrator.vibe_checker = _VibeChecker(score=0.9, loaded=True)
    orchestrator.era_tracker = _EraTracker()
    orchestrator.og_check = _OGCheck()
    orchestrator.the_yapper = _TheYapper()

    result = orchestrator.analyze({"TransactionAmt": 100})

    assert "final_score" in result
    assert "final_decision" in result
    assert "processing_time_ms" in result
    assert result["og_violations"] == ["HIGH_AMOUNT"]

    # agent_score = 0.4*0.2 + 0.6*0.5 = 0.38
    assert abs(result["agent_ensemble_score"] - 0.38) < 1e-6
    # Dynamic fusion: vibe_score=0.9 > 0.8 → trust vibe alone → final=0.9
    assert abs(result["final_score"] - 0.9) < 1e-6
    assert result["score_source"] == "vibe_high_confidence"
    assert result["final_decision"] == "BLOCK"

    orchestrator.close()


def test_orchestrator_parallel_path_handles_agent_errors():
    class _FailingEraTracker:
        def analyze(self, _txn):
            raise RuntimeError("era failure")

    orchestrator = AgentOrchestrator(
        OrchestratorConfig(enable_parallel=True, timeout_ms=1000, use_langgraph=False)
    )
    orchestrator.vibe_checker = _VibeChecker(score=0.6, loaded=True)
    orchestrator.era_tracker = _FailingEraTracker()
    orchestrator.og_check = _OGCheck()
    orchestrator.the_yapper = _TheYapper()

    result = orchestrator.analyze({"TransactionAmt": 100})

    # Fallback score is used when one agent fails.
    assert result["era_score"] == 0.5
    assert 0.0 <= result["final_score"] <= 1.0

    orchestrator.close()


def test_orchestrator_langgraph_path_when_available():
    if not orchestrator_module.LANGGRAPH_AVAILABLE:
        pytest.skip("LangGraph not installed in environment")

    orchestrator = AgentOrchestrator(
        OrchestratorConfig(
            use_langgraph=True,
            strict_langgraph=True,
            enable_parallel=True,
        )
    )

    orchestrator.vibe_checker = _VibeChecker(score=0.7, loaded=True)
    orchestrator.era_tracker = _EraTracker()
    orchestrator.og_check = _OGCheck()
    orchestrator.the_yapper = _TheYapper()
    orchestrator.workflow = orchestrator._build_workflow()

    result = orchestrator.analyze({"TransactionAmt": 100})

    assert orchestrator.workflow is not None
    assert result["final_decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert "explanation" in result

    orchestrator.close()


def test_orchestrator_strict_langgraph_raises_when_missing(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "LANGGRAPH_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="LangGraph requested but not installed"):
        AgentOrchestrator(
            OrchestratorConfig(
                use_langgraph=True,
                strict_langgraph=True,
            )
        )
