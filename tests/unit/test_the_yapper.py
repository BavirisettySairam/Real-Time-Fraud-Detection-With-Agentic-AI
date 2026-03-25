"""Unit tests for The Yapper — SHAP explainability and LLM fallback."""

import pytest

from services.agents.vibe_checker import VibeChecker
from services.agents.the_yapper import TheYapper, YapperResult, FeatureContribution


@pytest.fixture(scope="module")
def vibe_checker() -> VibeChecker:
    return VibeChecker()


@pytest.fixture(scope="module")
def yapper_no_llm(vibe_checker: VibeChecker) -> TheYapper:
    """Yapper initialised without an LLM key — forces template fallback."""
    return TheYapper(vibe_checker=vibe_checker, llm_api_key=None, use_llm=False)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

class TestYapperSHAP:
    def test_shap_explainer_initialised(self, yapper_no_llm: TheYapper):
        assert yapper_no_llm._shap_explainer is not None

    def test_shap_values_returned(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 500.0, "ProductCD": "W"},
            prediction=0.6,
            agent_scores={"vibe_checker": 0.6, "era_tracker": 0.4, "og_check": 0.3},
            violations=["HIGH_AMOUNT"],
        )
        assert isinstance(result, YapperResult)
        assert result.shap_available is True
        assert len(result.top_features) > 0

    def test_feature_contributions_structure(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 200.0},
            prediction=0.3,
            agent_scores={"vibe_checker": 0.3},
            violations=[],
        )
        for fc in result.top_features:
            assert isinstance(fc, FeatureContribution)
            assert isinstance(fc.feature_name, str)
            assert isinstance(fc.raw_name, str)
            assert isinstance(fc.shap_value, float)
            assert fc.direction in ("increases_risk", "decreases_risk")

    def test_shap_values_are_nonzero(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 500.0, "ProductCD": "W"},
            prediction=0.5,
            agent_scores={"vibe_checker": 0.5},
            violations=[],
        )
        nonzero = [f for f in result.top_features if abs(f.shap_value) > 1e-6]
        assert len(nonzero) > 0, "At least some SHAP values should be non-zero"


# ---------------------------------------------------------------------------
# Template fallback (no LLM)
# ---------------------------------------------------------------------------

class TestYapperTemplateFallback:
    def test_llm_not_used_without_key(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 300.0},
            prediction=0.7,
            agent_scores={"vibe_checker": 0.7, "era_tracker": 0.5, "og_check": 0.6},
            violations=["HIGH_AMOUNT", "LATE_NIGHT"],
        )
        assert result.llm_used is False

    def test_explanation_present_without_llm(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 100.0},
            prediction=0.2,
            agent_scores={"vibe_checker": 0.2},
            violations=[],
        )
        assert isinstance(result.natural_language_explanation, str)
        assert len(result.natural_language_explanation) > 20

    def test_summary_present(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 100.0},
            prediction=0.5,
            agent_scores={"vibe_checker": 0.5},
            violations=["LATE_NIGHT"],
        )
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_recommended_action_present(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 100.0},
            prediction=0.85,
            agent_scores={"vibe_checker": 0.85},
            violations=["HIGH_AMOUNT"],
        )
        assert isinstance(result.recommended_action, str)
        assert len(result.recommended_action) > 0

    def test_confidence_factors_list(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 100.0},
            prediction=0.5,
            agent_scores={"vibe_checker": 0.5, "era_tracker": 0.5, "og_check": 0.5},
            violations=[],
        )
        assert isinstance(result.confidence_factors, list)

    def test_to_dict_serialisable(self, yapper_no_llm: TheYapper):
        result = yapper_no_llm.explain(
            transaction={"TransactionAmt": 100.0},
            prediction=0.5,
            agent_scores={"vibe_checker": 0.5},
            violations=[],
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "top_features" in d
        import json
        json.dumps(d)  # must be JSON-serialisable


# ---------------------------------------------------------------------------
# No vibe checker at all
# ---------------------------------------------------------------------------

class TestYapperNoVibeChecker:
    def test_no_shap_without_vibe_checker(self):
        yapper = TheYapper(vibe_checker=None, use_llm=False)
        result = yapper.explain(
            transaction={"TransactionAmt": 100.0},
            prediction=0.5,
            agent_scores={"vibe_checker": 0.5},
            violations=[],
        )
        # Should still produce a result, just with heuristic contributions
        assert isinstance(result, YapperResult)
        assert isinstance(result.natural_language_explanation, str)
