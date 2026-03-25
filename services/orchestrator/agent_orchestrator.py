# =============================================================================
# AGENT ORCHESTRATOR - LangGraph-based workflow orchestration
# =============================================================================

from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass
import logging
import time
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np

# LangGraph imports (with fallback)
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available, using fallback orchestration")

from services.agents import (
    VibeChecker,
    OGCheck,
    EraTracker,
    TheYapper,
)

logger = logging.getLogger(__name__)


class FraudState(TypedDict):
    """State for the fraud detection workflow"""
    transaction: Dict[str, Any]
    pipeline_features: Optional[Any]  # np.ndarray from FeaturePipeline
    vibe_score: Optional[float]
    vibe_explanation: Optional[str]
    vibe_models_loaded: Optional[Dict[str, bool]]
    era_score: Optional[float]
    era_explanation: Optional[str]
    og_score: Optional[float]
    og_violations: List[str]
    og_explanation: Optional[str]
    agent_ensemble_score: Optional[float]
    score_source: Optional[str]
    final_score: Optional[float]
    final_decision: Optional[str]
    explanation: Optional[str]
    explanation_details: Optional[Dict[str, Any]]
    processing_time_ms: Optional[float]


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""
    model_primary_weight: float = 0.75
    era_weight: float = 0.5
    og_weight: float = 0.5
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4
    enable_parallel: bool = True
    timeout_ms: float = 5000
    parallel_workers: int = 3
    use_langgraph: bool = True
    strict_langgraph: bool = True

    def __post_init__(self) -> None:
        if self.parallel_workers < 1:
            self.parallel_workers = 1
        if self.timeout_ms < 100:
            self.timeout_ms = 100

        self.model_primary_weight = min(max(self.model_primary_weight, 0.0), 1.0)

        # Keep thresholds sane even if misconfigured.
        self.high_risk_threshold = min(max(self.high_risk_threshold, 0.0), 1.0)
        self.medium_risk_threshold = min(max(self.medium_risk_threshold, 0.0), 1.0)
        if self.high_risk_threshold < self.medium_risk_threshold:
            logger.warning(
                "Invalid thresholds (high < medium). Swapping values: high=%s medium=%s",
                self.high_risk_threshold,
                self.medium_risk_threshold,
            )
            self.high_risk_threshold, self.medium_risk_threshold = (
                self.medium_risk_threshold,
                self.high_risk_threshold,
            )

        total = self.era_weight + self.og_weight
        if total <= 0:
            self.era_weight, self.og_weight = 0.5, 0.5
            return
        self.era_weight /= total
        self.og_weight /= total


class AgentOrchestrator:
    """
    LangGraph-based orchestrator for fraud detection agents
    
    Workflow:
    1. Vibe Checker: ML Ensemble (XGBoost + LightGBM)
    2. Era Tracker: 24-hour window behavioural analysis
    3. OG Check: Business rule checks
    4. Decision: Combines scores
    5. The Yapper: SHAP + LLM explanation
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=max(3, self.config.parallel_workers),
            thread_name_prefix="fraud-agent"
        )

        # Initialize agents
        self.vibe_checker = VibeChecker()
        self.era_tracker = EraTracker()
        self.og_check = OGCheck()
        self.the_yapper = TheYapper(vibe_checker=self.vibe_checker)
        
        if self.config.use_langgraph and not LANGGRAPH_AVAILABLE:
            message = "LangGraph requested but not installed. Install 'langgraph' dependency."
            if self.config.strict_langgraph:
                raise RuntimeError(message)
            logger.warning(message)

        self.workflow = self._build_workflow() if (self.config.use_langgraph and LANGGRAPH_AVAILABLE) else None
    
    def _build_workflow(self) -> Any:
        """Build the LangGraph workflow"""
        workflow = StateGraph(FraudState)
        
        # Add nodes
        workflow.add_node("vibe_check", self._vibe_check_node)
        workflow.add_node("era_track", self._era_track_node)
        workflow.add_node("og_check", self._og_check_node)
        workflow.add_node("decision", self._decision_node)
        workflow.add_node("yapper", self._yapper_node)
        
        if self.config.enable_parallel:
            # Fan-out analysis nodes so LangGraph can run them concurrently.
            workflow.add_edge(START, "vibe_check")
            workflow.add_edge(START, "era_track")
            workflow.add_edge(START, "og_check")
            workflow.add_edge("vibe_check", "decision")
            workflow.add_edge("era_track", "decision")
            workflow.add_edge("og_check", "decision")
        else:
            workflow.add_edge(START, "vibe_check")
            workflow.add_edge("vibe_check", "era_track")
            workflow.add_edge("era_track", "og_check")
            workflow.add_edge("og_check", "decision")

        workflow.add_edge("decision", "yapper")
        workflow.add_edge("yapper", END)
        
        return workflow.compile()

    def _vibe_check_node(self, state: FraudState) -> Dict[str, Any]:
        """Vibe Checker — ML Ensemble (XGBoost + LightGBM) inference."""
        try:
            features = state.get('pipeline_features')
            if features is not None:
                result = self._invoke_with_timeout(self.vibe_checker.analyze, features)
            else:
                result = self._invoke_with_timeout(
                    self.vibe_checker.analyze, np.zeros(self.vibe_checker.num_features, dtype=np.float32)
                )
            return {
                'vibe_score': result.fraud_score,
                'vibe_explanation': result.explanation,
                'vibe_models_loaded': result.models_loaded,
            }
        except TimeoutError:
            logger.error("Vibe Checker timed out")
            return {
                'vibe_score': 0.5,
                'vibe_explanation': 'Vibe Checker timed out',
                'vibe_models_loaded': {'lightgbm': False, 'xgboost': False},
            }
        except Exception as e:
            logger.error(f"Vibe Checker failed: {e}")
            return {
                'vibe_score': 0.5,
                'vibe_explanation': f'Vibe Checker unavailable: {e}',
                'vibe_models_loaded': {'lightgbm': False, 'xgboost': False},
            }
    
    def _era_track_node(self, state: FraudState) -> Dict[str, Any]:
        """Era Tracker — 24hr window behavioural analysis."""
        try:
            features = state.get('pipeline_features')
            result = self._invoke_with_timeout(
                lambda txn: self.era_tracker.analyze(txn, pipeline_features=features),
                state['transaction'],
            )
            return {
                'era_score': result.fraud_score,
                'era_explanation': result.explanation
            }
        except TimeoutError:
            logger.error("Era Tracker timed out")
            return {
                'era_score': 0.5,
                'era_explanation': "Era Tracker timed out"
            }
        except Exception as e:
            logger.error(f"Era Tracker failed: {e}")
            return {
                'era_score': 0.5,
                'era_explanation': f"Era Tracker unavailable: {e}"
            }
    
    def _og_check_node(self, state: FraudState) -> Dict[str, Any]:
        """OG Check — rule-based analysis."""
        try:
            features = state.get('pipeline_features')
            result = self._invoke_with_timeout(
                lambda txn: self.og_check.analyze(txn, pipeline_features=features),
                state['transaction'],
            )
            violations = [v.rule_name for v in result.violations]
            return {
                'og_score': result.fraud_score,
                'og_violations': violations,
                'og_explanation': result.explanation
            }
        except TimeoutError:
            logger.error("OG Check timed out")
            return {
                'og_score': 0.5,
                'og_violations': [],
                'og_explanation': "OG Check timed out"
            }
        except Exception as e:
            logger.error(f"OG Check failed: {e}")
            return {
                'og_score': 0.5,
                'og_violations': [],
                'og_explanation': f"OG Check unavailable: {e}"
            }
    
    def _decision_node(self, state: FraudState) -> Dict[str, Any]:
        """Combine Vibe Checker primary score with secondary agent ensemble.
        
        Uses dynamic weighting: when Vibe is very confident (>0.8), trust it
        alone; otherwise blend all three agents.
        """

        vibe_score = self._coerce_score(state.get('vibe_score', 0.5))
        era_score = self._coerce_score(state.get('era_score', 0.5))
        og_score = self._coerce_score(state.get('og_score', 0.5))

        models_loaded = state.get('vibe_models_loaded') or {}
        any_model_loaded = any(models_loaded.values())

        if any_model_loaded:
            # Dynamic fusion: high-confidence Vibe → trust it; else blend
            if vibe_score > 0.8:
                final_score = vibe_score
                score_source = "vibe_high_confidence"
            else:
                final_score = (
                    0.60 * vibe_score
                    + 0.25 * era_score
                    + 0.15 * og_score
                )
                score_source = "dynamic_blend"
        else:
            final_score = (
                self.config.era_weight * era_score
                + self.config.og_weight * og_score
            )
            score_source = "agents_fallback"

        final_score = self._coerce_score(final_score)
        
        agent_score = (
            self.config.era_weight * era_score +
            self.config.og_weight * og_score
        )

        # Determine decision
        if final_score >= self.config.high_risk_threshold:
            decision = "BLOCK"
        elif final_score >= self.config.medium_risk_threshold:
            decision = "REVIEW"
        else:
            decision = "APPROVE"
        
        return {
            'agent_ensemble_score': agent_score,
            'score_source': score_source,
            'final_score': final_score,
            'final_decision': decision
        }

    @staticmethod
    def _coerce_score(value: Any, default: float = 0.5) -> float:
        """Convert potentially-missing or non-finite scores into bounded floats."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default

        if not math.isfinite(numeric):
            return default

        return min(max(numeric, 0.0), 1.0)
    
    def _yapper_node(self, state: FraudState) -> Dict[str, Any]:
        """The Yapper — SHAP analysis + optional LLM explanation."""
        try:
            agent_scores = {
                'Vibe Checker': state.get('vibe_score', 0.5),
                'Era Tracker': state.get('era_score', 0.5),
                'OG Check': state.get('og_score', 0.5)
            }
            
            result = self.the_yapper.explain(
                transaction=state['transaction'],
                prediction=state.get('final_score', 0.5),
                agent_scores=agent_scores,
                violations=state.get('og_violations', []),
                pipeline_features=state.get('pipeline_features'),
            )
            
            details = result.to_dict()
            
            return {
                'explanation': result.natural_language_explanation,
                'explanation_details': details,
            }
        except Exception as e:
            logger.error(f"The Yapper failed: {e}")
            return {
                'explanation': f"Score: {state.get('final_score', 0):.1%}",
                'explanation_details': None,
            }
    
    def analyze(self, transaction: Dict[str, Any], pipeline_features: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run complete fraud analysis on a transaction.
        
        Args:
            transaction: Transaction data dictionary (with derived time features).
            pipeline_features: Pre-computed pipeline features (np.ndarray from
                               FeaturePipeline.transform).  When called from the
                               API gateway this is always supplied; standalone
                               callers may pass None and agents will fall back
                               to heuristic scoring.
        
        Returns:
            Complete analysis result with scores and explanation.
        """
        start_time = time.time()

        # Initial state
        initial_state: FraudState = {
            'transaction': transaction,
            'pipeline_features': pipeline_features,
            'vibe_score': None,
            'vibe_explanation': None,
            'vibe_models_loaded': None,
            'era_score': None,
            'era_explanation': None,
            'og_score': None,
            'og_violations': [],
            'og_explanation': None,
            'agent_ensemble_score': None,
            'score_source': None,
            'final_score': None,
            'final_decision': None,
            'explanation': None,
            'explanation_details': None,
            'processing_time_ms': None
        }
        
        if self.workflow:
            # Run LangGraph workflow
            result = self.workflow.invoke(initial_state)
        else:
            # Fast path without LangGraph runtime overhead
            result = self._fast_analyze(initial_state)
        
        # Add processing time
        result['processing_time_ms'] = (time.time() - start_time) * 1000
        
        return result
    
    def _fast_analyze(self, state: FraudState) -> Dict[str, Any]:
        """Low-overhead analysis path with optional parallel agent execution."""
        if self.config.enable_parallel:
            state.update(self._run_parallel_agents(state))
        else:
            state.update(self._vibe_check_node(state))
            state.update(self._era_track_node(state))
            state.update(self._og_check_node(state))

        state.update(self._decision_node(state))
        state.update(self._yapper_node(state))

        return dict(state)

    def _run_parallel_agents(self, state: FraudState) -> Dict[str, Any]:
        """Run Vibe Checker + Era Tracker + OG Check concurrently."""
        txn = state['transaction']
        features = state.get('pipeline_features')
        timeout_seconds = self.config.timeout_ms / 1000.0
        futures = {
            'vibe': self._executor.submit(self.vibe_checker.analyze,
                                          features if features is not None else np.zeros(self.vibe_checker.num_features, dtype=np.float32)),
            'era': self._executor.submit(self.era_tracker.analyze, txn, features),
            'og': self._executor.submit(self.og_check.analyze, txn, features),
        }

        results: Dict[str, Any] = {
            'vibe_score': 0.5,
            'vibe_explanation': 'Vibe Checker unavailable',
            'vibe_models_loaded': {'lightgbm': False, 'xgboost': False},
            'era_score': 0.5,
            'era_explanation': 'Era Tracker unavailable',
            'og_score': 0.5,
            'og_violations': [],
            'og_explanation': 'OG Check unavailable',
        }

        try:
            vibe_result = futures['vibe'].result(timeout=timeout_seconds)
            results['vibe_score'] = vibe_result.fraud_score
            results['vibe_explanation'] = vibe_result.explanation
            results['vibe_models_loaded'] = vibe_result.models_loaded
        except TimeoutError:
            logger.error("Vibe Checker timed out")
            futures['vibe'].cancel()
        except Exception as exc:
            logger.error("Vibe Checker failed: %s", exc)

        try:
            era_result = futures['era'].result(timeout=timeout_seconds)
            results['era_score'] = era_result.fraud_score
            results['era_explanation'] = era_result.explanation
        except TimeoutError:
            logger.error("Era Tracker timed out")
            futures['era'].cancel()
        except Exception as exc:
            logger.error("Era Tracker failed: %s", exc)

        try:
            og_result = futures['og'].result(timeout=timeout_seconds)
            results['og_score'] = og_result.fraud_score
            results['og_violations'] = [v.rule_name for v in og_result.violations]
            results['og_explanation'] = og_result.explanation
        except TimeoutError:
            logger.error("OG Check timed out")
            futures['og'].cancel()
        except Exception as exc:
            logger.error("OG Check failed: %s", exc)

        return results

    def _invoke_with_timeout(self, fn: Any, *args: Any) -> Any:
        """Invoke an agent function with timeout enforcement."""
        future = self._executor.submit(fn, *args)
        return future.result(timeout=self.config.timeout_ms / 1000.0)

    def close(self) -> None:
        """Release orchestrator runtime resources."""
        self._executor.shutdown(wait=True, cancel_futures=True)
    
    def get_workflow_visualization(self) -> str:
        """Get Mermaid diagram of the workflow"""
        if self.workflow is not None and hasattr(self.workflow, "get_graph"):
            try:
                graph = self.workflow.get_graph()
                if hasattr(graph, "draw_mermaid"):
                    mermaid_text = graph.draw_mermaid()
                    if isinstance(mermaid_text, str) and mermaid_text.strip():
                        return mermaid_text
            except Exception as exc:
                logger.warning("Failed to derive Mermaid from LangGraph runtime: %s", exc)

        if self.config.enable_parallel:
            return """
```mermaid
graph TD
    A[Transaction Input] --> B[Vibe Checker - XGB+LGB]
    A --> D[Era Tracker - 24hr]
    A --> E[OG Check - Rules]
    B --> F[Decision Aggregator]
    D --> F
    E --> F
    F --> G[The Yapper - SHAP+LLM]
    G --> H[Final Output]

    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style F fill:#fff3e0
    style B fill:#d1fae5
```
            """

        return """
```mermaid
graph TD
    A[Transaction Input] --> B[Vibe Checker - XGB+LGB]
    B --> D[Era Tracker - 24hr]
    D --> E[OG Check - Rules]
    E --> F[Decision Aggregator]
    F --> G[The Yapper - SHAP+LLM]
    G --> H[Final Output]
    
    B -->|Primary Fraud Probability| F
    D -->|Behavioural Risk| F
    E -->|Rule Violations| F
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style F fill:#fff3e0
    style B fill:#d1fae5
```
        """


# Convenience function for quick analysis
def analyze_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Quick analysis of a single transaction"""
    orchestrator = AgentOrchestrator()
    try:
        return orchestrator.analyze(transaction)
    finally:
        orchestrator.close()


if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = AgentOrchestrator()
    
    test_transaction = {
        'TransactionID': 12345,
        'card1': 1000,
        'card2': 500,
        'TransactionAmt': 2500.0,
        'ProductCD': 'W',
        'hour': 3,
        'day': 5,
        'is_weekend': 1,
        'is_night': 1,
        'card1_txn_count': 2,
        'txn_count_1h': 8,
        'P_emaildomain': 'gmail.com',
        'DeviceInfo': 'Windows 10'
    }
    
    print("Running fraud analysis...")
    print("=" * 60)
    
    result = orchestrator.analyze(test_transaction)
    
    print(f"\n📊 ANALYSIS RESULTS")
    print(f"=" * 60)
    print(f"Vibe Checker Score: {result.get('vibe_score', 0):.1%}")
    print(f"Era Tracker Score: {result.get('era_score', 0):.1%}")
    print(f"OG Check Score: {result.get('og_score', 0):.1%}")
    print(f"\n🎯 FINAL DECISION: {result.get('final_decision')}")
    print(f"Final Score: {result.get('final_score', 0):.1%}")
    print(f"Processing Time: {result.get('processing_time_ms', 0):.2f}ms")
    print(f"\n📝 EXPLANATION:")
    print(result.get('explanation', 'No explanation available'))
    
    if result.get('og_violations'):
        print(f"\n⚠️ RULE VIOLATIONS: {', '.join(result['og_violations'])}")
