# =============================================================================
# THE YAPPER - SHAP + LLM explainability agent
# =============================================================================
"""
Pipeline:
  1. Receive a transaction + the Vibe Checker's fraud score (and secondary agents).
  2. Run SHAP TreeExplainer on the Vibe Checker LightGBM model → per-feature
     attributions.
  3. Build a structured explanation context (top drivers, transaction context,
     agent consensus, rule violations).
  4. Feed the context to an LLM (OpenRouter API) with a system prompt that
     asks for a simple, non-technical paragraph a human analyst can act on.
  5. Return the SHAP breakdown + the LLM plain-English explanation together.
"""

import os
import logging
import numpy as np
import requests
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ── Human-friendly feature name mapping ──────────────────────────────────────
FEATURE_LABELS: Dict[str, str] = {
    # Transaction basics
    "TransactionAmt": "Transaction Amount ($)",
    "TransactionAmt_log": "Transaction Amount (log-scaled)",
    "TransactionAmt_sqrt": "Transaction Amount (sqrt-scaled)",
    "TransactionAmt_cents": "Cents portion of amount",
    "ProductCD": "Product Category",
    # Time
    "TransactionDT": "Transaction Timestamp",
    "hour": "Hour of Day",
    "day": "Day of Week",
    "is_weekend": "Weekend Transaction",
    "is_night": "Night-time Transaction (10pm–6am)",
    # Card & address
    "card1": "Card Identifier",
    "card2": "Card Field 2",
    "card3": "Card Field 3",
    "card4": "Card Network (Visa/MC/…)",
    "card5": "Card Field 5",
    "card6": "Card Type (debit/credit)",
    "addr1": "Billing Address Region",
    "addr2": "Billing Country Code",
    "card_addr_combo": "Card + Address Combination",
    # Email
    "P_emaildomain": "Buyer Email Domain",
    "R_emaildomain": "Recipient Email Domain",
    "email_domain_pair": "Email Domain Pair",
    # Device
    "DeviceType": "Device Type",
    "DeviceInfo": "Device Information",
    # Counts & deltas
    "C1": "Payment Count (C1)",
    "C2": "Payment Count (C2)",
    "C4": "Payment Count (C4)",
    "C5": "Payment Count (C5)",
    "C6": "Payment Count (C6)",
    "C13": "Payment Count (C13)",
    "C14": "Payment Count (C14)",
    "D1": "Days Since Last Transaction",
    "D2": "Days Since Last Transaction (D2)",
    "D3": "Days Since Last Transaction (D3)",
    "D4": "Days Since Last Transaction (D4)",
    "D15": "Days Since Last Transaction (D15)",
    # Vesta
    "V12": "Vesta Risk Feature V12",
    "V13": "Vesta Risk Feature V13",
    "V14": "Vesta Risk Feature V14",
    "V45": "Vesta Risk Feature V45",
    "V54": "Vesta Risk Feature V54",
    "V75": "Vesta Risk Feature V75",
    "V78": "Vesta Risk Feature V78",
    "V83": "Vesta Risk Feature V83",
    "V91": "Vesta Risk Feature V91",
    "V126": "Vesta Risk Feature V126",
    "V127": "Vesta Risk Feature V127",
    "V128": "Vesta Risk Feature V128",
    "V131": "Vesta Risk Feature V131",
    "V258": "Vesta Risk Feature V258",
    "V279": "Vesta Risk Feature V279",
    "V280": "Vesta Risk Feature V280",
    "V283": "Vesta Risk Feature V283",
    "V285": "Vesta Risk Feature V285",
    "V306": "Vesta Risk Feature V306",
    "V307": "Vesta Risk Feature V307",
    "V310": "Vesta Risk Feature V310",
    "V312": "Vesta Risk Feature V312",
    "V313": "Vesta Risk Feature V313",
    "V314": "Vesta Risk Feature V314",
    "V315": "Vesta Risk Feature V315",
    "V317": "Vesta Risk Feature V317",
    # Match flags
    "M1": "Match Flag M1",
    "M2": "Match Flag M2",
    "M3": "Match Flag M3",
    "M4": "Match Flag M4",
    "M5": "Match Flag M5",
    "M6": "Match Flag M6",
    "M9": "Match Flag M9",
    # Engineered
    "missing_count": "Number of Missing Fields",
    "card1_freq": "Card Usage Frequency",
    "card1_txn_count": "Transaction Count per Card",
    "txn_count_1h": "Transactions in Last Hour",
    "amount_1h": "Total Amount in Last Hour",
}


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class FeatureContribution:
    """A single feature's contribution to the prediction."""
    feature_name: str
    raw_name: str
    feature_value: Any
    shap_value: float
    direction: str  # 'increases_risk' or 'decreases_risk'


@dataclass
class YapperResult:
    """Complete explanation of a fraud decision."""
    summary: str
    natural_language_explanation: str
    top_risk_factors: List[str]
    top_features: List[FeatureContribution]
    shap_available: bool
    llm_used: bool
    confidence_factors: List[str]
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["top_features"] = [asdict(f) for f in self.top_features]
        return d


# ── Main class ───────────────────────────────────────────────────────────────
class TheYapper:
    """
    The Yapper — SHAP + LLM explanation agent.

    Pipeline: SHAP TreeExplainer on LightGBM → structured context → LLM
    (OpenRouter) → plain-English explanation for non-technical analysts.
    Falls back to a template explanation when the LLM is unavailable.
    """

    def __init__(
        self,
        vibe_checker: Optional[Any] = None,
        llm_api_key: Optional[str] = None,
        use_llm: Optional[bool] = None,
        llm_model: Optional[str] = None,
        request_timeout_s: float = 15.0,
    ):
        self.vibe_checker = vibe_checker
        self.llm_api_key = llm_api_key or os.getenv("OPENROUTER_API_KEY")
        self.llm_model = llm_model or os.getenv(
            "OPENROUTER_MODEL", "openai/gpt-4o-mini"
        )
        self.openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.request_timeout_s = max(2.0, float(request_timeout_s))
        self._shap_explainer = None

        if use_llm is None:
            self.use_llm = bool(self.llm_api_key)
        else:
            self.use_llm = bool(use_llm and self.llm_api_key)

        self._init_shap_explainer()

    # ------------------------------------------------------------------
    # SHAP initialisation
    # ------------------------------------------------------------------

    def _init_shap_explainer(self) -> None:
        lgb_model = None
        if self.vibe_checker is not None:
            lgb_model = getattr(self.vibe_checker, "lgb_model", None)
            if lgb_model is None:
                lgb_model = getattr(self.vibe_checker, "model", None)
        if lgb_model is None:
            logger.info("No LightGBM model available — SHAP explainer not initialized")
            return
        try:
            import shap
            self._shap_explainer = shap.TreeExplainer(lgb_model)
            logger.info("SHAP TreeExplainer initialized on LightGBM (%d features)",
                        lgb_model.num_feature())
        except ImportError:
            logger.warning("SHAP not installed — pip install shap")
        except Exception as exc:
            logger.warning("Failed to init SHAP explainer: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        transaction: Dict[str, Any],
        prediction: float,
        agent_scores: Dict[str, float],
        violations: List[str],
        pipeline_features: Optional[np.ndarray] = None,
    ) -> YapperResult:
        """Run the full SHAP → LLM pipeline and return a structured result.

        Args:
            transaction: Raw transaction dict (for context in prompts).
            prediction: Final fraud probability from decision fusion.
            agent_scores: Per-agent scores.
            violations: Rule violation names.
            pipeline_features: 1-D array of preprocessed features from the
                               centralised FeaturePipeline (175 features).
        """
        if self._shap_explainer is None:
            self._init_shap_explainer()

        # Step 1 — SHAP feature attributions
        feature_contributions = self._calculate_feature_contributions(transaction, pipeline_features)
        shap_available = bool(
            self._shap_explainer is not None
            and feature_contributions
            and any(abs(f.shap_value) > 1e-6 for f in feature_contributions[:5])
        )

        # Step 2 — derive risk factors from SHAP + agent consensus
        top_risk_factors = self._identify_risk_factors(
            transaction, agent_scores, violations, feature_contributions,
        )
        summary = self._generate_summary(prediction, len(violations))

        # Step 3 — LLM plain-English explanation (falls back to template)
        llm_used = False
        if self.use_llm:
            nl_explanation = self._call_llm(
                transaction, prediction, agent_scores, violations,
                feature_contributions, top_risk_factors,
            )
            if nl_explanation:
                llm_used = True
            else:
                nl_explanation = self._generate_template_explanation(
                    transaction, prediction, agent_scores, violations,
                    top_risk_factors, feature_contributions,
                )
        else:
            nl_explanation = self._generate_template_explanation(
                transaction, prediction, agent_scores, violations,
                top_risk_factors, feature_contributions,
            )

        confidence_factors = self._assess_confidence(agent_scores, violations)
        recommended_action = self._recommend_action(prediction, violations)

        return YapperResult(
            summary=summary,
            natural_language_explanation=nl_explanation,
            top_risk_factors=top_risk_factors,
            top_features=feature_contributions[:10],
            shap_available=shap_available,
            llm_used=llm_used,
            confidence_factors=confidence_factors,
            recommended_action=recommended_action,
        )

    # ------------------------------------------------------------------
    # SHAP feature contributions
    # ------------------------------------------------------------------

    def _calculate_feature_contributions(
        self, transaction: Dict[str, Any],
        pipeline_features: Optional[np.ndarray] = None,
    ) -> List[FeatureContribution]:
        if self._shap_explainer is not None and self.vibe_checker is not None and pipeline_features is not None:
            shap_contribs = self._shap_contributions(pipeline_features)
            if shap_contribs:
                return shap_contribs
        return self._heuristic_contributions(transaction)

    def _shap_contributions(
        self, pipeline_features: np.ndarray,
    ) -> List[FeatureContribution]:
        try:
            vector = self.vibe_checker.vectorize_transaction(pipeline_features)
            shap_values = self._shap_explainer.shap_values(vector)
            shap_row = self._extract_shap_row(shap_values)
            if shap_row.size == 0:
                return []

            raw_names = self._get_raw_feature_names(shap_row.size)
            vector_row = vector[0]

            contribs: List[FeatureContribution] = []
            for idx, sv in enumerate(shap_row):
                raw = raw_names[idx]
                fv = float(vector_row[idx]) if idx < len(vector_row) else 0.0
                contribs.append(FeatureContribution(
                    feature_name=FEATURE_LABELS.get(raw, raw),
                    raw_name=raw,
                    feature_value=round(fv, 4),
                    shap_value=round(float(sv), 6),
                    direction="increases_risk" if sv > 0 else "decreases_risk",
                ))
            contribs.sort(key=lambda x: abs(x.shap_value), reverse=True)
            return contribs
        except Exception as exc:
            logger.warning("SHAP contribution failed: %s", exc)
            return []

    @staticmethod
    def _extract_shap_row(shap_values: Any) -> np.ndarray:
        values = shap_values
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        arr = np.asarray(values)
        if arr.ndim == 3:
            arr = arr[0, :, 1] if arr.shape[-1] > 1 else arr[0, :, 0]
        elif arr.ndim == 2:
            arr = arr[0]
        elif arr.ndim > 3:
            arr = arr.reshape(-1)
        return np.asarray(arr, dtype=float)

    def _get_raw_feature_names(self, n: int) -> List[str]:
        raw: Sequence[str] = []
        if self.vibe_checker is not None:
            raw = getattr(self.vibe_checker, "feature_columns", []) or []
        return [raw[i] if i < len(raw) else f"feature_{i}" for i in range(n)]

    def _heuristic_contributions(
        self, transaction: Dict[str, Any],
    ) -> List[FeatureContribution]:
        """Fallback when SHAP is unavailable — rough directional estimates."""
        contribs: List[FeatureContribution] = []
        rules = {
            "TransactionAmt": (0.15, lambda x: 0.3 if x > 1000 else (-0.1 if x < 50 else 0)),
            "hour": (0.08, lambda x: 0.2 if x in (2, 3, 4, 5) else 0),
            "is_night": (0.05, lambda x: 0.15 if x == 1 else 0),
            "card1_txn_count": (0.10, lambda x: -0.1 if x > 10 else (0.1 if x < 3 else 0)),
            "txn_count_1h": (0.12, lambda x: 0.25 if x > 5 else 0),
        }
        for feat, (bw, fn) in rules.items():
            val = transaction.get(feat, 0)
            sv = fn(val) * bw
            contribs.append(FeatureContribution(
                feature_name=FEATURE_LABELS.get(feat, feat),
                raw_name=feat,
                feature_value=val,
                shap_value=round(sv, 6),
                direction="increases_risk" if sv > 0 else "decreases_risk",
            ))
        contribs.sort(key=lambda x: abs(x.shap_value), reverse=True)
        return contribs

    # ------------------------------------------------------------------
    # Risk factors (human-readable bullets)
    # ------------------------------------------------------------------

    def _identify_risk_factors(
        self, transaction: Dict[str, Any], agent_scores: Dict[str, float],
        violations: List[str], features: List[FeatureContribution],
    ) -> List[str]:
        factors: List[str] = []
        for f in features:
            if f.shap_value > 0:
                factors.append(
                    f"{f.feature_name} pushes risk up (SHAP {f.shap_value:+.4f})"
                )
                if len(factors) >= 3:
                    break
        for agent, score in agent_scores.items():
            if score > 0.7:
                factors.append(f"High {agent} score ({score:.0%})")
            elif score > 0.5:
                factors.append(f"Elevated {agent} score ({score:.0%})")
        amt = transaction.get("TransactionAmt", 0)
        if amt > 5000:
            factors.append(f"High-value transaction (${amt:,.2f})")
        hour = transaction.get("hour", 12)
        if hour in (2, 3, 4, 5):
            factors.append(f"Unusual hour ({hour}:00)")
        if violations:
            factors.append(f"{len(violations)} rule violation(s)")
        return factors[:6]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _generate_summary(self, prediction: float, violation_count: int) -> str:
        if prediction > 0.8:
            lvl = "HIGH RISK"
        elif prediction > 0.5:
            lvl = "MEDIUM RISK"
        elif prediction > 0.3:
            lvl = "LOW RISK"
        else:
            lvl = "MINIMAL RISK"
        return f"{lvl} — Fraud probability {prediction:.1%}, {violation_count} rule violation(s)"

    # ------------------------------------------------------------------
    # Template explanation (no LLM)
    # ------------------------------------------------------------------

    def _generate_template_explanation(
        self, transaction: Dict[str, Any], prediction: float,
        agent_scores: Dict[str, float], violations: List[str],
        risk_factors: List[str], features: List[FeatureContribution],
    ) -> str:
        parts: List[str] = []

        # Risk level sentence
        if prediction > 0.7:
            parts.append(
                f"This transaction is HIGH RISK with a {prediction:.1%} fraud probability. "
                "Immediate review is recommended."
            )
        elif prediction > 0.4:
            parts.append(
                f"This transaction shows MODERATE RISK ({prediction:.1%} fraud probability). "
                "Some indicators should be reviewed."
            )
        else:
            parts.append(
                f"This transaction appears LOW RISK ({prediction:.1%} fraud probability). "
                "Standard processing is fine."
            )

        # Agent consensus
        agent_notes = [
            f"{a}: {s:.0%}" for a, s in agent_scores.items() if s > 0.5
        ]
        if agent_notes:
            parts.append("Agent signals: " + ", ".join(agent_notes) + ".")

        # Risk factors
        if risk_factors:
            parts.append("Key drivers: " + "; ".join(risk_factors[:3]) + ".")

        # SHAP top drivers
        positive = [f for f in features if f.shap_value > 0][:3]
        if positive:
            parts.append(
                "Top SHAP drivers: "
                + ", ".join(
                    f"{f.feature_name} ({f.shap_value:+.4f})" for f in positive
                )
                + "."
            )

        # Transaction context
        amt = transaction.get("TransactionAmt", 0)
        hour = transaction.get("hour", "?")
        parts.append(f"Amount: ${amt:,.2f} | Hour: {hour}:00.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # LLM explanation via OpenRouter
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        transaction: Dict[str, Any],
        prediction: float,
        agent_scores: Dict[str, float],
        violations: List[str],
        features: List[FeatureContribution],
        risk_factors: List[str],
    ) -> Optional[str]:
        """Call the LLM and return the explanation string, or None on failure."""
        if not self.llm_api_key:
            return None

        # Build SHAP context for the prompt (top 8 features)
        top_feats = features[:8]
        shap_lines = []
        for f in top_feats:
            arrow = "↑ risk" if f.direction == "increases_risk" else "↓ risk"
            shap_lines.append(
                f"  • {f.feature_name} = {f.feature_value}  →  SHAP {f.shap_value:+.4f} ({arrow})"
            )
        shap_block = "\n".join(shap_lines) if shap_lines else "  (no SHAP data available)"

        # Agent scores context
        agent_block = ", ".join(
            f"{name}: {score:.1%}" for name, score in agent_scores.items()
        )

        # Violations
        viol_block = (
            ", ".join(violations[:5]) if violations else "None"
        )

        # Risk factors
        risk_block = (
            "\n".join(f"  - {r}" for r in risk_factors[:5])
            if risk_factors else "  None identified"
        )

        user_prompt = f"""Here is a fraud detection result. Explain it in simple terms for a non-technical fraud analyst.

TRANSACTION:
  Amount: ${transaction.get('TransactionAmt', 0):,.2f}
  Product: {transaction.get('ProductCD', '?')}
  Card Network: {transaction.get('card4', '?')}
  Card Type: {transaction.get('card6', '?')}
  Hour: {transaction.get('hour', '?')}:00
  Email Domain: {transaction.get('P_emaildomain', '?')}

FRAUD SCORE: {prediction:.1%}
DECISION: {"BLOCK" if prediction > 0.7 else "REVIEW" if prediction > 0.4 else "APPROVE"}

AGENT SCORES: {agent_block}
RULE VIOLATIONS: {viol_block}

TOP SHAP FEATURE CONTRIBUTIONS (what the model cares about most):
{shap_block}

KEY RISK FACTORS:
{risk_block}

Please write:
1. A one-sentence verdict (is this likely fraud or legit, and how confident).
2. A short paragraph (3-4 sentences) explaining WHY in plain English — mention the specific factors that pushed the score up or down, referring to things like "the transaction amount", "the time of day", "the card type" etc. instead of technical feature names.
3. A one-sentence recommended next step for the analyst."""

        system_prompt = (
            "You are a fraud detection explainer. Your audience is a non-technical "
            "fraud analyst who needs to understand why a transaction was flagged. "
            "Use simple everyday language. Avoid jargon like 'SHAP values', "
            "'ensemble scores', or model names. Refer to features by their real-world "
            "meaning (e.g. 'transaction amount' not 'TransactionAmt_log'). "
            "Be concise and actionable."
        )

        body = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 350,
        }
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv(
                "OPENROUTER_SITE_URL", "http://localhost"
            ),
            "X-Title": os.getenv(
                "OPENROUTER_APP_NAME", "Fraud Detection System"
            ),
        }

        try:
            resp = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if content:
                logger.info("LLM explanation generated (%d chars)", len(content))
                return content
            logger.warning("LLM returned empty content")
        except requests.exceptions.Timeout:
            logger.warning("LLM request timed out after %.1fs", self.request_timeout_s)
        except requests.exceptions.HTTPError as e:
            logger.warning("LLM HTTP error: %s", e)
        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Confidence assessment
    # ------------------------------------------------------------------

    def _assess_confidence(
        self, agent_scores: Dict[str, float], violations: List[str],
    ) -> List[str]:
        factors: List[str] = []
        scores = list(agent_scores.values())
        if len(scores) >= 2:
            std = float(np.std(scores))
            if std < 0.1:
                factors.append("High confidence — all agents agree")
            elif std > 0.25:
                factors.append("Low confidence — agent scores diverge significantly")
        high_sev = sum(1 for v in violations if "HIGH" in str(v).upper())
        if high_sev > 0:
            factors.append(f"{high_sev} high-severity rule(s) support the prediction")
        if self._shap_explainer is not None:
            factors.append("Prediction grounded in SHAP feature attributions")
        return factors

    def _recommend_action(self, prediction: float, violations: List[str]) -> str:
        if prediction > 0.8:
            return "BLOCK — Decline transaction and flag account for review"
        elif prediction > 0.6:
            return "CHALLENGE — Request additional verification (2FA / security questions)"
        elif prediction > 0.4:
            return "REVIEW — Queue for manual analyst review"
        elif prediction > 0.2:
            return "MONITOR — Approve with enhanced monitoring"
        else:
            return "APPROVE — Process transaction normally"
