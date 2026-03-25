# =============================================================================
# OG CHECK - Rule-based fraud detection with learned thresholds
# =============================================================================

import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).resolve().parents[2] / "models" / "og_check_params.json"


@dataclass
class RuleViolation:
    """A single rule violation."""
    rule_name: str
    severity: str  # 'high', 'medium', 'low'
    description: str
    risk_contribution: float


@dataclass
class OGCheckResult:
    """Result from rule-based analysis."""
    fraud_score: float
    violations: List[RuleViolation]
    passed_rules: int
    failed_rules: int
    explanation: str


class OGCheck:
    """
    OG Check — Rule-based fraud detection agent.

    Applies deterministic business rules (amount limits, velocity, blacklists,
    time, card, email, device checks) and uses a trained logistic-regression
    model on binary rule indicators for the final fraud score.
    Falls back to violation-sum scoring if no trained params are available.
    """

    def __init__(
        self,
        velocity_limits: Optional[Dict[str, int]] = None,
        amount_limits: Optional[Dict[str, float]] = None,
        redis_client: Optional[Any] = None,
        params_path: Optional[str] = None,
    ):
        self.redis_client = redis_client
        self.model_loaded = False
        self._lgb_model = None
        self._use_lgb = False
        self._og_threshold = 0.5
        self._og_features = []

        self._learned_weights: Dict[str, float] = {}
        self._logreg_coefs: Dict[str, float] = {}
        self._intercept: float = 0.0
        self._learned_thresholds: Dict[str, Any] = {}
        self._load_params(params_path)

        self.velocity_limits = velocity_limits or {
            'txn_count_1h': 10,
            'txn_count_24h': 50,
            'amount_1h': 5000,
            'amount_24h': 20000,
        }

        if amount_limits:
            self.amount_limits = amount_limits
        elif self.model_loaded:
            t = self._learned_thresholds
            self.amount_limits = {
                'max_single': t.get('max_single', 10000),
                'suspicious_round': [100, 200, 500, 1000, 2000, 5000],
                'high_risk_threshold': t.get('high_risk_threshold', 3000),
                'micro_threshold': t.get('micro_threshold', 1.0),
            }
        else:
            self.amount_limits = {
                'max_single': 10000,
                'suspicious_round': [100, 200, 500, 1000, 2000, 5000],
                'high_risk_threshold': 3000,
                'micro_threshold': 1.0,
            }

        self.blacklisted_cards: set = set()
        self.blacklisted_emails: set = set()
        self.blacklisted_devices: set = set()
        self.high_risk_countries = {'XX', 'YY', 'ZZ'}

    def _load_params(self, params_path: Optional[str] = None) -> None:
        path = Path(params_path) if params_path else PARAMS_PATH
        try:
            if path.exists():
                data = json.loads(path.read_text())
                self._learned_thresholds = data.get("thresholds", {})
                raw_weights = data.get("rule_weights", {})
                self._learned_weights = {
                    k: v["risk_contribution"] for k, v in raw_weights.items()
                }
                self._logreg_coefs = data.get("logreg_coefficients", {})
                self._intercept = data.get("intercept", 0.0)
                self._og_threshold = data.get("best_threshold", 0.5)
                self._og_features = data.get("features", [])
                self.model_loaded = True
                logger.info("OG Check params loaded from %s", path)

                # Load LightGBM model if available
                og_model_file = data.get("og_model_path", "")
                if og_model_file:
                    model_dir = path.parent
                    og_model_path = model_dir / og_model_file
                    if og_model_path.exists():
                        try:
                            import lightgbm as lgb
                            self._lgb_model = lgb.Booster(model_file=str(og_model_path))
                            self._use_lgb = True
                            logger.info("OG Check LightGBM model loaded from %s", og_model_path)
                        except ImportError:
                            logger.warning("lightgbm not installed — falling back to rule scoring")
                        except Exception as e2:
                            logger.warning("Failed to load OG LightGBM model: %s", e2)
            else:
                logger.warning("OG Check params not found at %s — using defaults", path)
        except Exception as e:
            logger.warning("Failed to load OG Check params: %s — using defaults", e)

    def analyze(self, transaction: Dict[str, Any], pipeline_features: Optional[Any] = None) -> OGCheckResult:
        """Analyse a transaction using deterministic rules + LightGBM model.

        Args:
            transaction: Raw transaction dict (used for rule checks).
            pipeline_features: 1-D array of preprocessed features from the
                               centralised FeaturePipeline (175 features).
        """
        # Defensively replace None values with safe defaults.
        transaction = {
            k: (v if v is not None else 0) for k, v in transaction.items()
        }
        violations: List[RuleViolation] = []

        violations.extend(self._check_amount_rules(transaction))
        violations.extend(self._check_velocity_rules(transaction))
        violations.extend(self._check_blacklist_rules(transaction))
        violations.extend(self._check_time_rules(transaction))
        violations.extend(self._check_card_rules(transaction))
        violations.extend(self._check_email_rules(transaction))
        violations.extend(self._check_device_rules(transaction))

        total_rules = 15
        passed_rules = total_rules - len(violations)
        fraud_score = self._calculate_rule_score(violations, transaction, pipeline_features)
        explanation = self._generate_explanation(violations)

        return OGCheckResult(
            fraud_score=fraud_score,
            violations=violations,
            passed_rules=passed_rules,
            failed_rules=len(violations),
            explanation=explanation,
        )

    def _get_risk_weight(self, rule_name: str, default: float) -> float:
        return self._learned_weights.get(rule_name, default)

    # ------------------------------------------------------------------
    # Rule checks
    # ------------------------------------------------------------------

    def _check_amount_rules(self, transaction: Dict[str, Any]) -> List[RuleViolation]:
        violations = []
        amount = transaction.get('TransactionAmt') or 0
        max_single = self.amount_limits.get('max_single', 10000)
        high_risk = self.amount_limits.get('high_risk_threshold', 3000)
        micro = self.amount_limits.get('micro_threshold', 1.0)
        suspicious_round = self.amount_limits.get('suspicious_round', [100, 200, 500, 1000, 2000, 5000])

        if amount > max_single:
            violations.append(RuleViolation(
                rule_name='MAX_AMOUNT', severity='high',
                description=f"Amount ${amount:.2f} exceeds maximum ${max_single}",
                risk_contribution=self._get_risk_weight('MAX_AMOUNT', 0.3),
            ))
        if amount > high_risk:
            violations.append(RuleViolation(
                rule_name='HIGH_AMOUNT', severity='medium',
                description=f"Amount ${amount:.2f} exceeds high-risk threshold ${high_risk}",
                risk_contribution=self._get_risk_weight('HIGH_AMOUNT', 0.15),
            ))
        if amount in suspicious_round:
            violations.append(RuleViolation(
                rule_name='ROUND_AMOUNT', severity='low',
                description=f"Suspicious round amount: ${amount:.2f}",
                risk_contribution=self._get_risk_weight('ROUND_AMOUNT', 0.05),
            ))
        if 0 < amount < micro:
            violations.append(RuleViolation(
                rule_name='MICRO_AMOUNT', severity='high',
                description=f"Micro-transaction ${amount:.2f} - possible card testing",
                risk_contribution=self._get_risk_weight('MICRO_AMOUNT', 0.25),
            ))
        return violations

    def _check_velocity_rules(self, transaction: Dict[str, Any]) -> List[RuleViolation]:
        violations = []
        txn_count_1h = transaction.get('txn_count_1h', 0)
        txn_count_24h = transaction.get('txn_count_24h', 0)
        amount_1h = transaction.get('amount_1h', 0)

        if txn_count_1h > self.velocity_limits['txn_count_1h']:
            violations.append(RuleViolation(
                rule_name='VELOCITY_TXN_1H', severity='high',
                description=f"{txn_count_1h} transactions in last hour exceeds limit",
                risk_contribution=self._get_risk_weight('VELOCITY_TXN_1H', 0.3),
            ))
        if txn_count_24h > self.velocity_limits['txn_count_24h']:
            violations.append(RuleViolation(
                rule_name='VELOCITY_TXN_24H', severity='medium',
                description=f"{txn_count_24h} transactions in last 24h exceeds limit",
                risk_contribution=self._get_risk_weight('VELOCITY_TXN_24H', 0.2),
            ))
        current_amount = transaction.get('TransactionAmt') or 0
        if amount_1h + current_amount > self.velocity_limits['amount_1h']:
            violations.append(RuleViolation(
                rule_name='VELOCITY_AMOUNT_1H', severity='high',
                description=f"Total amount ${amount_1h + current_amount:.2f} in 1h exceeds limit",
                risk_contribution=self._get_risk_weight('VELOCITY_AMOUNT_1H', 0.25),
            ))
        return violations

    def _check_blacklist_rules(self, transaction: Dict[str, Any]) -> List[RuleViolation]:
        violations = []
        if transaction.get('card1', '') in self.blacklisted_cards:
            violations.append(RuleViolation(
                rule_name='BLACKLIST_CARD', severity='high',
                description="Card is on blacklist",
                risk_contribution=self._get_risk_weight('BLACKLIST_CARD', 0.5),
            ))
        if transaction.get('P_emaildomain', '') in self.blacklisted_emails:
            violations.append(RuleViolation(
                rule_name='BLACKLIST_EMAIL', severity='high',
                description="Email domain is on blacklist",
                risk_contribution=self._get_risk_weight('BLACKLIST_EMAIL', 0.4),
            ))
        if transaction.get('DeviceInfo', '') in self.blacklisted_devices:
            violations.append(RuleViolation(
                rule_name='BLACKLIST_DEVICE', severity='high',
                description="Device is on blacklist",
                risk_contribution=self._get_risk_weight('BLACKLIST_DEVICE', 0.4),
            ))
        return violations

    def _check_time_rules(self, transaction: Dict[str, Any]) -> List[RuleViolation]:
        violations = []
        hour = transaction.get('hour', 12)
        if hour in [2, 3, 4, 5]:
            violations.append(RuleViolation(
                rule_name='LATE_NIGHT', severity='low',
                description=f"Transaction at unusual hour: {hour}:00",
                risk_contribution=self._get_risk_weight('LATE_NIGHT', 0.1),
            ))
        is_weekend = transaction.get('is_weekend', 0)
        amount = transaction.get('TransactionAmt', 0)
        if is_weekend and amount > 1000:
            violations.append(RuleViolation(
                rule_name='WEEKEND_HIGH_VALUE', severity='low',
                description="High-value weekend transaction",
                risk_contribution=self._get_risk_weight('WEEKEND_HIGH', 0.05),
            ))
        return violations

    def _check_card_rules(self, transaction: Dict[str, Any]) -> List[RuleViolation]:
        violations = []
        cards_per_device = transaction.get('cards_per_device', 1)
        if cards_per_device > 3:
            violations.append(RuleViolation(
                rule_name='MULTI_CARD_DEVICE', severity='medium',
                description=f"{cards_per_device} different cards used on same device",
                risk_contribution=self._get_risk_weight('MULTI_CARD_DEVICE', 0.2),
            ))
        return violations

    def _check_email_rules(self, transaction: Dict[str, Any]) -> List[RuleViolation]:
        violations = []
        p_email = transaction.get('P_emaildomain', '')
        r_email = transaction.get('R_emaildomain', '')
        if p_email and r_email and p_email != r_email:
            violations.append(RuleViolation(
                rule_name='EMAIL_MISMATCH', severity='low',
                description="Purchaser and recipient email domains differ",
                risk_contribution=self._get_risk_weight('EMAIL_MISMATCH', 0.05),
            ))
        disposable_domains = ['tempmail', 'throwaway', 'guerrillamail', 'mailinator']
        if any(d in str(p_email).lower() for d in disposable_domains):
            violations.append(RuleViolation(
                rule_name='DISPOSABLE_EMAIL', severity='medium',
                description="Disposable email address detected",
                risk_contribution=self._get_risk_weight('DISPOSABLE_EMAIL', 0.15),
            ))
        return violations

    def _check_device_rules(self, transaction: Dict[str, Any]) -> List[RuleViolation]:
        violations = []
        device_info = str(transaction.get('DeviceInfo', '')).lower()
        amount = transaction.get('TransactionAmt', 0)
        if not device_info and amount > 500:
            violations.append(RuleViolation(
                rule_name='NO_DEVICE_HIGH_VALUE', severity='medium',
                description="No device info for high-value transaction",
                risk_contribution=self._get_risk_weight('NO_DEVICE_HIGH_VALUE', 0.15),
            ))
        bot_indicators = ['headless', 'phantom', 'selenium', 'crawler']
        if any(ind in device_info for ind in bot_indicators):
            violations.append(RuleViolation(
                rule_name='BOT_DEVICE', severity='high',
                description="Automated/bot device detected",
                risk_contribution=self._get_risk_weight('BOT_DEVICE', 0.35),
            ))
        return violations

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _calculate_rule_score(self, violations: List[RuleViolation], transaction: Dict[str, Any],
                              pipeline_features: Optional[Any] = None) -> float:
        if self._use_lgb and self._lgb_model is not None:
            return self._lgb_score(transaction, pipeline_features)
        if self._logreg_coefs:
            return self._logreg_score(transaction)
        if not violations:
            return 0.0
        return min(sum(v.risk_contribution for v in violations), 1.0)

    def _build_og_features(self, transaction: Dict[str, Any]) -> List[float]:
        """Build the 20-feature vector matching training OG_FEATURES."""
        amt = float(transaction.get('TransactionAmt', 0))
        hour = int(transaction.get('hour', 12))
        is_weekend = int(transaction.get('is_weekend', 0))
        p_email = str(transaction.get('P_emaildomain', '') or '')
        r_email = str(transaction.get('R_emaildomain', '') or '')
        device_info = str(transaction.get('DeviceInfo', '') or '')
        addr1 = transaction.get('addr1', None)
        card4 = str(transaction.get('card4', 'unknown'))

        max_single = self.amount_limits.get('max_single', 10000)
        high_risk = self.amount_limits.get('high_risk_threshold', 3000)
        micro = self.amount_limits.get('micro_threshold', 1.0)
        thresholds = self._learned_thresholds

        import math as _math

        features = [
            # Original 11 (rule_amount_log removed — pipeline handles it)
            1 if amt > max_single else 0,                           # rule_max_amount
            1 if amt > high_risk else 0,                            # rule_high_amount
            1 if (0 < amt < micro) else 0,                          # rule_micro_amount
            1 if hour in (2, 3, 4, 5) else 0,                      # rule_late_night
            1 if (is_weekend and amt > 1000) else 0,                # rule_weekend_high
            1 if amt in {100, 200, 500, 1000, 2000, 5000} else 0,  # rule_round_amount
            1 if (p_email and r_email and p_email != r_email) else 0,  # rule_email_mismatch
            1 if (not device_info and amt > 500) else 0,            # rule_no_device_high_value
            1 if hour in (0,1,2,3,4,5,22,23) else 0,               # rule_is_night
            1 if self._is_missing(addr1) else 0,                    # rule_addr_missing
            1 if card4 != 'visa' else 0,                            # rule_card_not_visa
            # New 8
            1 if self._is_missing(transaction.get('addr2')) else 0,                  # rule_addr2_missing
            1 if self._is_missing(transaction.get('D1')) else 0,                     # rule_D1_missing
            1 if float(transaction.get('D1', 0) or 0) > thresholds.get('D1_p95', 999999) else 0,  # rule_D1_high
            1 if float(transaction.get('C1', 0) or 0) > thresholds.get('C1_p95', 999999) else 0,  # rule_C1_high
            1 if float(transaction.get('C13', 0) or 0) > thresholds.get('C13_p95', 999999) else 0, # rule_C13_high
            self._count_m_mismatches(transaction),                                    # rule_M_mismatch_count
            1 if self._is_missing(transaction.get('id_01')) else 0,                  # rule_id_missing
            1 if amt > thresholds.get('moderate_spike', 999999) else 0,              # rule_moderate_spike
        ]
        return features

    @staticmethod
    def _is_missing(val) -> bool:
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        if isinstance(val, str) and val.strip() in ('', 'nan', 'None'):
            return True
        return False

    @staticmethod
    def _count_m_mismatches(transaction: Dict[str, Any]) -> float:
        count = 0
        for i in range(1, 10):
            v = transaction.get(f'M{i}', '')
            if str(v).strip().upper() == 'F':
                count += 1
        return float(count)

    def _lgb_score(self, transaction: Dict[str, Any],
                   pipeline_features: Optional[Any] = None) -> float:
        import numpy as _np
        rule_features = self._build_og_features(transaction)
        if pipeline_features is not None:
            pipe = _np.asarray(pipeline_features, dtype=_np.float64).ravel()
            combined = _np.concatenate([pipe, rule_features])
        else:
            combined = _np.asarray(rule_features, dtype=_np.float64)
        raw = float(self._lgb_model.predict([combined])[0])
        return max(0.0, min(raw, 1.0))

    def _logreg_score(self, transaction: Dict[str, Any]) -> float:
        amt = float(transaction.get('TransactionAmt', 0))
        hour = int(transaction.get('hour', 12))
        is_weekend = int(transaction.get('is_weekend', 0))
        p_email = str(transaction.get('P_emaildomain', ''))
        r_email = str(transaction.get('R_emaildomain', ''))
        device_info = str(transaction.get('DeviceInfo', ''))
        addr1 = transaction.get('addr1', None)
        card4 = str(transaction.get('card4', 'unknown'))

        max_single = self.amount_limits.get('max_single', 10000)
        high_risk = self.amount_limits.get('high_risk_threshold', 3000)
        micro = self.amount_limits.get('micro_threshold', 1.0)

        features = {
            'rule_max_amount': 1 if amt > max_single else 0,
            'rule_high_amount': 1 if amt > high_risk else 0,
            'rule_micro_amount': 1 if (0 < amt < micro) else 0,
            'rule_late_night': 1 if hour in (2, 3, 4, 5) else 0,
            'rule_weekend_high': 1 if (is_weekend and amt > 1000) else 0,
            'rule_round_amount': 1 if amt in {100, 200, 500, 1000, 2000, 5000} else 0,
            'rule_email_mismatch': 1 if (p_email and r_email and p_email != r_email) else 0,
            'rule_no_device_high_value': 1 if (not device_info and amt > 500) else 0,
            'rule_is_night': 1 if hour in (0, 1, 2, 3, 4, 5, 22, 23) else 0,
            'rule_addr_missing': 1 if (addr1 is None or (isinstance(addr1, float) and math.isnan(addr1))) else 0,
            'rule_card_not_visa': 1 if card4 != 'visa' else 0,
        }

        logit = self._intercept
        for feat, coef in self._logreg_coefs.items():
            logit += coef * features.get(feat, 0)

        return max(0.0, min(1.0 / (1.0 + math.exp(-logit)), 1.0))

    def _generate_explanation(self, violations: List[RuleViolation]) -> str:
        if not violations:
            return "All business rules passed — OG Check clear"
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_v = sorted(violations, key=lambda x: severity_order.get(x.severity, 3))
        lines = [f"[{v.severity.upper()}] {v.description}" for v in sorted_v[:5]]
        return "; ".join(lines)
