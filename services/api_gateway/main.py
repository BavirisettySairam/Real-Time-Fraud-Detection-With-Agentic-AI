# =============================================================================
# FASTAPI API GATEWAY - Production-ready local deployment
# =============================================================================

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from services.orchestrator import AgentOrchestrator, OrchestratorConfig

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class APISettings:
    service_name: str = "Fraud Detection API"
    version: str = "1.2.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_allow_origins: tuple[str, ...] = ("http://localhost:8501",)
    api_key: Optional[str] = None
    max_body_bytes: int = 1_048_576  # 1 MB
    rate_limit_predict: str = "60/minute"
    rate_limit_explain: str = "30/minute"
    rate_limit_default: str = "120/minute"
    batch_max_size: int = 256
    batch_max_workers: int = 8
    use_langgraph: bool = True
    strict_langgraph: bool = True
    orchestrator_parallel: bool = True
    orchestrator_timeout_ms: int = 5000
    model_primary_weight: float = 0.75

    @classmethod
    def from_env(cls) -> "APISettings":
        origins_env = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8501")
        origins = tuple(o.strip() for o in origins_env.split(",") if o.strip())

        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            cors_allow_origins=origins,
            api_key=os.getenv("API_KEY") or None,
            max_body_bytes=max(1024, int(os.getenv("MAX_BODY_BYTES", "1048576"))),
            rate_limit_predict=os.getenv("RATE_LIMIT_PREDICT", "60/minute"),
            rate_limit_explain=os.getenv("RATE_LIMIT_EXPLAIN", "30/minute"),
            rate_limit_default=os.getenv("RATE_LIMIT_DEFAULT", "120/minute"),
            batch_max_size=max(1, int(os.getenv("BATCH_MAX_SIZE", "256"))),
            batch_max_workers=max(1, int(os.getenv("BATCH_MAX_WORKERS", "8"))),
            use_langgraph=_env_bool("USE_LANGGRAPH", True),
            strict_langgraph=_env_bool("STRICT_LANGGRAPH", True),
            orchestrator_parallel=_env_bool("ORCHESTRATOR_PARALLEL", True),
            orchestrator_timeout_ms=max(100, int(os.getenv("ORCHESTRATOR_TIMEOUT_MS", "5000"))),
            model_primary_weight=min(max(float(os.getenv("MODEL_PRIMARY_WEIGHT", "0.75")), 0.0), 1.0),
        )


SETTINGS = APISettings.from_env()
logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# METRICS
# =============================================================================


if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "fraud_api_requests_total",
        "Total API requests",
        ["method", "path", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "fraud_api_request_latency_seconds",
        "API request latency in seconds",
        ["method", "path"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
    )
    PREDICTION_COUNT = Counter(
        "fraud_predictions_total",
        "Total fraud predictions",
        ["endpoint", "decision"],
    )
    PREDICTION_LATENCY = Histogram(
        "fraud_prediction_latency_seconds",
        "Prediction latency in seconds",
        ["endpoint"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
    )


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


_ALLOWED_EXTRA_FIELD = re.compile(
    r"^(V\d{1,3}|C\d{1,2}|D\d{1,2}|M\d|id_\d{2}|dist\d|isFraud|TransactionDT)$"
)
_MAX_TRANSACTION_FIELDS = 500


class Transaction(BaseModel):
    """Transaction input model with field-name validation."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "TransactionID": 3663549,
                "TransactionDT": 86400,
                "TransactionAmt": 150.00,
                "ProductCD": "W",
                "card1": 12345,
                "card4": "visa",
                "C1": 3,
                "V12": 0.0,
                "hour": 14,
                "P_emaildomain": "gmail.com",
            }
        }
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if len(data) > _MAX_TRANSACTION_FIELDS:
            raise ValueError(f"Too many fields ({len(data)}), maximum is {_MAX_TRANSACTION_FIELDS}")
        known = set(cls.model_fields.keys())
        for key in set(data.keys()) - known:
            if not _ALLOWED_EXTRA_FIELD.match(str(key)):
                raise ValueError(f"Unknown field: {key}")
        return data

    TransactionID: Optional[int] = Field(None, description="Unique transaction ID")
    TransactionDT: Optional[float] = Field(None, description="Seconds from dataset reference start")
    TransactionAmt: float = Field(..., description="Transaction amount", ge=0)
    ProductCD: Optional[str] = Field("W", description="Product code")
    card1: Optional[int] = Field(None, description="Card identifier")
    card2: Optional[float] = Field(None, description="Card field 2")
    card3: Optional[float] = Field(None, description="Card field 3")
    card4: Optional[str] = Field(None, description="Card type")
    card5: Optional[float] = Field(None, description="Card field 5")
    card6: Optional[str] = Field(None, description="Card category")
    addr1: Optional[float] = Field(None, description="Address field 1")
    addr2: Optional[float] = Field(None, description="Address field 2")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    DeviceType: Optional[str] = Field(None, description="Device type")
    DeviceInfo: Optional[str] = Field(None, description="Device information")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of transaction")
    day: Optional[int] = Field(None, ge=0, le=6, description="Day of week")
    txn_count_1h: Optional[int] = Field(0, ge=0, description="Transactions in last hour")
    amount_1h: Optional[float] = Field(0, ge=0, description="Total amount in last hour")


class PredictionResponse(BaseModel):
    """Fraud prediction response."""

    transaction_id: Optional[int]
    fraud_probability: float = Field(..., ge=0, le=1)
    decision: str = Field(..., description="APPROVE, REVIEW, or BLOCK")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    explanation: str
    agent_scores: Dict[str, float]
    rule_violations: List[str]
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str
    uptime_seconds: float
    components: Dict[str, str]


class BatchTransaction(BaseModel):
    """Batch of transactions for bulk processing."""

    transactions: List[Transaction] = Field(..., min_length=1)


class FeatureContributionResponse(BaseModel):
    """Single SHAP feature contribution."""
    feature_name: str
    raw_name: str
    feature_value: Any = None
    shap_value: float
    direction: str


class ExplainResponse(BaseModel):
    """Detailed SHAP + LLM explanation response."""
    transaction_id: Optional[int]
    fraud_probability: float = Field(..., ge=0, le=1)
    decision: str
    risk_level: str
    summary: str
    natural_language_explanation: str
    top_risk_factors: List[str]
    top_features: List[FeatureContributionResponse]
    shap_available: bool
    llm_used: bool
    confidence_factors: List[str]
    recommended_action: str
    agent_scores: Dict[str, float]
    rule_violations: List[str]
    processing_time_ms: float
    timestamp: str


class BatchResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[PredictionResponse]
    total_processed: int
    total_time_ms: float


# =============================================================================
# HELPERS
# =============================================================================


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_time_features(txn: Dict[str, Any]) -> Dict[str, Any]:
    """Derive time features when missing from request payload."""
    now = datetime.now(timezone.utc)
    if txn.get("hour") is None:
        txn["hour"] = now.hour
    if txn.get("day") is None:
        txn["day"] = now.weekday()
    txn["is_weekend"] = 1 if txn.get("day", 0) in [5, 6] else 0
    txn["is_night"] = 1 if txn.get("hour", 12) in [0, 1, 2, 3, 4, 5, 22, 23] else 0
    return txn


def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.4:
        return "MEDIUM"
    return "LOW"


def _build_prediction_response(transaction: Transaction, result: Dict[str, Any]) -> PredictionResponse:
    score = float(result.get("final_score", 0.5))
    return PredictionResponse(
        transaction_id=transaction.TransactionID,
        fraud_probability=max(0.0, min(score, 1.0)),
        decision=str(result.get("final_decision", "REVIEW")),
        risk_level=get_risk_level(score),
        explanation=str(result.get("explanation", "Analysis complete")),
        agent_scores={
            "vibe_checker": float(result.get("vibe_score", 0.5)),
            "agent_ensemble": float(result.get("agent_ensemble_score", 0.5)),
            "era_tracker": float(result.get("era_score", 0.5)),
            "og_check": float(result.get("og_score", 0.5)),
        },
        rule_violations=list(result.get("og_violations", [])),
        processing_time_ms=float(result.get("processing_time_ms", 0.0)),
        timestamp=_utc_now_iso(),
    )


def _observe_request_metrics(method: str, path: str, status: int, duration_s: float) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration_s)


def _observe_prediction_metrics(endpoint: str, decision: str, duration_s: float) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    PREDICTION_COUNT.labels(endpoint=endpoint, decision=decision).inc()
    PREDICTION_LATENCY.labels(endpoint=endpoint).observe(duration_s)


# =============================================================================
# APP FACTORY
# =============================================================================


def create_app(settings: Optional[APISettings] = None) -> FastAPI:
    cfg = settings or SETTINGS

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        app_instance.state.orchestrator = AgentOrchestrator(
            OrchestratorConfig(
                model_primary_weight=cfg.model_primary_weight,
                use_langgraph=cfg.use_langgraph,
                strict_langgraph=cfg.strict_langgraph,
                enable_parallel=cfg.orchestrator_parallel,
                timeout_ms=cfg.orchestrator_timeout_ms,
            )
        )
        logger.info(
            "API startup complete: batch_max_size=%s batch_max_workers=%s use_langgraph=%s",
            cfg.batch_max_size,
            cfg.batch_max_workers,
            cfg.use_langgraph,
        )
        try:
            yield
        finally:
            orch = app_instance.state.orchestrator
            if orch is not None:
                orch.close()
            logger.info("API shutdown complete")

    app = FastAPI(
        title=cfg.service_name,
        description=(
            "Fraud detection API using Vibe Checker, Era Tracker, OG Check, and "
            "The Yapper agents with explainable output and production-grade observability."
        ),
        version=cfg.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.state.settings = cfg
    app.state.started_monotonic = time.monotonic()
    app.state.orchestrator = None

    # -- Rate limiter setup --------------------------------------------------
    if SLOWAPI_AVAILABLE:
        limiter = Limiter(key_func=get_remote_address, default_limits=[cfg.rate_limit_default])
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    else:
        limiter = None
        logger.warning("slowapi not installed — rate limiting disabled")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )

    # -- Middleware: request tracing (innermost — defined first) ---------------
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id

        response: Optional[Response] = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_s = time.perf_counter() - start
            _observe_request_metrics(
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_s=duration_s,
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time-ms"] = f"{duration_s * 1000:.3f}"

    # -- Middleware: API key authentication ------------------------------------
    _PUBLIC_PATHS = frozenset({"/", "/health", "/ready", "/docs", "/redoc", "/openapi.json", "/metrics"})

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if cfg.api_key and request.url.path not in _PUBLIC_PATHS:
            provided = request.headers.get("X-API-Key", "")
            if not provided or not hmac.compare_digest(provided, cfg.api_key):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
        return await call_next(request)

    # -- Middleware: max request body size -------------------------------------
    @app.middleware("http")
    async def max_body_size_middleware(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > cfg.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (max {cfg.max_body_bytes} bytes)"},
            )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Validation error: %s", exc)
        errors = []
        for err in exc.errors():
            safe = {k: v for k, v in err.items() if k not in ("ctx", "url")}
            errors.append(safe)
        return JSONResponse(
            status_code=422,
            content={
                "detail": errors,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    def get_orchestrator() -> AgentOrchestrator:
        orch = app.state.orchestrator
        if orch is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        return orch

    @app.get("/", tags=["Root"])
    async def root() -> Dict[str, str]:
        return {
            "message": cfg.service_name,
            "version": cfg.version,
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        uptime = max(0.0, time.monotonic() - app.state.started_monotonic)
        components = {
            "api": "healthy",
            "orchestrator": "healthy" if app.state.orchestrator else "starting",
            "agents": "healthy" if app.state.orchestrator else "starting",
        }
        status = "healthy" if app.state.orchestrator else "degraded"
        return HealthResponse(
            status=status,
            version=cfg.version,
            timestamp=_utc_now_iso(),
            uptime_seconds=uptime,
            components=components,
        )

    @app.get("/ready", tags=["Health"])
    async def readiness_check() -> Dict[str, str]:
        if app.state.orchestrator is None:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        return {"status": "ready"}

    _predict_decorator = limiter.limit(cfg.rate_limit_predict) if limiter else (lambda f: f)
    _explain_decorator = limiter.limit(cfg.rate_limit_explain) if limiter else (lambda f: f)

    @app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Predictions"])
    @_predict_decorator
    async def predict_fraud(request: Request, transaction: Transaction) -> PredictionResponse:
        orchestrator = get_orchestrator()
        start = time.perf_counter()

        txn_dict = derive_time_features(transaction.model_dump())
        result = await run_in_threadpool(orchestrator.analyze, txn_dict)
        response = _build_prediction_response(transaction, result)

        _observe_prediction_metrics(
            endpoint="predict",
            decision=response.decision,
            duration_s=time.perf_counter() - start,
        )
        return response

    @app.post("/api/v1/predict/batch", response_model=BatchResponse, tags=["Predictions"])
    @_predict_decorator
    async def predict_batch(request: Request, batch: BatchTransaction) -> BatchResponse:
        orchestrator = get_orchestrator()

        if len(batch.transactions) > cfg.batch_max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large. Maximum allowed: {cfg.batch_max_size}",
            )

        start = time.perf_counter()
        semaphore = asyncio.Semaphore(cfg.batch_max_workers)

        async def process_one(transaction: Transaction) -> PredictionResponse:
            async with semaphore:
                item_start = time.perf_counter()
                txn_dict = derive_time_features(transaction.model_dump())
                result = await run_in_threadpool(orchestrator.analyze, txn_dict)
                response = _build_prediction_response(transaction, result)
                _observe_prediction_metrics(
                    endpoint="predict_batch",
                    decision=response.decision,
                    duration_s=time.perf_counter() - item_start,
                )
                return response

        predictions = await asyncio.gather(*(process_one(txn) for txn in batch.transactions))
        total_time_ms = (time.perf_counter() - start) * 1000

        return BatchResponse(
            predictions=predictions,
            total_processed=len(predictions),
            total_time_ms=total_time_ms,
        )

    @app.get("/api/v1/agents", tags=["Info"])
    async def get_agents_info() -> Dict[str, Any]:
        orchestrator = get_orchestrator()
        return {
            "agents": [
                {
                    "name": "Vibe Checker",
                    "description": "ML Ensemble (XGBoost + LightGBM)",
                    "weight": orchestrator.config.model_primary_weight,
                    "role": "primary",
                },
                {
                    "name": "Era Tracker",
                    "description": "24-hour window behavioural analysis",
                    "weight": orchestrator.config.era_weight,
                    "role": "secondary",
                },
                {
                    "name": "OG Check",
                    "description": "Business rule checks",
                    "weight": orchestrator.config.og_weight,
                    "role": "secondary",
                },
                {
                    "name": "The Yapper",
                    "description": "SHAP + LLM explanation agent",
                    "weight": "N/A",
                },
            ],
            "thresholds": {
                "high_risk": orchestrator.config.high_risk_threshold,
                "medium_risk": orchestrator.config.medium_risk_threshold,
            },
            "score_blending": {
                "model_primary_weight": orchestrator.config.model_primary_weight,
                "agent_ensemble_weight": round(1.0 - orchestrator.config.model_primary_weight, 4),
            },
            "execution": {
                "parallel_enabled": orchestrator.config.enable_parallel,
                "timeout_ms": orchestrator.config.timeout_ms,
                "use_langgraph": orchestrator.config.use_langgraph,
                "strict_langgraph": orchestrator.config.strict_langgraph,
                "langgraph_active": orchestrator.workflow is not None,
            },
        }

    @app.get("/api/v1/workflow", tags=["Info"])
    async def get_workflow() -> Dict[str, Any]:
        orchestrator = get_orchestrator()
        mermaid = orchestrator.get_workflow_visualization()
        return {
            "workflow": "LangGraph Agent Pipeline",
            "mode": "parallel" if orchestrator.config.enable_parallel else "sequential",
            "langgraph_active": orchestrator.workflow is not None,
            "generated_at": _utc_now_iso(),
            "mermaid_diagram": mermaid,
        }

    @app.post("/api/v1/explain", response_model=ExplainResponse, tags=["Predictions"])
    @_explain_decorator
    async def explain_fraud(request: Request, transaction: Transaction) -> ExplainResponse:
        """Run fraud analysis and return a detailed SHAP + LLM explanation."""
        orchestrator = get_orchestrator()
        start = time.perf_counter()

        txn_dict = derive_time_features(transaction.model_dump())
        result = await run_in_threadpool(orchestrator.analyze, txn_dict)

        score = float(result.get("final_score", 0.5))
        details = result.get("explanation_details") or {}

        top_features = []
        for f in details.get("top_features", []):
            top_features.append(FeatureContributionResponse(
                feature_name=f.get("feature_name", ""),
                raw_name=f.get("raw_name", ""),
                feature_value=f.get("feature_value"),
                shap_value=f.get("shap_value", 0.0),
                direction=f.get("direction", "unknown"),
            ))

        duration_s = time.perf_counter() - start
        _observe_prediction_metrics(
            endpoint="explain",
            decision=str(result.get("final_decision", "REVIEW")),
            duration_s=duration_s,
        )

        return ExplainResponse(
            transaction_id=transaction.TransactionID,
            fraud_probability=max(0.0, min(score, 1.0)),
            decision=str(result.get("final_decision", "REVIEW")),
            risk_level=get_risk_level(score),
            summary=details.get("summary", f"Score: {score:.1%}"),
            natural_language_explanation=details.get(
                "natural_language_explanation",
                str(result.get("explanation", "Analysis complete")),
            ),
            top_risk_factors=details.get("top_risk_factors", []),
            top_features=top_features,
            shap_available=details.get("shap_available", False),
            llm_used=details.get("llm_used", False),
            confidence_factors=details.get("confidence_factors", []),
            recommended_action=details.get(
                "recommended_action", "REVIEW — Queue for manual review"
            ),
            agent_scores={
                "vibe_checker": float(result.get("vibe_score", 0.5)),
                "agent_ensemble": float(result.get("agent_ensemble_score", 0.5)),
                "era_tracker": float(result.get("era_score", 0.5)),
                "og_check": float(result.get("og_score", 0.5)),
            },
            rule_violations=list(result.get("og_violations", [])),
            processing_time_ms=duration_s * 1000,
            timestamp=_utc_now_iso(),
        )

    @app.get("/metrics", tags=["Monitoring"])
    async def metrics() -> Response:
        if not PROMETHEUS_AVAILABLE:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "prometheus-client not installed",
                    "tip": "Install prometheus-client to expose metrics output",
                },
            )
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.api_gateway.main:app", host=SETTINGS.host, port=SETTINGS.port, reload=False)
