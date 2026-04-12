"""
Monitoring Module for Medical RAG Chatbot
Includes structured logging, metrics, and request tracing
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


try:
    from loguru import logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging

    logger = logging.getLogger("medical_rag")


try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class StructuredLogger:
    """Structured logging for the medical chatbot"""

    @staticmethod
    def setup_logging(
        log_file: str = "medical_rag.log",
        rotation: str = "10 MB",
        retention: str = "7 days",
        level: str = "INFO",
    ):
        """Configure structured logging"""
        if LOGURU_AVAILABLE:
            logger.remove()
            logger.add(
                log_file,
                rotation=rotation,
                retention=retention,
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            )
            logger.add(
                "stdout",
                format="{time:HH:mm:ss} | {level: <8} | {message}",
                level=level,
            )

    @staticmethod
    def log_request(
        request_id: str,
        method: str,
        path: str,
        client: str,
        duration: float,
        status: int,
    ):
        """Log HTTP request"""
        logger.info(
            f"request_id={request_id} method={method} path={path} "
            f"client={client} duration={duration:.3f}s status={status}"
        )

    @staticmethod
    def log_chat_request(
        request_id: str,
        session_id: str,
        message_length: int,
        symptoms_detected: int,
        confidence: float,
        duration: float,
        success: bool,
    ):
        """Log chat interaction"""
        logger.info(
            f"request_id={request_id} event=chat_interaction "
            f"session_id={session_id} message_length={message_length} "
            f"symptoms_detected={symptoms_detected} confidence={confidence:.2f} "
            f"duration={duration:.3f}s success={success}"
        )

    @staticmethod
    def log_llm_call(
        request_id: str,
        provider: str,
        model: str,
        duration: float,
        tokens: int,
        success: bool,
        error: Optional[str] = None,
    ):
        """Log LLM API call"""
        status = "success" if success else "error"
        base = f"request_id={request_id} event=llm_call provider={provider} model={model} duration={duration:.3f}s tokens={tokens} status={status}"

        if error:
            logger.warning(f"{base} error={error}")
        else:
            logger.info(base)

    @staticmethod
    def log_error(
        request_id: str, error_type: str, message: str, details: Optional[dict] = None
    ):
        """Log error with context"""
        details_str = f" details={details}" if details else ""
        logger.error(
            f"request_id={request_id} error_type={error_type} message={message}{details_str}"
        )

    @staticmethod
    def log_medical_warning(request_id: str, warning_type: str, details: dict):
        """Log medical-related warnings"""
        logger.warning(
            f"request_id={request_id} event=medical_warning "
            f"warning_type={warning_type} details={details}"
        )


class PrometheusMetrics:
    """Prometheus metrics for the medical chatbot"""

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            return

        self.requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )

        self.request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.chat_interactions_total = Counter(
            "chat_interactions_total",
            "Total chat interactions",
            ["status"],
        )

        self.symptoms_detected = Histogram(
            "symptoms_detected",
            "Number of symptoms detected per interaction",
            buckets=[0, 1, 2, 3, 5, 10],
        )

        self.llm_calls_total = Counter(
            "llm_calls_total",
            "Total LLM API calls",
            ["provider", "status"],
        )

        self.llm_call_duration = Histogram(
            "llm_call_duration_seconds",
            "LLM API call duration",
            ["provider"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )

        self.active_sessions = Gauge(
            "active_sessions",
            "Number of active sessions",
        )

        self.rag_cache_hits = Counter(
            "rag_cache_hits_total",
            "Total RAG cache hits",
        )

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        self.requests_total.labels(
            method=method, endpoint=endpoint, status=status
        ).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_chat(self, success: bool, symptoms_count: int):
        """Record chat interaction"""
        if not PROMETHEUS_AVAILABLE:
            return
        status = "success" if success else "error"
        self.chat_interactions_total.labels(status=status).inc()
        self.symptoms_detected.observe(symptoms_count)

    def record_llm_call(self, provider: str, duration: float, success: bool):
        """Record LLM API call"""
        if not PROMETHEUS_AVAILABLE:
            return
        status = "success" if success else "error"
        self.llm_calls_total.labels(provider=provider, status=status).inc()
        self.llm_call_duration.labels(provider=provider).observe(duration)

    def set_active_sessions(self, count: int):
        """Set active sessions gauge"""
        if not PROMETHEUS_AVAILABLE:
            return
        self.active_sessions.set(count)

    def inc_cache_hit(self):
        """Increment cache hit counter"""
        if not PROMETHEUS_AVAILABLE:
            return
        self.rag_cache_hits.inc()


_metrics = PrometheusMetrics()


def get_metrics() -> PrometheusMetrics:
    return _metrics


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID and trace requests"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.time()
        response: Response = await call_next(request)
        duration = time.time() - start_time

        response.headers["X-Request-ID"] = request_id

        method = request.method
        endpoint = request.url.path
        status = response.status_code

        StructuredLogger.log_request(
            request_id,
            method,
            endpoint,
            request.client.host if request.client else "unknown",
            duration,
            status,
        )

        _metrics.record_request(method, endpoint, status, duration)

        return response


def trace_llm_call(provider: str, model: str):
    """Decorator to trace LLM calls"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error = None
            tokens = 0

            try:
                result = await func(*args, **kwargs)
                success = True
                if isinstance(result, dict):
                    tokens = result.get("tokens", 0)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                request_id = getattr(kwargs.get("request", {}), "state", {}).get(
                    "request_id", "unknown"
                )

                StructuredLogger.log_llm_call(
                    request_id, provider, model, duration, tokens, success, error
                )
                _metrics.record_llm_call(provider, duration, success)

        return wrapper

    return decorator


def get_metrics_endpoint():
    """Generate Prometheus metrics endpoint"""

    async def metrics():
        if PROMETHEUS_AVAILABLE:
            from fastapi import Response

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
        return {"error": "Prometheus not available"}

    return metrics
