"""
Reliability Module for Medical Chatbot
Includes circuit breaker, health checks, and error handling
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from threading import Lock

import httpx
from fastapi import HTTPException

from ..config import config

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker pattern for LLM API calls"""

    def __init__(self, name: str = "llm", config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self.lock = Lock()

    def can_execute(self) -> bool:
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' transitioned to HALF_OPEN"
                    )
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                return self.half_open_calls < self.config.half_open_max_calls

            return False

    def record_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
            else:
                self.failure_count = 0
                self.success_count = 0

    def record_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self._trip()
            elif self.failure_count >= self.config.failure_threshold:
                self._trip()

    def _trip(self):
        self.state = CircuitState.OPEN
        logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN")

    def _reset(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure": self.last_failure_time,
            }


_llm_circuit_breaker = CircuitBreaker("llm")


def get_circuit_breaker() -> CircuitBreaker:
    return _llm_circuit_breaker


class LLMHealthChecker:
    """Check LLM backend health"""

    def __init__(self):
        self.base_url = config.original_llm_url
        self.timeout = 5.0

    async def check_health(self) -> Dict[str, Any]:
        """Check if LLM backend is reachable"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/health", follow_redirects=True
                )
                response_time = time.time() - start_time

                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "response_time": round(response_time, 3),
                        "status_code": response.status_code,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "response_time": round(response_time, 3),
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}",
                    }
        except httpx.TimeoutException:
            return {
                "status": "unhealthy",
                "response_time": time.time() - start_time,
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "response_time": time.time() - start_time,
                "error": str(e),
            }

    async def check_with_circuit(self) -> Dict[str, Any]:
        """Check health respecting circuit breaker state"""
        if not _llm_circuit_breaker.can_execute():
            return {
                "status": "unhealthy",
                "reason": "circuit_breaker_open",
                "circuit_state": _llm_circuit_breaker.state.value,
            }

        return await self.check_health()


_llm_health_checker = LLMHealthChecker()


def get_llm_health_checker() -> LLMHealthChecker:
    return _llm_health_checker


class APIError(Exception):
    """Base API error with structured response"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Dict[str, Any] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "details": self.details,
            "timestamp": time.time(),
        }


class LLMConnectionError(APIError):
    """LLM backend connection failed"""

    def __init__(self, message: str = "LLM backend unavailable"):
        super().__init__(
            message=message,
            status_code=503,
            error_code="LLM_UNAVAILABLE",
        )


class RateLimitError(APIError):
    """Rate limit exceeded"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMITED",
        )


class ValidationError(APIError):
    """Input validation failed"""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
        )


async def call_with_circuit_break(
    func,
    *args,
    circuit_breaker: CircuitBreaker = None,
    **kwargs,
):
    """Execute function with circuit breaker protection"""
    cb = circuit_breaker or _llm_circuit_breaker

    if not cb.can_execute():
        raise LLMConnectionError("LLM service temporarily unavailable")

    try:
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        cb.record_success()
        return result
    except Exception as e:
        cb.record_failure()
        raise
