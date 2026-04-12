"""
Security middleware for Medical RAG Chatbot
Implements HIPAA-compliant security measures
"""

import re
import uuid
import logging
from typing import Optional, Callable
from functools import wraps
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration settings"""

    # API Key for authentication (set via environment variable)
    API_KEY: Optional[str] = None

    # Request validation
    MAX_REQUEST_SIZE = 10_000  # 10KB max request
    MAX_MESSAGE_LENGTH = 5_000  # 5K characters max

    # Input sanitization patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|;|\/\*|\*\/|@@|@)",
        r"(\bOR\b.*=.*\bOR\b)",
        r"(\bUNION\b.*\bSELECT\b)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
    ]

    # Medical disclaimer
    MEDICAL_DISCLAIMER = (
        "This is an AI-powered medical information tool and is NOT a substitute "
        "for professional medical advice, diagnosis, or treatment. Always seek "
        "the advice of your physician or other qualified health provider with any "
        "questions you may have regarding a medical condition."
    )


class InputSanitizer:
    """Sanitize user input to prevent injection attacks"""

    @staticmethod
    def sanitize(message: str) -> str:
        """Sanitize user message"""
        if not message:
            return ""

        # Limit length
        message = message[: SecurityConfig.MAX_MESSAGE_LENGTH]

        # Remove null bytes
        message = message.replace("\x00", "")

        # Normalize whitespace
        message = re.sub(r"\s+", " ", message).strip()

        return message

    @staticmethod
    def check_sql_injection(text: str) -> bool:
        """Check for SQL injection attempts"""
        text_lower = text.lower()
        for pattern in SecurityConfig.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {text[:50]}")
                return True
        return False

    @staticmethod
    def check_xss(text: str) -> bool:
        """Check for XSS attempts"""
        for pattern in SecurityConfig.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {text[:50]}")
                return True
        return False

    @staticmethod
    def validate_message(message: str) -> str:
        """Validate and sanitize message"""
        if not message:
            raise HTTPException(status_code=400, detail="Empty message not allowed")

        if len(message) > SecurityConfig.MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message exceeds maximum length of {SecurityConfig.MAX_MESSAGE_LENGTH}",
            )

        # Check for malicious patterns
        if InputSanitizer.check_sql_injection(message):
            raise HTTPException(status_code=400, detail="Invalid input detected")

        if InputSanitizer.check_xss(message):
            raise HTTPException(status_code=400, detail="Invalid input detected")

        return InputSanitizer.sanitize(message)


class AuditLogger:
    """Audit logging for HIPAA compliance"""

    def __init__(self):
        self.log_file = "audit.log"

    def log_request(
        self,
        request_id: str,
        session_id: str,
        user_input: str,
        endpoint: str,
        status_code: int,
    ):
        """Log request for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "session_id": session_id,
            "endpoint": endpoint,
            "status_code": status_code,
            "input_length": len(user_input),
        }

        # Log to file (in production, use secure logging service)
        logger.info(f"AUDIT: {log_entry}")

        # Don't log actual PHI content
        logger.debug(f"Request {request_id} completed with status {status_code}")

    def log_medical_interaction(
        self,
        request_id: str,
        session_id: str,
        symptoms_detected: list,
        confidence_score: float,
    ):
        """Log medical interaction (no PHI)"""
        logger.info(
            f"MEDICAL: request={request_id}, session={session_id}, "
            f"symptoms_count={len(symptoms_detected)}, confidence={confidence_score}"
        )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> bool:
    """Verify API key if configured"""
    if not SecurityConfig.API_KEY:
        return True  # No API key configured, allow all

    if api_key != SecurityConfig.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, "request_id", "unknown")


def medical_warning_decorator(func: Callable) -> Callable:
    """Decorator to add medical disclaimer to responses"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        response = await func(*args, **kwargs)

        # Add disclaimer to response if applicable
        if hasattr(response, "response"):
            response.response += f"\n\n⚠️ {SecurityConfig.MEDICAL_DISCLAIMER}"

        return response

    return wrapper
