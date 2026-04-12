"""
Error response schemas for consistent API error handling
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes"""

    INVALID_INPUT = "INVALID_INPUT"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"


class ErrorResponse(BaseModel):
    """Standard error response format"""

    error: str = Field(..., description="Error message")
    error_code: ErrorCode = Field(..., description="Machine-readable error code")
    request_id: str = Field(..., description="Request ID for tracing")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Error timestamp",
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )


class ValidationErrorResponse(ErrorResponse):
    """Validation error response"""

    error_code: ErrorCode = ErrorCode.INVALID_INPUT
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Validation error details"
    )


class RateLimitResponse(ErrorResponse):
    """Rate limit exceeded response"""

    error_code: ErrorCode = ErrorCode.RATE_LIMITED
    retry_after: int = Field(..., description="Seconds until retry is allowed")


class ServiceUnavailableResponse(ErrorResponse):
    """Service unavailable response"""

    error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE
    retry_after: Optional[int] = Field(None, description="Suggested retry time")


def create_error_response(
    message: str,
    error_code: ErrorCode,
    request_id: str,
    details: Optional[Dict[str, Any]] = None,
) -> ErrorResponse:
    """Factory function to create error responses"""
    return ErrorResponse(
        error=message, error_code=error_code, request_id=request_id, details=details
    )
