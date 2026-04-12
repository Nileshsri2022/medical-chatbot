"""
API Routes - Chat endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import logging

from ..config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_request_id(request: Request) -> str:
    """Extract or generate request ID for tracing"""
    return request.headers.get("X-Request-ID", "unknown")


def get_client_ip(request: Request) -> str:
    """Get client IP for rate limiting"""
    return request.client.host if request.client else "unknown"
