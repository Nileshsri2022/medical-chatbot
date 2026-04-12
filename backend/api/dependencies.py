"""
Dependency injection for FastAPI
Provides reusable dependencies for routes including rate limiting and HTTP clients
"""

import time
from typing import Optional
from functools import lru_cache
from collections import defaultdict
from threading import Lock

import httpx
from fastapi import Request, HTTPException

from ..config import config


class RateLimiter:
    """Thread-safe rate limiter"""

    def __init__(self, requests_per_minute: int = 0):
        self.requests_per_minute = (
            requests_per_minute
            if requests_per_minute > 0
            else config.rate_limit_requests_per_minute
        )
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            self.requests[client_id] = [
                t for t in self.requests[client_id] if t > minute_ago
            ]
            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False
            self.requests[client_id].append(now)
            return True


_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


def check_rate_limit(client_id: str) -> bool:
    return _rate_limiter.is_allowed(client_id)


class HTTPClients:
    """Manage shared HTTP clients for connection pooling"""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._streaming_client: Optional[httpx.AsyncClient] = None

    def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    config.request_timeout, connect=config.http_connect_timeout
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=config.http_max_keepalive,
                    max_connections=config.http_max_connections,
                ),
                follow_redirects=True,
            )
        return self._client

    def get_streaming_client(self) -> httpx.AsyncClient:
        if self._streaming_client is None:
            self._streaming_client = httpx.AsyncClient(
                timeout=None,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._streaming_client

    async def close(self):
        if self._client:
            await self._client.aclose()
        if self._streaming_client:
            await self._streaming_client.aclose()


_http_clients = HTTPClients()


def get_http_client() -> httpx.AsyncClient:
    """Get shared HTTP client for connection pooling"""
    return _http_clients.get_client()


def get_streaming_client() -> httpx.AsyncClient:
    """Get streaming HTTP client"""
    return _http_clients.get_streaming_client()


def get_http_clients() -> HTTPClients:
    """Get HTTP clients manager"""
    return _http_clients


@lru_cache()
def get_llm_url() -> str:
    """Get LLM backend URL"""
    return config.original_llm_url


@lru_cache()
def get_timeout() -> float:
    """Get request timeout"""
    return config.request_timeout


async def get_client_id(request: Request) -> str:
    """Extract client ID for rate limiting"""
    return request.client.host if request.client else "unknown"


async def verify_rate_limit(request: Request) -> None:
    """Verify rate limit for request"""
    client_id = await get_client_id(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Please try again later."
        )
