"""
Dependency injection for FastAPI
Provides reusable dependencies for routes
"""

from typing import Optional
from functools import lru_cache

from ..config import config


@lru_cache()
def get_config():
    """Get cached configuration"""
    return config


def get_rate_limit() -> int:
    """Get rate limit from config"""
    return config.rate_limit_requests_per_minute


def get_llm_url() -> str:
    """Get LLM backend URL"""
    return config.original_llm_url


def get_timeout() -> float:
    """Get request timeout"""
    return config.request_timeout
