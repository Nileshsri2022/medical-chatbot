"""
Configuration management for Medical RAG Chatbot
Loads settings from environment variables with sensible defaults
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "info"

    # LLM Backend
    original_llm_url: str = "http://localhost:8001"
    request_timeout: float = 30.0

    # HTTP Client
    http_max_keepalive: int = 20
    http_max_connections: int = 100
    http_connect_timeout: float = 10.0

    # Rate Limiting
    rate_limit_requests_per_minute: int = 30

    # CORS
    allowed_origins: str = "http://localhost:3000"

    # RAG Engine
    max_conversation_history: int = 10
    cache_max_size: int = 100

    # Medical AI
    default_max_tokens: int = 200
    default_temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv("RAG_HOST", "0.0.0.0"),
            port=int(os.getenv("RAG_SERVER_PORT", "8002")),
            log_level=os.getenv("LOG_LEVEL", "info"),
            original_llm_url=os.getenv("ORIGINAL_LLM_URL", "http://localhost:8001"),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "30.0")),
            http_max_keepalive=int(os.getenv("HTTP_MAX_KEEPALIVE", "20")),
            http_max_connections=int(os.getenv("HTTP_MAX_CONNECTIONS", "100")),
            http_connect_timeout=float(os.getenv("HTTP_CONNECT_TIMEOUT", "10.0")),
            rate_limit_requests_per_minute=int(
                os.getenv("RATE_LIMIT_PER_MINUTE", "30")
            ),
            allowed_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000"),
            max_conversation_history=int(os.getenv("MAX_CONVERSATION_HISTORY", "10")),
            cache_max_size=int(os.getenv("CACHE_MAX_SIZE", "100")),
            default_max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "200")),
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
        )


# Global config instance
config = ServerConfig.from_env()
