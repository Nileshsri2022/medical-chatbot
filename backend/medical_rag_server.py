#!/usr/bin/env python3
"""
Medical RAG Server for Healthcare Chatbot
Provides context-aware medical conversation with memory and entity extraction

This server acts as middleware between the frontend and the original LLM backend,
enriching conversations with medical context and maintaining conversation memory.
"""

import os
import sys
import time
from collections import defaultdict
from threading import Lock

# Fix Windows console encoding for emoji/unicode characters
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import httpx
import json
import datetime
import logging
import asyncio
from typing import Dict, List, Optional

# Add the current directory to Python path to import our RAG engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from medical_rag_engine import MedicalRAGEnrichmentEngine
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from collections import defaultdict
from threading import Lock
from functools import wraps


# Rate limiter class
class RateLimiter:
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


rate_limiter = RateLimiter()

# Shared httpx client for connection pooling
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(config.request_timeout, connect=config.http_connect_timeout),
    limits=httpx.Limits(
        max_keepalive_connections=config.http_max_keepalive,
        max_connections=config.http_max_connections,
    ),
    follow_redirects=True,
)

# Separate client for streaming (no timeout)
streaming_client = httpx.AsyncClient(
    timeout=None, limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
)


def get_http_client() -> httpx.AsyncClient:
    """Get the shared HTTP client for connection pooling"""
    return http_client


def get_streaming_client() -> httpx.AsyncClient:
    """Get the streaming HTTP client"""
    return streaming_client


# FastAPI application with professional API design
app = FastAPI(
    title="Medical RAG API",
    description="Advanced medical conversation AI with context awareness and memory. Provides RAG-enhanced chat with medical entity recognition and symptom analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Custom OpenAPI schema for better API documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Medical RAG API",
        version="1.0.0",
        description="Advanced medical conversation AI with context awareness and memory",
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }
    openapi_schema["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# CORS middleware - configurable via environment
ALLOWED_ORIGINS = config.allowed_origins.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class EnhancedChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    session_id: str = Field(..., description="Unique session identifier")
    max_tokens: int = Field(
        config.default_max_tokens,
        description="Maximum tokens for response",
        ge=50,
        le=1000,
    )
    temperature: float = Field(
        config.default_temperature, description="LLM temperature", ge=0.0, le=2.0
    )


class EnhancedChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    conversation_context: Dict = Field(
        ..., description="Conversation context and memory"
    )
    extracted_entities: Dict = Field(
        ..., description="Medical entities extracted from input"
    )
    symptoms_detected: List[Dict] = Field(
        ..., description="Symptoms detected and analyzed"
    )
    confidence_score: float = Field(..., description="Overall confidence in analysis")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")
    rag_metadata: Dict = Field(..., description="RAG processing metadata")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    rag_engine_status: str
    original_llm_status: str
    active_sessions: int


class ConversationHistoryResponse(BaseModel):
    session_id: str
    conversation_context: Dict
    total_interactions: int
    session_start_time: str


# Global RAG engine instance
rag_engine = MedicalRAGEnrichmentEngine()

# Session statistics
session_stats = {
    "total_sessions": 0,
    "active_sessions": set(),
    "total_interactions": 0,
    "server_start_time": datetime.datetime.now().isoformat(),
}


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG server"""
    logger.info("🚀 Starting Medical RAG Enhancement Server")
    logger.info(
        f"📊 RAG Engine initialized with {rag_engine.symptom_extractor.symptom_count} symptom patterns"
    )
    logger.info(f"🔗 Original LLM backend: {config.original_llm_url}")
    logger.info(f"🌐 Server will run on port {config.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    await http_client.aclose()
    await streaming_client.aclose()
    logger.info("🔄 HTTP clients closed")


@app.get("/api/v1/", response_model=Dict)
@app.get("/api/v1/health", response_model=HealthResponse)
@app.post("/api/v1/chat", response_model=EnhancedChatResponse)
@app.post("/api/v1/chat/stream")
@app.get(
    "/api/v1/conversation-history/{session_id}",
    response_model=ConversationHistoryResponse,
)
@app.delete("/api/v1/conversation/{session_id}")
@app.get("/api/v1/session-stats")
@app.get("/api/v1/debug/session/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to inspect session details"""

    if session_id not in rag_engine.conversation_memory.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = rag_engine.conversation_memory.sessions[session_id]

    return {
        "session_id": session_id,
        "debug_info": {
            "conversation_history_length": len(session_data["conversation_history"]),
            "accumulated_symptoms": list(session_data["accumulated_symptoms"]),
            "accumulated_conditions": list(session_data["accumulated_conditions"]),
            "conversation_state": session_data["conversation_state"].value,
            "urgency_level": session_data["urgency_level"],
            "last_interaction": session_data["conversation_history"][-1]
            if session_data["conversation_history"]
            else None,
        },
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.datetime.now().isoformat(),
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.datetime.now().isoformat(),
        },
    )


# Development utilities
@app.get("/dev/test-rag")
async def test_rag_engine():
    """Development endpoint to test RAG engine functionality"""

    test_inputs = [
        "I have chest pain",
        "It started an hour ago and I feel nauseous",
        "The pain is radiating to my left arm",
    ]

    session_id = "test_session_" + str(datetime.datetime.now().timestamp())
    results = []

    for i, test_input in enumerate(test_inputs):
        result = rag_engine.process_user_input(test_input, session_id)

        # Simulate storing the interaction
        rag_engine.conversation_memory.add_interaction(
            session_id=session_id,
            user_input=test_input,
            extracted_info={
                "entities": result["entities"],
                "symptoms": result["symptoms"],
            },
            ai_response=f"Test response {i + 1}",
            confidence_score=result["confidence_score"],
        )

        results.append(
            {
                "input": test_input,
                "symptoms_detected": len(result["symptoms"]),
                "entities_detected": sum(
                    len(v) if isinstance(v, list) else 0
                    for v in result["entities"].values()
                ),
                "confidence_score": result["confidence_score"],
                "conversation_state": result["conversation_context"].get(
                    "conversation_state", "unknown"
                ),
            }
        )

    return {
        "test_session_id": session_id,
        "results": results,
        "final_context": rag_engine.conversation_memory.get_context(session_id),
    }


if __name__ == "__main__":
    import uvicorn

    print("🏥 Medical RAG Enhancement Server")
    print("=" * 50)
    print(f"🚀 Starting server on port {config.port}")
    print(f"🔗 Original LLM backend: {config.original_llm_url}")
    print(f"🧠 RAG engine initialized")
    print(f"📊 Medical entity patterns loaded")
    print("=" * 50)

    uvicorn.run(
        app, host="0.0.0.0", port=config.port, log_level="info", access_log=True
    )
