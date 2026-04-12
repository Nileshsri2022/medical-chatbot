"""
Chat API Routes
Modular API endpoints for the medical chatbot
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging

from .dependencies import get_http_client, get_client_id, check_rate_limit, require_auth
from ..config import config
from ..rag import MedicalRAGEnrichmentEngine
from ..core.security import InputSanitizer, MedicalDisclaimer
from ..core.reliability import get_llm_health_checker, get_circuit_breaker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

_rag_engine = MedicalRAGEnrichmentEngine()


class ChatRequestModel(BaseModel):
    message: str = Field(..., description="User's message")
    session_id: str = Field(..., description="Unique session identifier")
    max_tokens: int = Field(
        config.default_max_tokens, description="Maximum tokens", ge=50, le=1000
    )
    temperature: float = Field(
        config.default_temperature, description="LLM temperature", ge=0.0, le=2.0
    )


@router.get("/")
async def root():
    """Root endpoint"""
    return {"service": "Medical RAG API", "version": "1.0.0", "status": "running"}


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    client_id = request.client.host if request.client else "unknown"

    llm_health = await get_llm_health_checker().check_with_circuit()
    circuit_status = get_circuit_breaker().get_status()

    return {
        "status": "healthy" if llm_health.get("status") != "unhealthy" else "degraded",
        "timestamp": "",
        "version": "1.0.0",
        "rag_engine_status": "ready",
        "original_llm_status": llm_health.get("status", "unknown"),
        "llm_response_time": llm_health.get("response_time"),
        "circuit_breaker": circuit_status,
        "active_sessions": _rag_engine.conversation_memory.active_sessions_count,
    }


class ChatRequest:
    """Chat request model"""

    message: str
    session_id: str
    max_tokens: int = config.default_max_tokens
    temperature: float = config.default_temperature


@router.post("/chat")
async def chat(
    chat_request: ChatRequestModel,
    request: Request = None,
    _auth: bool = Depends(require_auth),
):
    """Enhanced chat with RAG"""
    client_id = request.client.host if request and request.client else "unknown"

    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        validated_message = InputSanitizer.validate_message(chat_request.message)

        result = _rag_engine.process_user_input(
            validated_message, chat_request.session_id
        )

        response = result.get("enriched_prompt", "")

        response = MedicalDisclaimer.add_disclaimer(
            response,
            symptoms=result.get("symptoms", []),
            entities=result.get("entities", {}),
        )

        return {
            "response": response,
            "conversation_context": result.get("conversation_context", {}),
            "extracted_entities": result.get("entities", {}),
            "symptoms_detected": result.get("symptoms", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "session_id": chat_request.session_id,
            "timestamp": "",
            "processing_time": 0.0,
            "rag_metadata": {
                "from_cache": result.get("from_cache", False),
                "urgency": result.get("context", {}).get("medical_urgency", "low"),
            },
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation-history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    context = _rag_engine.get_context(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "conversation_context": context,
        "total_interactions": context.get("total_interactions", 0),
        "session_start_time": context.get("session_start_time", ""),
    }


@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """Clear a conversation session"""
    _rag_engine.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/session-stats")
async def session_stats():
    """Get session statistics"""
    return {
        "active_sessions": _rag_engine.conversation_memory.active_sessions_count,
        "cache_size": len(_rag_engine._cache) if hasattr(_rag_engine, "_cache") else 0,
    }


@router.get("/debug/session/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to inspect session details"""
    if session_id not in _rag_engine.conversation_memory.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = _rag_engine.conversation_memory.sessions[session_id]

    return {
        "session_id": session_id,
        "debug_info": {
            "conversation_history_length": len(
                session_data.get("conversation_history", [])
            ),
            "accumulated_symptoms": list(
                session_data.get("accumulated_symptoms", set())
            ),
            "accumulated_conditions": list(
                session_data.get("accumulated_conditions", set())
            ),
            "conversation_state": session_data.get("conversation_state", {}).value,
            "urgency_level": session_data.get("urgency_level", "low"),
            "last_interaction": session_data.get("conversation_history", [])[-1]
            if session_data.get("conversation_history")
            else None,
        },
    }


@router.get("/circuit-status")
async def circuit_status():
    """Get circuit breaker status"""
    return get_circuit_breaker().get_status()


@router.post("/circuit-reset")
async def reset_circuit():
    """Reset circuit breaker to closed state"""
    cb = get_circuit_breaker()
    cb._reset()
    return {"status": "reset", "circuit_state": cb.state.value}


@router.get("/test-rag")
async def test_rag():
    """Test RAG engine functionality"""
    test_inputs = [
        "I have chest pain",
        "It started an hour ago and I feel nauseous",
        "The pain is radiating to my left arm",
    ]

    import datetime

    session_id = "test_session_" + str(datetime.datetime.now().timestamp())
    results = []

    for i, test_input in enumerate(test_inputs):
        result = _rag_engine.process_user_input(test_input, session_id)

        _rag_engine.add_interaction(
            session_id=session_id,
            user_input=test_input,
            extracted_info={
                "entities": result.get("entities", {}),
                "symptoms": result.get("symptoms", []),
            },
            ai_response=f"Test response {i + 1}",
            confidence_score=result.get("confidence_score", 0.0),
        )

        results.append(
            {
                "input": test_input,
                "symptoms_detected": len(result.get("symptoms", [])),
                "entities_detected": sum(
                    len(v) if isinstance(v, list) else 0
                    for v in result.get("entities", {}).values()
                ),
                "confidence_score": result.get("confidence_score", 0.0),
                "conversation_state": result.get("conversation_context", {}).get(
                    "conversation_state", "unknown"
                ),
            }
        )

    return {
        "test_session_id": session_id,
        "results": results,
        "final_context": _rag_engine.get_context(session_id),
    }
