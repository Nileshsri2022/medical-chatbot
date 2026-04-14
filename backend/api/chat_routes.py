"""
Chat API Routes
Modular API endpoints for the medical chatbot
"""

from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import logging
import uuid
import time
import json
import asyncio

import httpx

from .dependencies import get_http_client, get_client_id, check_rate_limit, require_auth
from ..config import config
from ..rag import MedicalRAGEnrichmentEngine
from ..core.security import (
    InputSanitizer,
    MedicalDisclaimer,
    require_jwt_auth,
    optional_jwt_auth,
    get_jwt_auth,
)
from ..core.reliability import get_llm_health_checker, get_circuit_breaker
from ..core.monitoring import get_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

_rag_engine = MedicalRAGEnrichmentEngine()


def _format_llm_diagnosis_response(llm_payload: Dict) -> str:
    """Convert structured diagnosis JSON into user-friendly medical guidance text."""
    symptoms = llm_payload.get("symptoms", []) if isinstance(llm_payload, dict) else []
    illnesses = llm_payload.get("illnesses", []) if isinstance(llm_payload, dict) else []

    lines: List[str] = [
        "Thanks for sharing your symptoms. Based on your details, here is a preliminary analysis:",
        "",
    ]

    if symptoms:
        lines.append("Possible symptoms identified: " + ", ".join(symptoms[:8]))

    if illnesses:
        lines.append("")
        lines.append("Possible conditions to discuss with a clinician:")
        for idx, illness in enumerate(illnesses[:3], start=1):
            if not isinstance(illness, dict):
                continue
            name = illness.get("name", "Unspecified condition")
            illness_cov = illness.get("illness_coverage", 0)
            condition_cov = illness.get("condition_coverage", 0)
            lines.append(
                f"{idx}. {name} (match quality: illness {illness_cov}%, symptom fit {condition_cov}%)"
            )
    else:
        lines.append("No high-confidence condition match was returned by the model.")

    lines.append("")
    lines.append(
        "If symptoms are severe, sudden, worsening, or include chest pain, breathing trouble, severe headache, confusion, or fainting, seek emergency care immediately."
    )
    return "\n".join(lines)


class ChatRequestModel(BaseModel):
    message: str = Field(
        ..., description="User's message", min_length=1, max_length=5000
    )
    session_id: str = Field(
        ..., description="Unique session identifier", min_length=1, max_length=100
    )
    max_tokens: int = Field(
        config.default_max_tokens, description="Maximum tokens", ge=50, le=1000
    )
    temperature: float = Field(
        config.default_temperature, description="LLM temperature", ge=0.0, le=2.0
    )

    @validator("message", "session_id")
    def validate_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Cannot be empty")
        return v.strip()


class ChatResponseModel(BaseModel):
    response: str
    conversation_context: Dict
    extracted_entities: Dict
    symptoms_detected: List[Dict]
    confidence_score: float
    session_id: str
    timestamp: str
    processing_time: float
    rag_metadata: Dict


class HealthResponseModel(BaseModel):
    status: str
    timestamp: str
    version: str
    rag_engine_status: str
    original_llm_status: str
    llm_response_time: Optional[float]
    circuit_breaker: Dict
    active_sessions: int


class ConversationHistoryResponseModel(BaseModel):
    session_id: str
    conversation_context: Dict
    total_interactions: int
    session_start_time: str


class TokenRequestModel(BaseModel):
    user_id: str = Field(
        ..., description="User identifier", min_length=1, max_length=100
    )
    expires_minutes: Optional[int] = Field(
        60, description="Token expiration in minutes", ge=1, le=1440
    )


class TokenResponseModel(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


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


@router.post("/auth/token")
async def create_token(token_request: TokenRequestModel):
    """Generate JWT access token"""
    jwt_auth = get_jwt_auth()

    token_data = {
        "sub": token_request.user_id,
        "user_id": token_request.user_id,
        "iat": datetime.utcnow(),
    }

    access_token = jwt_auth.create_token(token_data)

    return TokenResponseModel(
        access_token=access_token,
        token_type="bearer",
        expires_in=token_request.expires_minutes * 60,
    )


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

    logger.info(
        f"[CHAT] Non-stream endpoint hit: /api/v1/chat session={chat_request.session_id} client={client_id}"
    )

    started = time.perf_counter()

    try:
        validated_message = InputSanitizer.validate_message(chat_request.message)

        result = _rag_engine.process_user_input(
            validated_message, chat_request.session_id
        )

        llm_payload = {
            "description": result.get("enriched_prompt", validated_message),
            "max_tokens": chat_request.max_tokens,
            "temperature": chat_request.temperature,
        }

        circuit_breaker = get_circuit_breaker()
        if not circuit_breaker.can_execute():
            raise HTTPException(status_code=503, detail="LLM service temporarily unavailable")

        llm_json: Dict = {}
        try:
            llm_response = await get_http_client().post(
                f"{config.original_llm_url}/diagnose",
                json=llm_payload,
                headers={"Content-Type": "application/json"},
            )
            llm_response.raise_for_status()
            llm_json = llm_response.json()
            circuit_breaker.record_success()
        except httpx.HTTPStatusError as exc:
            circuit_breaker.record_failure()
            logger.error(f"LLM backend returned HTTP {exc.response.status_code}: {exc}")
            raise HTTPException(status_code=502, detail="LLM backend returned an error")
        except httpx.HTTPError as exc:
            circuit_breaker.record_failure()
            logger.error(f"LLM backend request failed: {exc}")
            raise HTTPException(status_code=502, detail="Failed to reach LLM backend")

        response = _format_llm_diagnosis_response(llm_json)

        _rag_engine.add_interaction(
            session_id=chat_request.session_id,
            user_input=validated_message,
            extracted_info={
                "entities": result.get("entities", {}),
                "symptoms": result.get("symptoms", []),
                "llm_diagnosis": llm_json,
            },
            ai_response=response,
            confidence_score=result.get("confidence_score", 0.0),
        )

        processing_time = round(time.perf_counter() - started, 3)

        response = MedicalDisclaimer.add_disclaimer(
            response,
            symptoms=result.get("symptoms", []),
            entities=result.get("entities", {}),
        )

        get_metrics().record_chat(
            success=True, symptoms_count=len(result.get("symptoms", []))
        )

        return {
            "response": response,
            "conversation_context": result.get("conversation_context", {}),
            "extracted_entities": result.get("entities", {}),
            "symptoms_detected": result.get("symptoms", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "session_id": chat_request.session_id,
            "timestamp": "",
            "processing_time": processing_time,
            "rag_metadata": {
                "from_cache": result.get("from_cache", False),
                "urgency": result.get("context", {}).get("medical_urgency", "low"),
            },
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    chat_request: ChatRequestModel,
    request: Request = None,
    _auth: bool = Depends(require_auth),
):
    """Enhanced chat with RAG and streaming response"""
    client_id = request.client.host if request and request.client else "unknown"

    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    logger.info(
        f"[CHAT] Stream endpoint hit: /api/v1/chat/stream session={chat_request.session_id} client={client_id}"
    )

    started = time.perf_counter()

    async def generate():
        try:
            validated_message = InputSanitizer.validate_message(chat_request.message)
            result = _rag_engine.process_user_input(
                validated_message, chat_request.session_id
            )

            llm_payload = {
                "description": result.get("enriched_prompt", validated_message),
                "max_tokens": chat_request.max_tokens,
                "temperature": chat_request.temperature,
            }

            circuit_breaker = get_circuit_breaker()
            if not circuit_breaker.can_execute():
                yield f'data: {json.dumps({"error": "LLM service temporarily unavailable"})}\n\n'
                return

            try:
                diagnosis_response = await get_http_client().post(
                    f"{config.original_llm_url}/diagnose",
                    json=llm_payload,
                    headers={"Content-Type": "application/json"},
                )
                diagnosis_response.raise_for_status()
                llm_json = diagnosis_response.json()
                circuit_breaker.record_success()
            except Exception as exc:
                circuit_breaker.record_failure()
                logger.error(f"LLM call failed: {exc}")
                yield f'data: {json.dumps({"error": "LLM backend error"})}\n\n'
                return

            formatted_response = _format_llm_diagnosis_response(llm_json)
            
            processing_time = round(time.perf_counter() - started, 3)
            
            final_response = MedicalDisclaimer.add_disclaimer(
                formatted_response,
                symptoms=result.get("symptoms", []),
                entities=result.get("entities", {}),
            )

            _rag_engine.add_interaction(
                session_id=chat_request.session_id,
                user_input=validated_message,
                extracted_info={
                    "entities": result.get("entities", {}),
                    "symptoms": result.get("symptoms", []),
                    "llm_diagnosis": llm_json,
                },
                ai_response=final_response,
                confidence_score=result.get("confidence_score", 0.0),
            )

            get_metrics().record_chat(
                success=True, symptoms_count=len(result.get("symptoms", []))
            )
            
            logger.info(f"🚀 [STREAMING] Starting streaming response for session {chat_request.session_id}")
            logger.info(f"📊 [STREAMING] Response length: {len(final_response)} characters, {len(final_response.split())} words")
            
            yield f'data: {json.dumps({"text": "🧠 Analyzing your symptoms...\n\n", "type": "start"})}\n\n'
            logger.debug("▶️ [STREAMING] Sent start signal")
            
            words = final_response.split()
            for i, word in enumerate(words):
                yield f'data: {json.dumps({"text": word + (" " if i < len(words) - 1 else ""), "type": "chunk"})}\n\n'
                if (i + 1) % 20 == 0:
                    logger.debug(f"📊 [STREAMING] Streamed {i + 1}/{len(words)} words")
                await asyncio.sleep(0.01)

            logger.info(f"✅ [STREAMING] All chunks sent ({len(words)} words)")
            yield f'data: {json.dumps({"type": "complete", "data": {"response": final_response, "conversation_context": result.get("conversation_context", {}), "extracted_entities": result.get("entities", {}), "symptoms_detected": result.get("symptoms", []), "confidence_score": result.get("confidence_score", 0.0), "session_id": chat_request.session_id, "processing_time": processing_time, "rag_metadata": {"from_cache": result.get("from_cache", False), "urgency": result.get("context", {}).get("medical_urgency", "low")}}})}\n\n'
            logger.debug("🏁 [STREAMING] Sent complete signal")
            yield "data: [DONE]\n\n"
            logger.info(f"🏁 [STREAMING] Stream completed. Total chunks: {len(words)}")
            logger.info(f"✅ [STREAMING] Streaming finished for session {chat_request.session_id} (took {processing_time}s)")

        except Exception as e:
            logger.error(f"❌ [STREAMING] Stream error: {str(e)}", exc_info=True)
            yield f'data: {json.dumps({"error": str(e)})}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")


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


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/performance-profile")
async def performance_profile():
    """Get performance profiling results"""
    from ..core.profiling import get_profiler

    profiler = get_profiler()
    results = profiler.get_results()

    return {
        "profiling_enabled": profiler.enabled,
        "results": {
            name: {
                "total_time": result.total_time,
                "call_count": result.call_count,
            }
            for name, result in results.items()
        },
    }


@router.post("/performance-profile/reset")
async def reset_performance_profile():
    """Reset performance profiling results"""
    from ..core.profiling import get_profiler

    get_profiler().reset()
    return {"status": "reset"}
