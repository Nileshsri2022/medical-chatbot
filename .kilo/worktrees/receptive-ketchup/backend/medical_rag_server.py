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


# FastAPI application
app = FastAPI(
    title="Medical RAG Enhancement Server",
    description="Advanced medical conversation AI with context awareness and memory",
    version="1.0.0",
)

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


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with server information"""
    return {
        "message": "Medical RAG Enhancement Server",
        "status": "operational",
        "version": "1.0.0",
        "description": "Enhanced medical conversation AI with context awareness",
        "endpoints": {
            "enhanced_chat": "/enhanced-chat",
            "health": "/health",
            "conversation_history": "/conversation-history/{session_id}",
            "reset_conversation": "/reset-conversation/{session_id}",
            "session_stats": "/session-stats",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""

    # Check original LLM backend connectivity
    llm_status = "unknown"
    try:
        response = await http_client.get(f"{config.original_llm_url}/health")
        llm_status = "connected" if response.status_code == 200 else "disconnected"
    except Exception:
        llm_status = "disconnected"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.datetime.now().isoformat(),
        version="1.0.0",
        rag_engine_status="operational",
        original_llm_status=llm_status,
        active_sessions=len(session_stats["active_sessions"]),
    )


@app.post("/enhanced-chat", response_model=EnhancedChatResponse)
async def enhanced_chat(request: EnhancedChatRequest):
    """
    Enhanced chat endpoint that provides context-aware medical conversation

    This endpoint:
    1. Extracts medical entities and symptoms from user input
    2. Builds conversation context and memory
    3. Creates enriched prompt for LLM
    4. Calls original LLM backend with enhanced prompt
    5. Returns response with context information
    """

    start_time = datetime.datetime.now()

    try:
        logger.info(
            f"🔍 Processing enhanced chat request for session {request.session_id}"
        )

        # Track session
        session_stats["active_sessions"].add(request.session_id)
        session_stats["total_interactions"] += 1

        # 1. Process user input through RAG pipeline
        logger.info("🧠 Running RAG analysis...")
        rag_result = rag_engine.process_user_input(request.message, request.session_id)

        # 2. Call original LLM backend - parallelize response and medical info extraction
        logger.info("📡 Sending user message to medical AI for symptom analysis...")

        # Run both calls in parallel using asyncio.gather
        medical_ai_response, llm_medical_info = await asyncio.gather(
            call_original_llm_backend(
                request.message,
                request.max_tokens,
                request.temperature,
            ),
            parse_llm_medical_response(
                request.message, request.max_tokens, request.temperature
            ),
        )

        # 3. Store interaction in conversation memory with enhanced medical information
        logger.info("💾 Storing interaction in memory...")
        enhanced_extracted_info = {
            "entities": rag_result["entities"],
            "symptoms": rag_result["symptoms"],
            "llm_symptoms": llm_medical_info.get("symptoms", []),
            "llm_illnesses": llm_medical_info.get("illnesses", []),
        }

        rag_engine.conversation_memory.add_interaction(
            session_id=request.session_id,
            user_input=request.message,
            extracted_info=enhanced_extracted_info,
            ai_response=medical_ai_response,
            confidence_score=rag_result["confidence_score"],
        )

        # 4. Calculate processing time
        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        # 5. Prepare response with comprehensive metadata including LLM medical analysis
        response = EnhancedChatResponse(
            response=medical_ai_response,
            conversation_context=rag_result["conversation_context"],
            extracted_entities=rag_result["entities"],
            symptoms_detected=rag_result["symptoms"],
            confidence_score=rag_result["confidence_score"],
            session_id=request.session_id,
            timestamp=datetime.datetime.now().isoformat(),
            processing_time=processing_time,
            rag_metadata={
                "symptoms_count": len(rag_result["symptoms"]),
                "entities_count": sum(
                    len(v) if isinstance(v, list) else 0
                    for v in rag_result["entities"].values()
                ),
                "conversation_state": rag_result["conversation_context"].get(
                    "conversation_state", "unknown"
                ),
                "urgency_level": rag_result["conversation_context"].get(
                    "urgency_level", "low"
                ),
                "prompt_length": len(rag_result["enriched_prompt"]),
                "context_quality": calculate_context_quality(rag_result),
                "llm_medical_analysis": {
                    "symptoms_identified": llm_medical_info.get("symptoms", []),
                    "illnesses_suggested": llm_medical_info.get("illnesses", []),
                    "symptoms_count": len(llm_medical_info.get("symptoms", [])),
                    "illnesses_count": len(llm_medical_info.get("illnesses", [])),
                },
            },
        )

        logger.info(f"✅ Enhanced chat completed in {processing_time:.2f}s")
        return response

    except httpx.TimeoutException:
        logger.error("⏰ Timeout calling LLM backend")
        raise HTTPException(
            status_code=504,
            detail="LLM backend timeout. Please check if the original medical AI server is running and responsive.",
        )
    except httpx.ConnectError:
        logger.error("🔌 Cannot connect to LLM backend")
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to LLM backend at {config.original_llm_url}. Please ensure the original medical AI server is running.",
        )
    except Exception as e:
        logger.error(f"❌ Error in enhanced chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG processing error: {str(e)}")


from fastapi.responses import StreamingResponse


@app.post("/enhanced-chat-stream")
async def enhanced_chat_stream(request: EnhancedChatRequest):
    """
    Stream Server-Sent Events (SSE) word by word back to the UI.
    First chunk contains RAG metadata, following chunks contain generated text.
    """

    async def event_generator():
        start_time = datetime.datetime.now()

        # 1. Standard RAG Engine Match
        rag_result = rag_engine.process_user_input(request.message, request.session_id)

        # 2. Package all UI Metadata and send as the first chunk
        metadata = {
            "type": "metadata",
            "conversation_context": rag_result["conversation_context"],
            "extracted_entities": rag_result["entities"],
            "symptoms_detected": rag_result["symptoms"],
            "confidence_score": rag_result["confidence_score"],
            "session_id": request.session_id,
            "rag_metadata": {
                "symptoms_count": len(rag_result["symptoms"]),
            },
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        # 3. Stream from the LLM Backend proxying the ENRICHED prompt
        payload = {
            "description": rag_result["enriched_prompt"],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        try:
            # Stream the chunks over HTTP back
            async with streaming_client.stream(
                "POST", f"{config.original_llm_url}/chat/stream", json=payload
            ) as response:
                full_response = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(raw_data)
                            text_fragment = chunk_json.get("text", "")
                            full_response += text_fragment
                            # Yield the chunk immediately to the frontend
                            yield f"data: {json.dumps({'type': 'chunk', 'text': text_fragment})}\n\n"
                        except Exception:
                            pass

                # 4. Save to history once generation concludes
                enhanced_extracted_info = {
                    "entities": rag_result["entities"],
                    "symptoms": rag_result["symptoms"],
                }
                rag_engine.conversation_memory.add_interaction(
                    session_id=request.session_id,
                    user_input=request.message,
                    extracted_info=enhanced_extracted_info,
                    ai_response=full_response,
                    confidence_score=rag_result["confidence_score"],
                )
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'chunk', 'text': ' [Connection Error to LLM] '})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def format_medical_ai_response(api_response: Dict) -> str:
    """
    Format the original medical AI response into a readable text response

    Original medical AI returns: {
        "input_text": "user message",
        "matched_symptoms": [
            {
                "id": "HP:0004889",
                "label": "Intermittent episodes of respiratory insufficiency due to muscle weakness",
                "similarity": 0.872
            }
        ],
        "probable_diseases": [
            {
                "id": "OMIM:265400",
                "name": "Pulmonary hypertension, primary, autosomal recessive",
                "score": 1.709
            }
        ]
    }

    This function converts it into a coherent medical response.
    """

    try:
        input_text = api_response.get("input_text", "")
        matched_symptoms = api_response.get("matched_symptoms", [])
        probable_diseases = api_response.get("probable_diseases", [])

        # Build a coherent medical response
        response_parts = []

        # Start with acknowledgment of the patient's concern
        if input_text:
            response_parts.append(
                f"Thank you for describing your symptoms: {input_text}"
            )

        # Add symptom analysis if symptoms were matched
        if matched_symptoms:
            # Process medical symptom objects
            symptom_labels = []
            high_confidence_symptoms = []

            for symptom in matched_symptoms:
                if isinstance(symptom, dict):
                    # Extract symptom label and confidence
                    label = symptom.get("label", symptom.get("name", str(symptom)))
                    similarity = symptom.get("similarity", 0)

                    symptom_labels.append(label)

                    # Flag high-confidence symptoms (>0.85 similarity)
                    if similarity > 0.85:
                        high_confidence_symptoms.append(label)
                else:
                    symptom_labels.append(str(symptom))

            if symptom_labels:
                # Prioritize high-confidence symptoms for response
                primary_symptoms = (
                    high_confidence_symptoms[:3]
                    if high_confidence_symptoms
                    else symptom_labels[:3]
                )

                if len(primary_symptoms) == 1:
                    response_parts.append(
                        f"Based on your description, I've identified: {primary_symptoms[0]}."
                    )
                else:
                    symptoms_text = (
                        ", ".join(primary_symptoms[:-1])
                        + f", and {primary_symptoms[-1]}"
                    )
                    response_parts.append(
                        f"Based on your description, I've identified these symptoms: {symptoms_text}."
                    )

        # Add possible conditions if any were identified
        if probable_diseases:
            # Process medical disease objects and prioritize by score
            disease_info = []
            has_coverage_info = False

            for disease in probable_diseases:
                if isinstance(disease, dict):
                    name = disease.get(
                        "name",
                        disease.get("disease", disease.get("condition", str(disease))),
                    )
                    score = disease.get("score", 0)
                    illness_coverage = disease.get("illness_coverage", None)
                    condition_coverage = disease.get("condition_coverage", None)

                    # Check if we have coverage information from new format
                    if illness_coverage is not None or condition_coverage is not None:
                        has_coverage_info = True
                        disease_info.append(
                            (name, score, illness_coverage, condition_coverage)
                        )
                    else:
                        disease_info.append((name, score, None, None))
                else:
                    disease_info.append((str(disease), 0, None, None))

            # Sort by score (highest first) and take top 3
            disease_info.sort(key=lambda x: x[1], reverse=True)

            if disease_info:
                if has_coverage_info:
                    # Enhanced format with coverage information
                    response_parts.append(
                        "Based on the analysis, possible conditions to consider include:"
                    )
                    for i, (name, score, illness_cov, condition_cov) in enumerate(
                        disease_info[:3]
                    ):
                        coverage_text = ""
                        if illness_cov is not None and condition_cov is not None:
                            coverage_text = f" (illness match: {illness_cov}%, condition match: {condition_cov}%)"
                        response_parts.append(f"• {name}{coverage_text}")
                else:
                    # Standard format
                    top_diseases = [name for name, score, _, _ in disease_info[:3]]
                    if len(top_diseases) == 1:
                        response_parts.append(
                            f"This could potentially be related to: {top_diseases[0]}."
                        )
                    else:
                        diseases_text = (
                            ", ".join(top_diseases[:-1]) + f", or {top_diseases[-1]}"
                        )
                        response_parts.append(
                            f"Possible conditions to consider include: {diseases_text}."
                        )

                # Add medical advice
                response_parts.append(
                    "However, this is a preliminary assessment based on symptom matching."
                )

        # Add standard medical disclaimer and advice
        if matched_symptoms or probable_diseases:
            response_parts.append(
                "I strongly recommend consulting with a healthcare professional for proper diagnosis and treatment. If you're experiencing severe symptoms, please seek immediate medical attention."
            )
        else:
            # No specific symptoms or diseases identified
            response_parts.append(
                "While I couldn't match specific symptoms or conditions from your description, I recommend discussing your concerns with a healthcare professional who can provide a proper evaluation."
            )

        # Join all parts into a coherent response
        if response_parts:
            return " ".join(response_parts)
        else:
            return "I've received your message, but I couldn't provide a specific medical assessment. Please consult with a healthcare professional for proper guidance."

    except Exception as e:
        logger.error(
            f"Error formatting medical AI response: {str(e)} - Response data: {api_response}"
        )
        return "I apologize, but I encountered an issue while processing your medical inquiry. Please try rephrasing your question or consult with a healthcare professional directly."


async def parse_llm_medical_response(
    user_message: str, max_tokens: int, temperature: float
) -> Dict:
    """
    Parse the LLM response to extract structured medical information (symptoms and illnesses)
    Returns the raw JSON response for storing in conversation context
    """
    try:
        endpoint = f"{config.original_llm_url}/diagnose"

        response = await http_client.post(
            endpoint,
            json={
                "description": user_message,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Raw LLM medical data: {data}")

            # Extract symptoms and illnesses from the new format
            if isinstance(data, dict) and "symptoms" in data and "illnesses" in data:
                return {
                    "symptoms": data.get("symptoms", []),
                    "illnesses": data.get("illnesses", []),
                }
            else:
                # Fallback to empty structure
                return {"symptoms": [], "illnesses": []}
        else:
            logger.warning(
                f"Failed to parse LLM medical response: {response.status_code}"
            )
            return {"symptoms": [], "illnesses": []}

    except Exception as e:
        logger.error(f"Error parsing LLM medical response: {str(e)}")
        return {"symptoms": [], "illnesses": []}
        logger.error(f"Error parsing LLM medical response: {str(e)}")
        return {"symptoms": [], "illnesses": []}


async def call_original_llm_backend(
    user_message: str, max_tokens: int, temperature: float
) -> str:
    """
    Call the original medical AI backend via CORS proxy with the user's message

    This function handles the communication with the CORS proxy server,
    which forwards requests to the medical AI server via SSH tunnel.
    The medical AI expects simple symptom descriptions, not enriched prompts.
    """

    try:
        endpoint = f"{config.original_llm_url}/diagnose"

        logger.debug(f"Calling CORS proxy endpoint: {endpoint}")

        response = await http_client.post(
            endpoint,
            json={
                "description": user_message,  # Send user's original message to medical AI
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Raw API response: {data}")

            # Handle the original medical AI response format
            if isinstance(data, dict):
                # Check for original medical AI format: {input_text, matched_symptoms, probable_diseases}
                if "input_text" in data:
                    logger.info(
                        f"Formatting medical AI response with {len(data.get('matched_symptoms', []))} symptoms and {len(data.get('probable_diseases', []))} diseases"
                    )
                    formatted_response = format_medical_ai_response(data)
                    logger.debug(f"Formatted response: {formatted_response[:200]}...")
                    return formatted_response
                # Check for alternative format: {symptoms, illnesses}
                elif "symptoms" in data and "illnesses" in data:
                    logger.info(
                        f"Formatting alternative medical AI response with {len(data.get('symptoms', []))} symptoms and {len(data.get('illnesses', []))} illnesses"
                    )
                    # Convert new format to expected format for existing formatter
                    # Handle illnesses as objects with coverage scores
                    converted_illnesses = []
                    for illness in data.get("illnesses", []):
                        if isinstance(illness, dict):
                            # Use illness_coverage as score for sorting
                            illness_score = illness.get("illness_coverage", 0)
                            converted_illnesses.append(
                                {
                                    "name": illness.get("name", "Unknown condition"),
                                    "score": illness_score,
                                    "illness_coverage": illness.get(
                                        "illness_coverage", 0
                                    ),
                                    "condition_coverage": illness.get(
                                        "condition_coverage", 0
                                    ),
                                }
                            )
                        else:
                            # Fallback for simple string format
                            converted_illnesses.append(
                                {"name": str(illness), "score": 0}
                            )

                    converted_data = {
                        "input_text": "Patient symptoms analysis",
                        "matched_symptoms": [
                            {"label": symptom, "similarity": 0.9}
                            for symptom in data.get("symptoms", [])
                        ],
                        "probable_diseases": converted_illnesses,
                    }
                    formatted_response = format_medical_ai_response(converted_data)
                    logger.debug(
                        f"Formatted alternative response: {formatted_response[:200]}..."
                    )
                    return formatted_response
                # Fallback to other possible response formats
                fallback_response = data.get(
                    "response",
                    data.get("result", data.get("answer", "No response received")),
                )
                logger.warning(
                    f"Using fallback response format: {fallback_response[:100]}..."
                )
                return fallback_response
            elif isinstance(data, str):
                logger.info(f"Received string response: {data[:100]}...")
                return data
            else:
                logger.warning(f"Received unknown response type: {type(data)}")
                return str(data)
        else:
            error_data = (
                response.json() if response.content else {"error": "No error details"}
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"CORS proxy returned {response.status_code}: {error_data}",
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM backend request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from CORS proxy: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"CORS proxy error: {e.response.status_code}",
        )
    except httpx.ConnectError:
        logger.error("Cannot connect to CORS proxy")
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to LLM backend at {config.original_llm_url}. Please ensure the original medical AI server is running.",
        )
    except Exception as e:
        logger.error(f"Error calling LLM backend: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Error communicating with LLM backend: {str(e)}",
        )
        logger.error(f"Error calling LLM backend: {str(e)}")
        raise HTTPException(
            status_code=502, detail=f"Error communicating with LLM backend: {str(e)}"
        )


def calculate_context_quality(rag_result: Dict) -> float:
    """Calculate the quality of the context built by RAG"""

    score = 0.0

    # Symptom detection quality
    symptoms = rag_result.get("symptoms", [])
    if symptoms:
        avg_symptom_confidence = sum(s.get("confidence", 0) for s in symptoms) / len(
            symptoms
        )
        score += avg_symptom_confidence * 0.4

    # Entity extraction quality
    entities = rag_result.get("entities", {})
    entity_count = sum(len(v) if isinstance(v, list) else 0 for v in entities.values())
    score += min(1.0, entity_count * 0.1) * 0.3

    # Conversation context quality
    context = rag_result.get("conversation_context", {})
    interactions = context.get("total_interactions", 0)
    score += min(1.0, interactions * 0.1) * 0.3

    return round(min(1.0, score), 2)


@app.get(
    "/conversation-history/{session_id}", response_model=ConversationHistoryResponse
)
async def get_conversation_history(session_id: str):
    """Get detailed conversation history for a session"""

    try:
        context = rag_engine.conversation_memory.get_context(session_id)

        return ConversationHistoryResponse(
            session_id=session_id,
            conversation_context=context,
            total_interactions=context.get("total_interactions", 0),
            session_start_time=context.get("session_start_time", "unknown"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found or error retrieving history: {str(e)}",
        )


@app.delete("/reset-conversation/{session_id}")
async def reset_conversation(session_id: str, background_tasks: BackgroundTasks):
    """Reset conversation memory for a session"""

    try:
        # Remove from active sessions
        session_stats["active_sessions"].discard(session_id)

        # Clear conversation memory
        rag_engine.conversation_memory.clear_session(session_id)
        logger.info(f"🗑️ Reset conversation for session {session_id} in SQLite")

        return {
            "message": f"Conversation {session_id} reset successfully",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error resetting conversation: {str(e)}"
        )


@app.get("/session-stats")
async def get_session_stats():
    """Get server and session statistics"""

    uptime = datetime.datetime.now() - datetime.datetime.fromisoformat(
        session_stats["server_start_time"]
    )

    # Calculate memory usage for active sessions
    memory_usage = {
        "active_sessions": len(session_stats["active_sessions"]),
        "total_conversations_in_memory": len(rag_engine.conversation_memory.sessions),
        "total_interactions_stored": sum(
            len(session["conversation_history"])
            for session in rag_engine.conversation_memory.sessions.values()
        ),
    }

    return {
        "server_stats": {
            "uptime_seconds": uptime.total_seconds(),
            "uptime_formatted": str(uptime),
            "server_start_time": session_stats["server_start_time"],
            "total_sessions_created": session_stats["total_sessions"],
            "total_interactions_processed": session_stats["total_interactions"],
        },
        "memory_usage": memory_usage,
        "rag_engine_stats": {
            "symptom_patterns": rag_engine.symptom_extractor.symptom_count,
            "entity_patterns": len(rag_engine.medical_ner.medical_patterns),
            "active_conversation_states": list(
                set(
                    session["conversation_state"].value
                    for session in rag_engine.conversation_memory.sessions.values()
                )
            ),
        },
    }


@app.get("/debug/session/{session_id}")
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
