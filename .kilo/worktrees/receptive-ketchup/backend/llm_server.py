#!/usr/bin/env python3
"""
Real Medical LLM Backend Server
Unified OpenAI-compatible backend that works with Groq, Google Gemini, OpenAI,
or any OpenAI-compatible API. Switch providers by changing .env values.

Runs on port 8001 and provides the /diagnose endpoint expected by the RAG server.

Architecture:
  Frontend -> RAG Server (8002) -> This LLM Server (8001) -> LLM Provider API
"""

import os
import sys
import json
import re
import datetime
import logging
import time
from typing import Dict, List, Optional

# Fix Windows console encoding
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# Load .env from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_SERVER_PORT: int = int(os.getenv("LLM_SERVER_PORT", "8001"))

if not LLM_API_KEY or LLM_API_KEY.endswith("_here"):
    logger.error("=" * 60)
    logger.error("  LLM_API_KEY is not set!")
    logger.error("  Copy .env.example to .env and add your API key.")
    logger.error("=" * 60)
    sys.exit(1)

# ─── OpenAI Client ───────────────────────────────────────────────────────────

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical LLM Backend",
    description="Real LLM backend using OpenAI-compatible API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ──────────────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    description: str = Field(..., description="Patient symptom description")
    max_tokens: int = Field(200, description="Maximum tokens for response")
    temperature: float = Field(0.7, description="LLM temperature")


# ─── Medical System Prompt ───────────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = """You are a medical symptom analysis AI. Your job is to:
1. Identify symptoms from the patient's description
2. Suggest possible conditions/illnesses based on those symptoms

IMPORTANT: You MUST respond ONLY with valid JSON in exactly this format, no markdown, no extra text:
{
  "symptoms": ["symptom1", "symptom2"],
  "illnesses": [
    {
      "name": "Condition Name",
      "illness_coverage": 75,
      "condition_coverage": 60
    }
  ]
}

Rules:
- "symptoms" is a list of identified symptom strings (use standard medical terminology)
- "illnesses" is a list of possible conditions, each with:
  - "name": condition name
  - "illness_coverage": percentage (0-100) of that illness's typical symptoms that match
  - "condition_coverage": percentage (0-100) of the patient's symptoms explained by this condition
- List up to 5 most likely conditions, sorted by relevance
- Be thorough in symptom identification
- Include both common and serious possibilities
- Always include a disclaimer-worthy common condition and any serious red flags

Respond ONLY with the JSON object. No other text."""


# ─── Rate Limiting (simple) ─────────────────────────────────────────────────

_last_request_time: float = 0.0
_min_interval: float = 0.5  # seconds between requests


def rate_limit() -> None:
    """Simple rate limiter to avoid hitting provider limits."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_request_time = time.time()


# ─── LLM Call ────────────────────────────────────────────────────────────────

def call_llm(description: str, max_tokens: int, temperature: float) -> Dict:
    """Call the LLM provider and parse the medical analysis response."""

    rate_limit()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze these symptoms: {description}"},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        raw_text = response.choices[0].message.content.strip()
        logger.info(f"LLM raw response length: {len(raw_text)} chars")

        # Try to parse JSON from the response
        parsed = _extract_json(raw_text)

        if parsed and "symptoms" in parsed and "illnesses" in parsed:
            return parsed

        # Fallback: return raw text as a single symptom
        logger.warning("LLM response was not valid JSON, using fallback")
        return {
            "symptoms": [description],
            "illnesses": [
                {
                    "name": "Unable to determine - please consult a doctor",
                    "illness_coverage": 0,
                    "condition_coverage": 0,
                }
            ],
        }

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response, handling markdown code blocks."""

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block ```json ... ```
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Medical LLM Backend (Real)",
        "status": "operational",
        "version": "2.0.0",
        "provider": LLM_BASE_URL,
        "model": LLM_MODEL,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "2.0.0",
        "provider": LLM_BASE_URL,
        "model": LLM_MODEL,
    }


@app.post("/diagnose")
async def diagnose(request: DiagnoseRequest):
    """
    Real diagnosis endpoint using LLM.
    Returns: { "symptoms": [...], "illnesses": [...] }
    Same format the RAG server expects.
    """
    logger.info(f"Diagnosis request: {request.description[:80]}...")

    result = call_llm(request.description, request.max_tokens, request.temperature)

    symptom_count = len(result.get("symptoms", []))
    illness_count = len(result.get("illnesses", []))
    logger.info(f"LLM identified {symptom_count} symptoms, {illness_count} conditions")

    return result


from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: DiagnoseRequest):
    """
    Stream a natural language response directly from the LLM provider.
    """
    def generate():
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an intelligent medical AI assistant. Answer the user based on the provided context if any."},
                {"role": "user", "content": request.description},
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True
        )
        for chunk in response:
            if getattr(chunk.choices[0].delta, "content", None):
                yield f"data: {json.dumps({'text': chunk.choices[0].delta.content})}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Detect provider name for display
    provider_name = "Unknown"
    if "groq" in LLM_BASE_URL:
        provider_name = "Groq"
    elif "generativelanguage.googleapis" in LLM_BASE_URL:
        provider_name = "Google Gemini"
    elif "api.openai.com" in LLM_BASE_URL:
        provider_name = "OpenAI"
    elif "integrate.api.nvidia.com" in LLM_BASE_URL:
        provider_name = "NVIDIA NIM"
    elif "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL:
        provider_name = "Local Server"

    print("=" * 58)
    print("  Medical LLM Backend Server (Real AI)")
    print(f"  Provider:  {provider_name}")
    print(f"  Model:     {LLM_MODEL}")
    print(f"  Base URL:  {LLM_BASE_URL}")
    print(f"  Server:    http://localhost:{LLM_SERVER_PORT}")
    print("=" * 58)

    uvicorn.run(app, host="0.0.0.0", port=LLM_SERVER_PORT, log_level="info")
