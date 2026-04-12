#!/usr/bin/env python3
"""
Medical RAG Server for Healthcare Chatbot
Modular FastAPI server using separate route modules
"""

import os
import sys
import datetime
import logging
from contextlib import asynccontextmanager

os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import httpx

from .config import config
from .api.dependencies import get_http_clients

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Medical RAG Enhancement Server")
    logger.info(f"📊 Server configured on port {config.port}")
    logger.info(f"🔗 LLM backend: {config.original_llm_url}")
    yield
    http_clients = get_http_clients()
    await http_clients.close()
    logger.info("🔄 HTTP clients closed")


app = FastAPI(
    title="Medical RAG API",
    description="Advanced medical conversation AI with context awareness and memory",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Medical RAG API",
        version="1.0.0",
        description="Advanced medical conversation AI with context awareness and memory",
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }
    openapi_schema["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

ALLOWED_ORIGINS = config.allowed_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from .api import chat_routes

app.include_router(chat_routes.router)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    print("🏥 Medical RAG Enhancement Server")
    print("=" * 50)
    print(f"🚀 Starting server on port {config.port}")
    print(f"🔗 LLM backend: {config.original_llm_url}")
    print(f"🧠 RAG engine initialized (modular)")
    print("=" * 50)

    uvicorn.run(
        "medical_rag_server:app",
        host="0.0.0.0",
        port=config.port,
        log_level="info",
        access_log=True,
    )
