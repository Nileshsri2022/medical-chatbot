# Medical RAG Chatbot System
## Healthcare - Advanced Conversational AI

![RAG System](https://img.shields.io/badge/RAG-Enhanced-blue) ![Medical AI](https://img.shields.io/badge/Medical-AI-green) ![FastAPI](https://img.shields.io/badge/FastAPI-Framework-red) ![Python](https://img.shields.io/badge/Python-3.9+-blue)

### 🧠 Intelligent Medical Conversation System with Memory & Context

This advanced RAG (Retrieval-Augmented Generation) system transforms a basic medical chatbot into an intelligent conversational assistant that:
- **Remembers** previous conversations and builds context over time
- **Extracts** medical entities, symptoms, and urgency indicators  
- **Analyzes** conversation flow and provides contextual responses
- **Maintains** conversation state and patient interaction history
- **Provides** real-time context visualization and confidence scoring

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 4GB+ RAM recommended

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create `.env` file:
```bash
# Server
RAG_SERVER_PORT=8002
RAG_HOST=0.0.0.0

# LLM Backend
ORIGINAL_LLM_URL=http://localhost:8001
REQUEST_TIMEOUT=30.0

# Security (optional)
REQUIRE_AUTH=false
API_KEY=your-api-key
JWT_SECRET=your-jwt-secret

# Rate Limiting
RATE_LIMIT_PER_MINUTE=30
```

### 3. Start the Server
```bash
cd backend
python medical_rag_server.py
```

### 4. Access the API
- API Docs: http://localhost:8002/docs
- Health Check: http://localhost:8002/api/v1/health

---

## 🏗️ Architecture

### Service Architecture
```
Frontend (Port 3000)
    ↓
RAG Server (Port 8002) ← API Key / JWT Auth
    ↓
LLM Backend (Port 8001)
```

### Modular Structure
```
backend/
├── api/                    # API routes and dependencies
│   ├── chat_routes.py      # All API endpoints
│   ├── dependencies.py     # Rate limiting, HTTP clients
│   └── __init__.py
├── rag/                    # RAG components (modular)
│   ├── engine.py           # Main RAG coordinator
│   ├── entities.py         # Medical entity recognition
│   ├── symptoms.py         # ML symptom extraction (TF-IDF)
│   ├── memory.py           # SQLite-backed conversation memory
│   ├── context.py          # Context builder for prompts
│   ├── vector_store.py     # ChromaDB + BM25 hybrid retrieval
│   └── document_loader.py  # PDF document loading
├── core/                   # Core utilities
│   ├── security.py         # JWT auth, input sanitization, disclaimers
│   ├── reliability.py      # Circuit breaker, health checks
│   └── monitoring.py       # Structured logging, Prometheus metrics
├── config.py               # Configuration management
└── medical_rag_server.py  # FastAPI server entry point
```

---

## 📡 API Endpoints

### Authentication
```bash
# Generate JWT token
POST /api/v1/auth/token
{"user_id": "user123", "expires_minutes": 60}
```

### Chat
```bash
POST /api/v1/chat
Authorization: Bearer <token>
{
    "message": "I have chest pain",
    "session_id": "session123",
    "max_tokens": 200,
    "temperature": 0.7
}
```

### Other Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check with LLM status |
| `/api/v1/circuit-status` | GET | Circuit breaker status |
| `/api/v1/circuit-reset` | POST | Reset circuit breaker |
| `/api/v1/conversation-history/{id}` | GET | Get session history |
| `/api/v1/conversation/{id}` | DELETE | Clear session |
| `/api/v1/session-stats` | GET | Session statistics |
| `/api/v1/metrics` | GET | Prometheus metrics |

---

## 🔐 Security Features

### Authentication Options
1. **API Key** - Set `REQUIRE_AUTH=true` and `API_KEY`
2. **JWT Tokens** - Use `/auth/token` endpoint

### Medical Safety
- **Input Sanitization** - SQL injection & XSS protection
- **Medical Disclaimers** - Auto-added to all responses
- **Urgent Warnings** - For high-risk symptoms (chest pain, breathing issues)

### Environment Variables
```bash
REQUIRE_AUTH=false       # Enable authentication
API_KEY=your-key         # API key for auth
JWT_SECRET=secret        # JWT signing secret
JWT_EXPIRE_MINUTES=60    # Token expiration
```

---

## 📊 Monitoring

### Metrics Available
- `http_requests_total` - HTTP request count
- `http_request_duration_seconds` - Request latency
- `chat_interactions_total` - Chat interactions
- `symptoms_detected` - Symptoms per interaction
- `llm_calls_total` - LLM API calls
- `active_sessions` - Active session count

### Access Metrics
```bash
curl http://localhost:8002/api/v1/metrics
```

### Structured Logging
Logs written to `medical_rag.log` with request tracing via `X-Request-ID`

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_SERVER_PORT` | 8002 | Server port |
| `ORIGINAL_LLM_URL` | http://localhost:8001 | LLM backend |
| `REQUEST_TIMEOUT` | 30.0 | Request timeout |
| `RATE_LIMIT_PER_MINUTE` | 30 | Rate limit |
| `MAX_CONVERSATION_HISTORY` | 10 | History length |
| `DEFAULT_MAX_TOKENS` | 200 | LLM max tokens |
| `DEFAULT_TEMPERATURE` | 0.7 | LLM temperature |

---

## 🛠️ Technical Implementation

### RAG Pipeline
```
User Input → Medical NER → Symptom Extraction → Context Building → RAG Enhancement → LLM → Response
```

### Components

1. **MedicalEntityRecognizer** - Regex-based entity extraction
2. **SymptomExtractor** - TF-IDF + cosine similarity matching
3. **ConversationMemory** - SQLite-backed session storage
4. **ContextBuilder** - Prompt enrichment with conversation state
5. **HybridRetriever** - ChromaDB vectors + BM25 sparse search

---

## 🐛 Troubleshooting

```bash
# Check health
curl http://localhost:8002/api/v1/health

# Test RAG
curl http://localhost:8002/api/v1/test-rag

# Get stats
curl http://localhost:8002/api/v1/session-stats

# Reset circuit breaker
curl -X POST http://localhost:8002/api/v1/circuit-reset
```

---

## 📈 Version History

### v1.1.0 (Current)
- ✅ Modular codebase (rag/, api/, core/)
- ✅ JWT authentication
- ✅ Pydantic request validation
- ✅ Circuit breaker for LLM failures
- ✅ Prometheus metrics
- ✅ Structured logging (loguru)
- ✅ Hybrid retrieval (ChromaDB + BM25)
- ✅ Medical disclaimers and warnings

### v1.0.0
- ✅ Core RAG engine
- ✅ Medical entity recognition
- ✅ Conversation memory

---

## 📄 License & Compliance

### Medical Compliance
- **Disclaimer**: AI is not a substitute for professional medical advice
- **HIPAA Ready**: Audit logging, secure defaults
- **Data Privacy**: No PHI stored in logs

---

**🏥 Medical RAG Chatbot System**  
*Transforming Medical Conversations with AI Intelligence*