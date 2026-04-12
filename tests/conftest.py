"""
Shared test fixtures and configuration
Implements pytest best practices from python-testing-patterns skill
"""

import os
import sys
import pytest
import tempfile
import shutil
from typing import Generator
from unittest.mock import Mock, MagicMock, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_medical_text() -> str:
    """Sample medical text for testing"""
    return (
        "I have severe chest pain that started this morning and radiates to my left arm"
    )


@pytest.fixture
def sample_symptom_text() -> str:
    """Sample symptom descriptions"""
    return "I have a headache and fever"


@pytest.fixture
def mock_llm_response() -> dict:
    """Mock LLM response for testing"""
    return {
        "choices": [
            {
                "message": {
                    "content": "Based on your symptoms, please seek immediate medical attention."
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture
def rag_engine():
    """Create RAG engine instance for testing"""
    from backend.rag import MedicalRAGEnrichmentEngine

    engine = MedicalRAGEnrichmentEngine()
    yield engine
    for session_id in list(engine.conversation_memory.sessions.keys()):
        if session_id.startswith("test_"):
            engine.conversation_memory.clear_session(session_id)


@pytest.fixture
def entity_recognizer():
    """Create entity recognizer instance for testing"""
    from backend.rag.entities import MedicalEntityRecognizer

    return MedicalEntityRecognizer()


@pytest.fixture
def conversation_memory():
    """Create conversation memory for testing"""
    from backend.rag.memory import ConversationMemory

    memory = ConversationMemory()
    yield memory
    for session_id in list(memory.sessions.keys()):
        if session_id.startswith("test_"):
            memory.clear_session(session_id)


@pytest.fixture
def unique_session_id() -> str:
    """Generate unique session ID for tests"""
    return f"test_{os.urandom(8).hex()}"


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing"""
    client = Mock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def app_config() -> dict:
    """Test application configuration"""
    return {
        "RAG_SERVER_PORT": 8002,
        "original_llm_url": "http://localhost:8001",
        "request_timeout": 30.0,
        "rate_limit_requests_per_minute": 30,
        "default_max_tokens": 200,
        "default_temperature": 0.7,
    }


@pytest.fixture(autouse=True)
def reset_test_env(monkeypatch):
    """Reset environment for each test"""
    monkeypatch.setenv("TESTING", "true")
    yield


@pytest.fixture
def temp_db_path(tmp_path) -> str:
    """Create temporary database path"""
    db_path = tmp_path / "test_medical_chats.db"
    return str(db_path)


@pytest.fixture
def sample_conversation_history() -> list:
    """Sample conversation history for testing"""
    return [
        {
            "user_input": "I have chest pain",
            "ai_response": "When did this start?",
            "timestamp": "2024-01-15T10:00:00",
            "confidence_score": 0.8,
        },
        {
            "user_input": "It started this morning",
            "ai_response": "Any other symptoms?",
            "timestamp": "2024-01-15T10:01:00",
            "confidence_score": 0.7,
        },
    ]


@pytest.fixture
def sample_entities() -> dict:
    """Sample extracted entities"""
    return {
        "symptoms": ["chest pain", "shortness of breath"],
        "body_parts": ["chest", "arm"],
        "conditions": ["heart disease"],
        "medications": ["aspirin"],
        "temporal": ["today", "morning"],
        "severity": "severe",
        "duration": "acute",
        "urgency_indicators": ["chest pain"],
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
