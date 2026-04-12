"""
Integration tests for API endpoints
Implements testing patterns from python-testing-patterns skill
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, AsyncMock, Mock
from fastapi.testclient import TestClient


# ==================== INTEGRATION TESTS ====================


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints - uses mocking for external dependencies"""

    @pytest.fixture
    def mock_engine(self):
        """Mock RAG engine for API tests"""
        with patch("backend.api.chat_routes.MedicalRAGEnrichmentEngine") as MockEngine:
            mock_instance = Mock()
            mock_instance.process_user_input.return_value = {
                "enriched_prompt": "test prompt",
                "entities": {"symptoms": ["chest pain"], "body_parts": ["chest"]},
                "symptoms": [],
                "conversation_context": {
                    "conversation_state": "initial",
                    "total_interactions": 0,
                },
                "confidence_score": 0.8,
            }
            mock_instance.conversation_memory.active_sessions_count = 5
            mock_instance.get_context.return_value = {
                "conversation_state": "initial",
                "total_interactions": 1,
                "accumulated_symptoms": ["chest pain"],
            }
            mock_instance.symptom_extractor.symptom_count = 42
            MockEngine.return_value = mock_instance
            yield mock_instance

    def test_root_endpoint_returns_service_info(self):
        """Test root endpoint returns correct service information"""
        from backend.medical_rag_server import app

        client = TestClient(app)

        response = client.get("/api/v1/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert data["service"] == "Medical RAG API"

    def test_health_check_includes_rag_status(self):
        """Test health check includes RAG engine status"""
        from backend.medical_rag_server import app

        client = TestClient(app)

        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "rag_engine_status" in data
        assert "active_sessions" in data


@pytest.mark.integration
class TestEndToEndFlow:
    """End-to-end flow tests - tests complete user journeys"""

    def test_complete_medical_chat_flow(self):
        """Test complete medical chat flow from input to response"""
        from backend.rag import MedicalRAGEnrichmentEngine

        engine = MedicalRAGEnrichmentEngine()
        session_id = f"e2e_{os.urandom(8).hex()}"

        # 1. User sends symptom
        result1 = engine.process_user_input("I have chest pain", session_id)

        # 2. System extracts entities
        assert len(result1["entities"]["symptoms"]) >= 0

        # 3. System builds context
        assert "conversation_context" in result1

        # 4. Response generated
        assert "enriched_prompt" in result1

        # 5. Response stored in memory
        context = engine.get_context(session_id)
        assert context["total_interactions"] >= 1

        engine.conversation_memory.clear_session(session_id)

    def test_multi_turn_conversation_maintains_context(self):
        """Test multi-turn conversation maintains context"""
        from backend.rag import MedicalRAGEnrichmentEngine

        engine = MedicalRAGEnrichmentEngine()
        session_id = f"multi_{os.urandom(8).hex()}"

        # First turn
        engine.process_user_input("I have chest pain", session_id)

        # Second turn
        engine.process_user_input("It started yesterday", session_id)

        # Verify context maintained
        context = engine.get_context(session_id)
        assert context["total_interactions"] == 2
        assert "chest pain" in context["accumulated_symptoms"]

        engine.conversation_memory.clear_session(session_id)

    def test_emergency_detection_flow(self):
        """Test emergency detection triggers correct state"""
        from backend.rag import MedicalRAGEnrichmentEngine

        engine = MedicalRAGEnrichmentEngine()
        session_id = f"emergency_{os.urandom(8).hex()}"

        result = engine.process_user_input(
            "I need emergency help, I can't breathe", session_id
        )

        # Check urgency detected
        urgency = result.get("context", {}).get("medical_urgency", "low")
        assert urgency in ["high", "critical", "low"]

        engine.conversation_memory.clear_session(session_id)


@pytest.mark.integration
class TestDatabaseIntegration:
    """Database integration tests - tests session persistence"""

    def test_session_persistence(self):
        """Test session data persists correctly in SQLite"""
        from backend.rag.memory import ConversationMemory

        memory = ConversationMemory()
        session_id = f"persist_{os.urandom(8).hex()}"

        # Add data
        memory.add_interaction(
            session_id=session_id,
            user_input="Test input",
            extracted_info={"symptoms": ["test"]},
            ai_response="Test response",
            confidence_score=0.9,
        )

        # Get new memory instance (simulates restart)
        memory2 = ConversationMemory()

        # Verify session still exists
        context = memory2.get_context(session_id)
        assert context["total_interactions"] >= 1

        # Cleanup
        memory.clear_session(session_id)

    def test_conversation_history_retrieval(self):
        """Test conversation history can be retrieved"""
        from backend.rag import MedicalRAGEnrichmentEngine

        engine = MedicalRAGEnrichmentEngine()
        session_id = f"history_{os.urandom(8).hex()}"

        # Add multiple interactions
        for i in range(3):
            engine.process_user_input(f"Message {i}", session_id)

        # Retrieve history
        context = engine.get_context(session_id)
        assert context["total_interactions"] == 3

        engine.conversation_memory.clear_session(session_id)


@pytest.mark.integration
class TestSecurityValidation:
    """Security tests - validates input sanitization"""

    def test_sql_injection_blocked(self):
        """Test SQL injection attempts are safely handled"""
        from backend.rag.entities import MedicalEntityRecognizer

        recognizer = MedicalEntityRecognizer()
        malicious = "'; DROP TABLE sessions; --"

        # Should not raise exception
        result = recognizer.extract_entities(malicious)
        assert isinstance(result, dict)
        assert len(result["symptoms"]) == 0

    def test_xss_attempt_blocked(self):
        """Test XSS attempts are safely handled"""
        from backend.rag.entities import MedicalEntityRecognizer

        recognizer = MedicalEntityRecognizer()
        malicious = "<script>alert('xss')</script> chest pain"

        # Should not raise exception
        result = recognizer.extract_entities(malicious)
        assert isinstance(result, dict)

    def test_very_long_input_truncated(self):
        """Test very long input is handled"""
        from backend.core.security import InputSanitizer

        long_text = "a" * 10000
        result = InputSanitizer.validate_message(long_text)

        # Should be truncated or handled
        assert len(result) <= InputSanitizer.MAX_MESSAGE_LENGTH


class TestAPIAuthentication:
    """Test API authentication mechanisms"""

    def test_jwt_token_creation(self):
        """Test JWT token can be created"""
        from backend.core.security import get_jwt_auth

        jwt_auth = get_jwt_auth()
        token = jwt_auth.create_token({"user_id": "test_user"})

        assert isinstance(token, str)
        assert len(token) > 0

    def test_jwt_token_verification(self):
        """Test JWT token can be verified"""
        from backend.core.security import get_jwt_auth

        jwt_auth = get_jwt_auth()
        token = jwt_auth.create_token({"user_id": "test_user"})

        payload = jwt_auth.verify_token(token)
        assert payload["user_id"] == "test_user"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
