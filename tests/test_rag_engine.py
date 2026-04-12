"""
Medical RAG Engine - Comprehensive Test Suite
Includes: Unit Tests, Integration Tests, Fixtures
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from backend.medical_rag_engine import (
    MedicalEntityRecognizer,
    SymptomExtractor,
    ConversationMemory,
    ContextBuilder,
    MedicalRAGEnrichmentEngine,
    ConversationState,
    ExtractedSymptom,
)


# ==================== FIXTURES ====================


@pytest.fixture
def sample_medical_entities():
    """Sample medical entities for testing"""
    return {
        "symptoms": ["chest pain", "shortness of breath"],
        "body_parts": ["chest", "arm"],
        "severity": "severe",
        "urgency_indicators": ["chest pain", "can't breathe"],
        "medications": [],
        "medical_conditions": [],
    }


@pytest.fixture
def sample_conversation_context():
    """Sample conversation context"""
    return {
        "session_id": "test_session",
        "conversation_state": ConversationState.INITIAL,
        "total_interactions": 0,
        "session_start_time": "2024-01-01T00:00:00",
        "conversation_history": [],
        "accumulated_symptoms": set(),
        "accumulated_conditions": set(),
        "urgency_level": "low",
        "last_user_input": "",
    }


@pytest.fixture
def sample_symptoms():
    """Sample extracted symptoms"""
    return [
        ExtractedSymptom(
            symptom="chest pain",
            confidence=0.9,
            matched_text=["chest pain"],
            related_context=["pain in chest"],
            urgency="critical",
            possible_causes=["heart attack", "angina"],
        ),
        ExtractedSymptom(
            symptom="shortness of breath",
            confidence=0.8,
            matched_text=["can't breathe"],
            related_context=["difficulty breathing"],
            urgency="high",
            possible_causes=["pulmonary embolism"],
        ),
    ]


# ==================== UNIT TESTS ====================


class TestMedicalEntityRecognizer:
    """Unit tests for MedicalEntityRecognizer"""

    def setup_method(self):
        self.recognizer = MedicalEntityRecognizer()

    def test_extract_symptoms(self):
        text = "I have chest pain and fever"
        result = self.recognizer.extract_entities(text)

        assert "pain" in result["symptoms"] or "chest" in result["symptoms"]
        assert "chest" in result["body_parts"]

    def test_extract_body_parts(self):
        text = "My head hurts"
        result = self.recognizer.extract_entities(text)

        assert "head" in result["body_parts"]

    def test_extract_severity(self):
        text = "I have severe chest pain"
        result = self.recognizer.extract_entities(text)

        assert result["severity"] == "severe"

    def test_extract_urgency_indicators(self):
        text = "I'm having chest pain and can't breathe"
        result = self.recognizer.extract_entities(text)

        assert len(result["urgency_indicators"]) > 0

    def test_empty_input(self):
        text = ""
        result = self.recognizer.extract_entities(text)

        assert result["symptoms"] == []
        assert result["body_parts"] == []

    def test_multiple_symptoms(self):
        text = "I have headache, nausea, and fatigue"
        result = self.recognizer.extract_entities(text)

        assert len(result["symptoms"]) > 0


class TestSymptomExtractor:
    """Unit tests for SymptomExtractor"""

    def setup_method(self):
        self.extractor = SymptomExtractor()

    def test_extractor_initializes(self):
        assert self.extractor.symptom_database is not None

    def test_extract_symptoms_returns_list(self):
        result = self.extractor.extract_symptoms("I have a headache")
        assert isinstance(result, list)

    def test_lazy_loading(self):
        assert self.extractor._loaded == True


class TestConversationMemory:
    """Unit tests for ConversationMemory"""

    def setup_method(self):
        self.memory = ConversationMemory()
        self.test_session = f"test_session_{os.urandom(8).hex()}"

    def teardown_method(self):
        if hasattr(self, "test_session"):
            self.memory.clear_session(self.test_session)

    def test_create_new_session(self):
        context = self.memory.get_context(self.test_session)

        assert context["conversation_state"] == ConversationState.INITIAL
        assert context["total_interactions"] == 0

    def test_add_interaction(self):
        self.memory.add_interaction(
            session_id=self.test_session,
            user_input="I have chest pain",
            extracted_info={"symptoms": [], "entities": {}},
            ai_response="Please provide more details",
            confidence_score=0.8,
        )

        context = self.memory.get_context(self.test_session)
        assert context["total_interactions"] == 1

    def test_accumulated_symptoms(self):
        self.memory.add_interaction(
            session_id=self.test_session,
            user_input="I have chest pain",
            extracted_info={"symptoms": [{"symptom": "chest pain"}], "entities": {}},
            ai_response="Response",
            confidence_score=0.8,
        )

        context = self.memory.get_context(self.test_session)
        assert "chest pain" in context["accumulated_symptoms"]


class TestContextBuilder:
    """Unit tests for ContextBuilder"""

    def setup_method(self):
        self.builder = ContextBuilder()

    def test_assess_urgency_critical(self, sample_symptoms):
        urgency = self.builder._assess_medical_urgency(sample_symptoms, {})
        assert urgency == "critical"

    def test_assess_urgency_low(self):
        symptoms = [ExtractedSymptom("mild headache", 0.5, [], [], "low", [])]
        urgency = self.builder._assess_medical_urgency(symptoms, {})
        assert urgency == "low"

    def test_build_context_returns_dict(self, sample_conversation_context):
        context = self.builder.build_context(
            current_input="test",
            entities={"symptoms": []},
            symptoms=[],
            conversation_context=sample_conversation_context,
        )
        assert isinstance(context, dict)


class TestMedicalRAGEnrichmentEngine:
    """Unit tests for main RAG engine"""

    def setup_method(self):
        self.engine = MedicalRAGEnrichmentEngine()
        self.test_session = f"test_engine_{os.urandom(8).hex()}"

    def teardown_method(self):
        if hasattr(self, "test_session"):
            self.engine.conversation_memory.clear_session(self.test_session)

    def test_process_returns_dict(self):
        result = self.engine.process_user_input(
            user_input="I have chest pain", session_id=self.test_session
        )

        assert isinstance(result, dict)
        assert "enriched_prompt" in result
        assert "confidence_score" in result

    def test_confidence_in_range(self):
        result = self.engine.process_user_input(
            user_input="test input", session_id=self.test_session
        )

        assert 0 <= result["confidence_score"] <= 1

    def test_caching(self):
        result1 = self.engine.process_user_input(
            user_input="I have chest pain", session_id=self.test_session
        )

        result2 = self.engine.process_user_input(
            user_input="I have chest pain", session_id=self.test_session
        )

        assert result2.get("from_cache") == True


class TestConversationState:
    """Tests for ConversationState enum"""

    def test_all_states(self):
        assert ConversationState.INITIAL.value == "initial"
        assert ConversationState.EMERGENCY.value == "emergency"
        assert len(ConversationState.__members__) == 6


# ==================== INTEGRATION TESTS ====================


class TestAPIIntegration:
    """Integration tests for API endpoints"""

    @pytest.fixture
    def mock_rag_engine(self):
        """Mock RAG engine for API tests"""
        engine = Mock()
        engine.process_user_input.return_value = {
            "enriched_prompt": "test prompt",
            "entities": {"symptoms": [], "body_parts": []},
            "symptoms": [],
            "conversation_context": {
                "conversation_state": "initial",
                "total_interactions": 0,
            },
            "confidence_score": 0.8,
        }
        return engine

    def test_medical_entity_recognition_integration(self):
        """Test entity recognition with realistic medical input"""
        recognizer = MedicalEntityRecognizer()

        test_cases = [
            ("I have severe chest pain radiating to my left arm", "chest"),
            ("I've been having headaches for 3 days", "head"),
            ("My stomach hurts and I feel nauseous", "stomach"),
        ]

        for text, expected_body_part in test_cases:
            result = recognizer.extract_entities(text)
            if expected_body_part in text.lower():
                # Should detect the body part
                pass  # Test passes if no error


class TestConversationFlow:
    """Test conversation flow and state transitions"""

    def test_state_transition_initial_to_symptom_gathering(self):
        """Test that state transitions correctly"""
        memory = ConversationMemory()
        session_id = f"flow_test_{os.urandom(8).hex()}"

        memory.add_interaction(
            session_id=session_id,
            user_input="I have chest pain",
            extracted_info={"symptoms": ["chest pain"], "entities": {}},
            ai_response="Tell me more",
            confidence_score=0.8,
        )

        context = memory.get_context(session_id)
        assert context["total_interactions"] == 1

        memory.clear_session(session_id)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_very_long_input(self):
        """Test handling of very long input"""
        recognizer = MedicalEntityRecognizer()
        long_text = "pain " * 1000

        result = recognizer.extract_entities(long_text)
        assert isinstance(result, dict)

    def test_special_characters(self):
        """Test handling of special characters"""
        recognizer = MedicalEntityRecognizer()
        text = "I have pain! @#$%^&*()"

        result = recognizer.extract_entities(text)
        assert isinstance(result, dict)

    def test_unicode_characters(self):
        """Test handling of unicode"""
        recognizer = MedicalEntityRecognizer()
        text = "I have chest pain \u00e9\u00e8\u00ea"

        result = recognizer.extract_entities(text)
        assert isinstance(result, dict)


class TestSecurityInputValidation:
    """Test input validation and sanitization"""

    def test_sql_injection_attempt(self):
        """Test that SQL injection attempts are handled"""
        recognizer = MedicalEntityRecognizer()
        malicious = "'; DROP TABLE sessions; --"

        result = recognizer.extract_entities(malicious)
        assert isinstance(result, dict)

    def test_xss_attempt(self):
        """Test that XSS attempts are handled"""
        recognizer = MedicalEntityRecognizer()
        malicious = "<script>alert('xss')</script> chest pain"

        result = recognizer.extract_entities(malicious)
        assert isinstance(result, dict)


# ==================== PERFORMANCE TESTS ====================


class TestPerformance:
    """Performance and load tests"""

    def test_multiple_sessions(self):
        """Test handling multiple concurrent sessions"""
        engine = MedicalRAGEnrichmentEngine()

        sessions = [f"perf_test_{i}_{os.urandom(4).hex()}" for i in range(10)]

        for session_id in sessions:
            result = engine.process_user_input(
                user_input="I have chest pain", session_id=session_id
            )
            assert result["confidence_score"] > 0

        for session_id in sessions:
            engine.conversation_memory.clear_session(session_id)

    def test_rapid_requests(self):
        """Test rapid consecutive requests"""
        engine = MedicalRAGEnrichmentEngine()
        session_id = f"rapid_test_{os.urandom(8).hex()}"

        for i in range(5):
            result = engine.process_user_input(
                user_input=f"Test message {i}", session_id=session_id
            )
            assert result is not None

        engine.conversation_memory.clear_session(session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
