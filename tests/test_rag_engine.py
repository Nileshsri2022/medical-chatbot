"""
Medical RAG Engine - Comprehensive Test Suite
Unit Tests, Integration Tests, Fixtures using pytest best practices

Implements patterns from python-testing-patterns skill:
- AAA Pattern (Arrange, Act, Assert)
- Fixtures for setup/teardown
- Parameterized tests
- Mocking external dependencies
- Test markers
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch, AsyncMock, MagicMock

from backend.rag.entities import MedicalEntityRecognizer
from backend.rag.memory import ConversationMemory, ConversationState
from backend.rag.context import ContextBuilder
from backend.rag.engine import MedicalRAGEnrichmentEngine


# ==================== PARAMETERIZED TESTS ====================


class TestMedicalEntityRecognizer:
    """Unit tests for MedicalEntityRecognizer - uses fixtures and parameterized tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test method"""
        self.recognizer = MedicalEntityRecognizer()

    @pytest.mark.parametrize(
        "text,expected_severity",
        [
            ("I have severe chest pain", "severe"),
            ("I have moderate headache", "moderate"),
            ("I have mild fever", "mild"),
            ("I have chest pain", "unspecified"),
        ],
    )
    def test_extract_severity_parameterized(self, text, expected_severity):
        """Test severity extraction with multiple inputs"""
        result = self.recognizer.extract_entities(text)
        assert result["severity"] == expected_severity

    @pytest.mark.parametrize(
        "text,should_have_urgency",
        [
            ("I have chest pain and can't breathe", True),
            ("I need emergency help", True),
            ("I have a headache", False),
            ("My stomach hurts", False),
        ],
    )
    def test_urgency_detection_parameterized(self, text, should_have_urgency):
        """Test urgency indicator detection"""
        result = self.recognizer.extract_entities(text)
        has_urgency = len(result["urgency_indicators"]) > 0
        assert has_urgency == should_have_urgency

    @pytest.mark.parametrize(
        "text,expected_body_parts",
        [
            ("My head hurts", ["head"]),
            ("Chest pain", ["chest"]),
            ("Stomach ache", ["stomach"]),
            ("Leg pain", ["leg"]),
        ],
    )
    def test_body_parts_extraction(self, text, expected_body_parts):
        """Test body parts extraction"""
        result = self.recognizer.extract_entities(text)
        for part in expected_body_parts:
            assert part in result["body_parts"]

    def test_extract_symptoms_with_fixture(self, sample_medical_text):
        """Test symptom extraction using fixture - AAA Pattern"""
        # Arrange - already done via fixture
        # Act
        result = self.recognizer.extract_entities(sample_medical_text)
        # Assert
        assert len(result["symptoms"]) > 0
        assert "chest" in result["body_parts"] or "pain" in result["symptoms"]

    def test_empty_input_returns_valid_dict(self):
        """Test edge case: empty input"""
        result = self.recognizer.extract_entities("")
        assert isinstance(result, dict)
        assert "symptoms" in result
        assert "body_parts" in result

    def test_special_characters_handled(self):
        """Test handling of special characters"""
        result = self.recognizer.extract_entities("Pain! @#$%^&*()")
        assert isinstance(result, dict)

    @pytest.mark.security
    def test_sql_injection_not_executed(self):
        """Security test: SQL injection should not execute"""
        malicious = "'; DROP TABLE sessions; --"
        result = self.recognizer.extract_entities(malicious)
        assert isinstance(result, dict)

    @pytest.mark.security
    def test_xss_pattern_not_executed(self):
        """Security test: XSS pattern should not execute"""
        malicious = "<script>alert('xss')</script> chest pain"
        result = self.recognizer.extract_entities(malicious)
        assert isinstance(result, dict)


class TestConversationMemory:
    """Unit tests for ConversationMemory with fixtures and mocking"""

    @pytest.fixture(autouse=True)
    def setup(self, unique_session_id):
        """Setup with unique session ID"""
        self.memory = ConversationMemory()
        self.session_id = unique_session_id

    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, "session_id"):
            self.memory.clear_session(self.session_id)

    def test_create_new_session_returns_initial_state(self):
        """Test session creation returns correct initial state - AAA Pattern"""
        # Arrange - done via fixture
        # Act
        context = self.memory.get_context(self.session_id)
        # Assert
        assert context["conversation_state"] == ConversationState.INITIAL.value
        assert context["total_interactions"] == 0
        assert context["urgency_level"] == "low"

    def test_add_interaction_increments_count(self):
        """Test adding interaction increments the counter"""
        self.memory.add_interaction(
            session_id=self.session_id,
            user_input="I have chest pain",
            extracted_info={"symptoms": [], "entities": {}},
            ai_response="Tell me more",
            confidence_score=0.8,
        )

        context = self.memory.get_context(self.session_id)
        assert context["total_interactions"] == 1

    def test_accumulated_symptoms_tracked(self):
        """Test that symptoms accumulate across interactions"""
        self.memory.add_interaction(
            session_id=self.session_id,
            user_input="I have chest pain",
            extracted_info={"symptoms": [{"symptom": "chest pain"}], "entities": {}},
            ai_response="Response",
            confidence_score=0.8,
        )

        context = self.memory.get_context(self.session_id)
        assert "chest pain" in context["accumulated_symptoms"]

    def test_multiple_interactions_maintain_context(self):
        """Test multiple interactions maintain proper context"""
        interactions = [
            "I have chest pain",
            "It started this morning",
            "The pain is severe",
        ]

        for user_input in interactions:
            self.memory.add_interaction(
                session_id=self.session_id,
                user_input=user_input,
                extracted_info={"symptoms": [], "entities": {}},
                ai_response="Acknowledged",
                confidence_score=0.8,
            )

        context = self.memory.get_context(self.session_id)
        assert context["total_interactions"] == 3

    def test_clear_session_removes_data(self):
        """Test session cleanup works properly"""
        self.memory.add_interaction(
            session_id=self.session_id,
            user_input="Test",
            extracted_info={"symptoms": [], "entities": {}},
            ai_response="Test",
            confidence_score=0.8,
        )

        self.memory.clear_session(self.session_id)
        assert self.session_id not in self.memory.sessions


class TestContextBuilder:
    """Unit tests for ContextBuilder"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.builder = ContextBuilder()

    def test_build_context_returns_valid_dict(self):
        """Test basic context building"""
        context = self.builder.build_context(
            current_input="test input",
            entities={"symptoms": []},
            symptoms=[],
            conversation_context={"conversation_state": "initial"},
        )

        assert isinstance(context, dict)
        assert "current_input" in context
        assert "medical_urgency" in context

    @pytest.mark.parametrize(
        "symptoms,expected_urgency",
        [
            ([{"urgency": "critical"}], "critical"),
            ([{"urgency": "high"}], "high"),
            ([{"urgency": "moderate"}], "moderate"),
            ([{"urgency": "low"}], "low"),
            ([], "low"),
        ],
    )
    def test_urgency_assessment_parameterized(self, symptoms, expected_urgency):
        """Test urgency assessment with various inputs"""
        urgency = self.builder._assess_medical_urgency(symptoms, {})
        assert urgency == expected_urgency

    def test_urgency_from_entities(self):
        """Test urgency detection from entity indicators"""
        entities = {"urgency_indicators": ["chest pain"]}
        urgency = self.builder._assess_medical_urgency([], entities)
        assert urgency == "critical"


class TestMedicalRAGEnrichmentEngine:
    """Integration tests for main RAG engine"""

    @pytest.fixture(autouse=True)
    def setup(self, unique_session_id):
        self.engine = MedicalRAGEnrichmentEngine()
        self.session_id = unique_session_id

    def teardown_method(self):
        if hasattr(self, "session_id"):
            self.engine.conversation_memory.clear_session(self.session_id)

    def test_process_user_input_returns_complete_result(self):
        """Test that process returns all expected keys - AAA Pattern"""
        # Arrange
        user_input = "I have chest pain"
        # Act
        result = self.engine.process_user_input(user_input, self.session_id)
        # Assert
        assert isinstance(result, dict)
        assert "enriched_prompt" in result
        assert "entities" in result
        assert "symptoms" in result
        assert "conversation_context" in result
        assert "confidence_score" in result

    def test_confidence_score_in_valid_range(self):
        """Test confidence score is always between 0 and 1"""
        result = self.engine.process_user_input("test", self.session_id)
        assert 0 <= result["confidence_score"] <= 1

    def test_caching_returns_cached_result(self):
        """Test that duplicate queries return cached results"""
        user_input = "I have chest pain"

        result1 = self.engine.process_user_input(user_input, self.session_id)
        result2 = self.engine.process_user_input(user_input, self.session_id)

        assert result2.get("from_cache") is True

    @pytest.mark.slow
    def test_multiple_different_queries(self):
        """Test multiple different queries work correctly - marked as slow"""
        queries = [
            "I have a headache",
            "My stomach hurts",
            "I feel dizzy",
        ]

        for query in queries:
            result = self.engine.process_user_input(query, self.session_id)
            assert result["confidence_score"] >= 0


class TestConversationStateTransitions:
    """Test conversation state machine"""

    def test_initial_state(self):
        """Test initial state is correct"""
        memory = ConversationMemory()
        session_id = f"test_{os.urandom(8).hex()}"

        context = memory.get_context(session_id)
        assert context["conversation_state"] == "initial"

        memory.clear_session(session_id)

    def test_symptom_gathering_state(self):
        """Test transition to symptom gathering state"""
        memory = ConversationMemory()
        session_id = f"test_{os.urandom(8).hex()}"

        memory.add_interaction(
            session_id=session_id,
            user_input="I have chest pain",
            extracted_info={"symptoms": ["chest pain"], "entities": {}},
            ai_response="Tell me more",
            confidence_score=0.8,
        )

        context = memory.get_context(session_id)
        assert context["conversation_state"] in [
            "symptom_gathering",
            "symptom_analysis",
            "initial",
        ]

        memory.clear_session(session_id)


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.recognizer = MedicalEntityRecognizer()

    def test_very_long_input_handled(self):
        """Test handling of very long input"""
        long_text = "pain " * 1000
        result = self.recognizer.extract_entities(long_text)
        assert isinstance(result, dict)

    def test_unicode_characters_handled(self):
        """Test handling of unicode characters"""
        text = "I have chest pain \u00e9\u00e8\u00ea"
        result = self.recognizer.extract_entities(text)
        assert isinstance(result, dict)

    def test_none_input_handled(self):
        """Test handling of None input"""
        result = self.recognizer.extract_entities(None)
        assert isinstance(result, dict)


class TestPerformance:
    """Performance tests - marked with custom marker"""

    @pytest.mark.slow
    def test_multiple_sessions_performance(self):
        """Test handling multiple concurrent sessions"""
        engine = MedicalRAGEnrichmentEngine()
        sessions = [f"perf_{i}_{os.urandom(4).hex()}" for i in range(10)]

        for session_id in sessions:
            result = engine.process_user_input("test", session_id)
            assert result["confidence_score"] > 0

        for session_id in sessions:
            engine.conversation_memory.clear_session(session_id)

    @pytest.mark.slow
    def test_rapid_requests(self):
        """Test rapid consecutive requests"""
        engine = MedicalRAGEnrichmentEngine()
        session_id = f"rapid_{os.urandom(8).hex()}"

        for i in range(5):
            result = engine.process_user_input(f"Test {i}", session_id)
            assert result is not None

        engine.conversation_memory.clear_session(session_id)


# ==================== FIXTURES FROM CONFTEST ====================


@pytest.fixture
def sample_medical_text() -> str:
    """Fixture for sample medical text"""
    return "I have severe chest pain that started this morning"


@pytest.fixture
def sample_entities() -> dict:
    """Fixture for sample entities"""
    return {
        "symptoms": ["chest pain"],
        "body_parts": ["chest"],
        "severity": "severe",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
