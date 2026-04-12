"""
Unit tests for Medical RAG Engine
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.medical_rag_engine import (
    MedicalEntityRecognizer,
    SymptomExtractor,
    ConversationMemory,
    ContextBuilder,
    MedicalRAGEnrichmentEngine,
    ConversationState,
    ExtractedSymptom,
)


class TestMedicalEntityRecognizer:
    """Tests for MedicalEntityRecognizer"""

    def setup_method(self):
        self.recognizer = MedicalEntityRecognizer()

    def test_extract_symptoms(self):
        text = "I have chest pain and fever"
        result = self.recognizer.extract_entities(text)

        assert "pain" in result["symptoms"] or "chest" in result["symptoms"], (
            "Should extract symptoms"
        )
        assert "chest" in result["body_parts"], "Should extract body parts"

    def test_extract_body_parts(self):
        text = "My head hurts"
        result = self.recognizer.extract_entities(text)

        assert "head" in result["body_parts"], "Should extract head"

    def test_extract_severity(self):
        text = "I have severe chest pain"
        result = self.recognizer.extract_entities(text)

        assert result["severity"] == "severe", "Should detect severe"

    def test_extract_urgency_indicators(self):
        text = "I'm having chest pain and can't breathe"
        result = self.recognizer.extract_entities(text)

        assert len(result["urgency_indicators"]) > 0, "Should detect urgency"


class TestSymptomExtractor:
    """Tests for SymptomExtractor"""

    def setup_method(self):
        self.extractor = SymptomExtractor()

    def test_extractor_initializes(self):
        """Test that extractor initializes with database"""
        assert self.extractor.symptom_database is not None

    def test_extract_symptoms_returns_list(self):
        result = self.extractor.extract_symptoms("I have a headache")

        assert isinstance(result, list), "Should return list"


class TestConversationMemory:
    """Tests for ConversationMemory"""

    def setup_method(self):
        self.memory = ConversationMemory()

    def test_create_new_session(self):
        context = self.memory.get_context("test_session")

        assert context["conversation_state"] == "initial"
        assert context["total_interactions"] == 0

    def test_add_interaction(self):
        self.memory.add_interaction(
            session_id="test_session",
            user_input="I have chest pain",
            extracted_info={"symptoms": [], "entities": {}},
            ai_response="Please provide more details",
            confidence_score=0.8,
        )

        context = self.memory.get_context("test_session")
        assert context["total_interactions"] == 1

    def test_accumulated_symptoms(self):
        self.memory.add_interaction(
            session_id="test_session",
            user_input="I have chest pain",
            extracted_info={"symptoms": [{"symptom": "chest pain"}], "entities": {}},
            ai_response="Response",
            confidence_score=0.8,
        )

        context = self.memory.get_context("test_session")
        assert "chest pain" in context["accumulated_symptoms"]


class TestContextBuilder:
    """Tests for ContextBuilder"""

    def setup_method(self):
        self.builder = ContextBuilder()

    def test_assess_medical_urgency_critical(self):
        symptoms = [
            ExtractedSymptom(
                symptom="chest pain",
                confidence=0.9,
                matched_text=[],
                related_context=[],
                urgency="critical",
                possible_causes=[],
            )
        ]

        urgency = self.builder._assess_medical_urgency(symptoms, {})

        assert urgency == "critical", "Should detect critical urgency"

    def test_assess_medical_urgency_low(self):
        symptoms = [
            ExtractedSymptom(
                symptom="mild headache",
                confidence=0.5,
                matched_text=[],
                related_context=[],
                urgency="low",
                possible_causes=[],
            )
        ]

        urgency = self.builder._assess_medical_urgency(symptoms, {})

        assert urgency == "low", "Should detect low urgency"


class TestMedicalRAGEnrichmentEngine:
    """Tests for the main RAG engine"""

    def setup_method(self):
        self.engine = MedicalRAGEnrichmentEngine()

    def test_process_user_input_returns_dict(self):
        result = self.engine.process_user_input(
            user_input="I have chest pain", session_id="test_session"
        )

        assert isinstance(result, dict)
        assert "enriched_prompt" in result
        assert "entities" in result
        assert "symptoms" in result
        assert "confidence_score" in result

    def test_confidence_score_in_range(self):
        result = self.engine.process_user_input(
            user_input="test input", session_id="test_session"
        )

        assert 0 <= result["confidence_score"] <= 1, (
            "Confidence should be between 0 and 1"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
