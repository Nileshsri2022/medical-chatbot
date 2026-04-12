"""
Type definitions and schemas for Medical RAG Chatbot
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class ConversationState(str, Enum):
    """Conversation state enumeration"""

    INITIAL = "initial"
    SYMPTOM_GATHERING = "symptom_gathering"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    TREATMENT_DISCUSSION = "treatment_discussion"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class ExtractedSymptom:
    """Extracted symptom with confidence and context"""

    symptom: str
    confidence: float
    matched_text: List[str] = field(default_factory=list)
    related_context: List[str] = field(default_factory=list)
    urgency: str = "low"
    possible_causes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationInteraction:
    """Single conversation interaction"""

    timestamp: str
    user_input: str
    extracted_symptoms: List[Dict[str, Any]]
    extracted_entities: Dict[str, List[str]]
    ai_response: str
    conversation_turn: int
    confidence_score: float


@dataclass
class ConversationContext:
    """Complete conversation context for a session"""

    session_id: str
    conversation_state: ConversationState
    total_interactions: int
    session_start_time: str
    conversation_history: List[Dict[str, Any]]
    accumulated_symptoms: set = field(default_factory=set)
    accumulated_conditions: set = field(default_factory=set)
    urgency_level: str = "low"
    last_user_input: str = ""


@dataclass
class ExtractedEntities:
    """Medical entities extracted from user input"""

    symptoms: List[str] = field(default_factory=list)
    body_parts: List[str] = field(default_factory=list)
    severity: str = "unknown"
    duration: str = ""
    urgency_indicators: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    medical_conditions: List[str] = field(default_factory=list)


@dataclass
class RAGResult:
    """Result from RAG processing pipeline"""

    enriched_prompt: str
    context: Dict[str, Any]
    entities: ExtractedEntities
    symptoms: List[ExtractedSymptom]
    conversation_context: ConversationContext
    confidence_score: float
    from_cache: bool = False


# Type aliases for better readability
EntityDict = Dict[str, List[str]]
SymptomList = List[Dict[str, Any]]
MedicalInfo = Dict[str, Any]
