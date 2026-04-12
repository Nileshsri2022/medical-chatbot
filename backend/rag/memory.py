"""
Conversation Memory Module
Manages session state and history with SQLite persistence
"""

import os
import json
import sqlite3
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ConversationState(Enum):
    INITIAL = "initial"
    SYMPTOM_GATHERING = "symptom_gathering"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    TREATMENT_DISCUSSION = "treatment_discussion"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"


@dataclass
class ConversationInteraction:
    timestamp: str
    user_input: str
    extracted_symptoms: List[Dict]
    extracted_entities: Dict
    ai_response: str
    conversation_turn: int
    confidence_score: float


class ConversationMemory:
    """Advanced conversation memory with medical context tracking backed by SQLite"""

    def __init__(self, max_history_length: int = 10):
        self.sessions: Dict[str, Dict] = {}
        self.max_history_length = max_history_length
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(base_dir, "data", "medical_chats.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._load_all()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sessions 
                (session_id TEXT PRIMARY KEY, data TEXT)"""
            )

    def _load_all(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT session_id, data FROM sessions")
            for row in cursor:
                try:
                    data = json.loads(row[1])
                    data["accumulated_symptoms"] = set(
                        data.get("accumulated_symptoms", [])
                    )
                    data["accumulated_conditions"] = set(
                        data.get("accumulated_conditions", [])
                    )
                    data["accumulated_medications"] = set(
                        data.get("accumulated_medications", [])
                    )
                    if "conversation_state" in data and isinstance(
                        data["conversation_state"], str
                    ):
                        data["conversation_state"] = ConversationState(
                            data["conversation_state"]
                        )
                    self.sessions[row[0]] = data
                except Exception:
                    pass

    def _save_session(self, session_id: str):
        sess = self.sessions[session_id].copy()
        sess["accumulated_symptoms"] = list(sess["accumulated_symptoms"])
        sess["accumulated_conditions"] = list(sess["accumulated_conditions"])
        sess["accumulated_medications"] = list(sess["accumulated_medications"])
        sess["conversation_state"] = (
            sess["conversation_state"].value
            if hasattr(sess["conversation_state"], "value")
            else "initial"
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, data) VALUES (?, ?)",
                (session_id, json.dumps(sess)),
            )

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def add_interaction(
        self,
        session_id: str,
        user_input: str,
        extracted_info: Dict,
        ai_response: str,
        confidence_score: float = 0.8,
    ):
        """Store interaction with comprehensive medical context"""

        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_new_session()

        session = self.sessions[session_id]

        interaction = ConversationInteraction(
            timestamp=datetime.datetime.now().isoformat(),
            user_input=user_input,
            extracted_symptoms=extracted_info.get("symptoms", []),
            extracted_entities=extracted_info.get("entities", {}),
            ai_response=ai_response,
            conversation_turn=len(session["conversation_history"]) + 1,
            confidence_score=confidence_score,
        )

        session["conversation_history"].append(asdict(interaction))

        if len(session["conversation_history"]) > self.max_history_length:
            session["conversation_history"] = session["conversation_history"][
                -self.max_history_length :
            ]

        self._update_accumulated_info(session, extracted_info)
        self._update_conversation_state(session, extracted_info)

        self._save_session(session_id)

    def get_context(self, session_id: str) -> Dict:
        """Get comprehensive conversation context"""

        if session_id not in self.sessions:
            return self._create_new_session_context()

        session = self.sessions[session_id]

        return {
            "conversation_history": session["conversation_history"][-5:],
            "accumulated_symptoms": list(session["accumulated_symptoms"]),
            "accumulated_conditions": list(session["accumulated_conditions"]),
            "accumulated_medications": list(session["accumulated_medications"]),
            "patient_profile": session.get("patient_profile", {}),
            "conversation_state": session["conversation_state"].value,
            "conversation_summary": self._generate_conversation_summary(session),
            "urgency_level": session.get("urgency_level", "low"),
            "last_topic": session.get("last_topic"),
            "session_start_time": session.get("session_start_time"),
            "total_interactions": len(session["conversation_history"]),
        }

    def _create_new_session(self) -> Dict:
        """Create new session with default values"""
        return {
            "conversation_history": [],
            "accumulated_symptoms": set(),
            "accumulated_conditions": set(),
            "accumulated_medications": set(),
            "patient_profile": {},
            "conversation_state": ConversationState.INITIAL,
            "urgency_level": "low",
            "last_topic": None,
            "session_start_time": datetime.datetime.now().isoformat(),
        }

    def _create_new_session_context(self) -> Dict:
        """Create context for new session"""
        return {
            "conversation_history": [],
            "accumulated_symptoms": [],
            "accumulated_conditions": [],
            "accumulated_medications": [],
            "patient_profile": {},
            "conversation_state": "initial",
            "conversation_summary": "New conversation started",
            "urgency_level": "low",
            "last_topic": None,
            "session_start_time": datetime.datetime.now().isoformat(),
            "total_interactions": 0,
        }

    def _update_accumulated_info(self, session: Dict, extracted_info: Dict):
        """Update accumulated medical information"""
        if "symptoms" in extracted_info:
            for symptom_obj in extracted_info["symptoms"]:
                if isinstance(symptom_obj, dict):
                    session["accumulated_symptoms"].add(symptom_obj.get("symptom", ""))
                else:
                    session["accumulated_symptoms"].add(str(symptom_obj))

        entities = extracted_info.get("entities", {})
        if "conditions" in entities:
            session["accumulated_conditions"].update(entities["conditions"])
        if "medications" in entities:
            session["accumulated_medications"].update(entities["medications"])

    def _update_conversation_state(self, session: Dict, extracted_info: Dict):
        """Update conversation state based on medical content"""
        current_state = session["conversation_state"]
        symptoms = extracted_info.get("symptoms", [])
        entities = extracted_info.get("entities", {})

        urgency_indicators = entities.get("urgency_indicators", [])
        if urgency_indicators or any(
            s.get("urgency") == "critical" for s in symptoms if isinstance(s, dict)
        ):
            session["conversation_state"] = ConversationState.EMERGENCY
            session["urgency_level"] = "critical"
            return

        if current_state == ConversationState.INITIAL:
            if symptoms:
                session["conversation_state"] = ConversationState.SYMPTOM_GATHERING
        elif current_state == ConversationState.SYMPTOM_GATHERING:
            if len(session["accumulated_symptoms"]) >= 2:
                session["conversation_state"] = ConversationState.SYMPTOM_ANALYSIS
        elif current_state == ConversationState.SYMPTOM_ANALYSIS:
            if "medications" in entities and entities["medications"]:
                session["conversation_state"] = ConversationState.TREATMENT_DISCUSSION

    def _generate_conversation_summary(self, session: Dict) -> str:
        """Generate a summary of the conversation so far"""
        history_length = len(session["conversation_history"])
        symptom_count = len(session["accumulated_symptoms"])
        condition_count = len(session["accumulated_conditions"])

        if history_length == 0:
            return "New conversation - no previous interactions"

        summary_parts = [
            f"Conversation with {history_length} interactions",
            f"{symptom_count} symptoms discussed"
            if symptom_count > 0
            else "No symptoms mentioned yet",
            f"{condition_count} conditions mentioned"
            if condition_count > 0
            else "No specific conditions discussed",
        ]

        if session.get("urgency_level") != "low":
            summary_parts.append(f"Urgency level: {session['urgency_level']}")

        return ". ".join(summary_parts) + "."

    @property
    def active_sessions_count(self) -> int:
        return len(self.sessions)
