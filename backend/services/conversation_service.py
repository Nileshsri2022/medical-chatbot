"""
Conversation Memory Service
Manages conversation state and history for medical chatbot sessions
"""

import os
import json
import sqlite3
from typing import Dict, Optional, List, Any
from datetime import datetime
from ..models.schemas import ConversationState, ConversationContext


class ConversationService:
    """Service for managing conversation memory with SQLite persistence"""

    def __init__(self, max_history_length: int = 10) -> None:
        self.max_history_length = max_history_length
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "medical_chats.db"
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
        self._load_sessions()

    def _init_database(self) -> None:
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions 
                (session_id TEXT PRIMARY KEY, data TEXT)
            """)

    def _load_sessions(self) -> None:
        """Load all sessions from database"""
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
                    if "conversation_state" in data and isinstance(
                        data["conversation_state"], str
                    ):
                        data["conversation_state"] = ConversationState(
                            data["conversation_state"]
                        )
                    self.sessions[row[0]] = data
                except Exception:
                    pass

    def _save_session(self, session_id: str) -> None:
        """Persist session to database"""
        session = self.sessions[session_id].copy()
        session["accumulated_symptoms"] = list(session["accumulated_symptoms"])
        session["accumulated_conditions"] = list(session["accumulated_conditions"])
        session["conversation_state"] = (
            session["conversation_state"].value
            if hasattr(session["conversation_state"], "value")
            else "initial"
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, data) VALUES (?, ?)",
                (session_id, json.dumps(session)),
            )

    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation context for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_new_session()
        return self.sessions[session_id]

    def add_interaction(
        self,
        session_id: str,
        user_input: str,
        extracted_info: Dict[str, Any],
        ai_response: str,
        confidence_score: float = 0.8,
    ) -> None:
        """Add a new interaction to conversation history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_new_session()

        session = self.sessions[session_id]
        session["total_interactions"] += 1
        session["last_user_input"] = user_input

        # Add to accumulated symptoms
        if "symptoms" in extracted_info:
            for symptom in extracted_info["symptoms"]:
                if isinstance(symptom, dict) and "symptom" in symptom:
                    session["accumulated_symptoms"].add(symptom["symptom"])
                elif isinstance(symptom, str):
                    session["accumulated_symptoms"].add(symptom)

        # Add to accumulated conditions
        if (
            "entities" in extracted_info
            and "medical_conditions" in extracted_info["entities"]
        ):
            session["accumulated_conditions"].update(
                extracted_info["entities"]["medical_conditions"]
            )

        # Add to conversation history
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "confidence_score": confidence_score,
            "turn": session["total_interactions"],
            "extracted_info": extracted_info,
        }
        session["conversation_history"].append(interaction)

        # Trim history if needed
        if len(session["conversation_history"]) > self.max_history_length:
            session["conversation_history"] = session["conversation_history"][
                -self.max_history_length :
            ]

        # Update conversation state
        self._update_state(session, user_input, extracted_info)

        self._save_session(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear a session from memory and database"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def _create_new_session(self) -> Dict[str, Any]:
        """Create a new session structure"""
        return {
            "session_id": "",
            "conversation_state": ConversationState.INITIAL,
            "total_interactions": 0,
            "session_start_time": datetime.now().isoformat(),
            "conversation_history": [],
            "accumulated_symptoms": set(),
            "accumulated_conditions": set(),
            "urgency_level": "low",
            "last_user_input": "",
        }

    def _update_state(
        self, session: Dict[str, Any], user_input: str, extracted_info: Dict[str, Any]
    ) -> None:
        """Update conversation state based on interaction"""
        state = session["conversation_state"]
        user_lower = user_input.lower()

        # Check for emergency indicators
        emergency_words = [
            "chest pain",
            "can't breathe",
            "emergency",
            "severe",
            "dying",
        ]
        if any(word in user_lower for word in emergency_words):
            session["conversation_state"] = ConversationState.EMERGENCY
            session["urgency_level"] = "critical"
            return

        # State machine transition logic
        if state == ConversationState.INITIAL:
            if extracted_info.get("symptoms"):
                session["conversation_state"] = ConversationState.SYMPTOM_GATHERING
        elif state == ConversationState.SYMPTOM_GATHERING:
            if session["total_interactions"] >= 3:
                session["conversation_state"] = ConversationState.SYMPTOM_ANALYSIS
        elif state == ConversationState.SYMPTOM_ANALYSIS:
            treatment_words = ["treatment", "medicine", "medication", "remedy", "help"]
            if any(word in user_lower for word in treatment_words):
                session["conversation_state"] = ConversationState.TREATMENT_DISCUSSION
        elif state == ConversationState.TREATMENT_DISCUSSION:
            session["conversation_state"] = ConversationState.FOLLOW_UP

    @property
    def active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session summary information"""
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "total_interactions": session["total_interactions"],
            "conversation_state": session["conversation_state"].value,
            "accumulated_symptoms_count": len(session["accumulated_symptoms"]),
            "urgency_level": session["urgency_level"],
        }
