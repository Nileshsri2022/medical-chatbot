#!/usr/bin/env python3
"""
Medical RAG Engine - Core Components
Advanced conversation memory, medical entity recognition, and context building
"""

import os
import re
import json
import datetime
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor


class ConversationState(Enum):
    INITIAL = "initial"
    SYMPTOM_GATHERING = "symptom_gathering"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    TREATMENT_DISCUSSION = "treatment_discussion"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"


@dataclass
class ExtractedSymptom:
    symptom: str
    confidence: float
    matched_text: List[str]
    related_context: List[str]
    urgency: str
    possible_causes: List[str]


@dataclass
class ConversationInteraction:
    timestamp: str
    user_input: str
    extracted_symptoms: List[Dict]
    extracted_entities: Dict
    ai_response: str
    conversation_turn: int
    confidence_score: float


class MedicalEntityRecognizer:
    """Advanced medical entity recognition using pattern matching and medical knowledge"""

    def __init__(self):
        self.medical_patterns = {
            "symptoms": [
                # Pain patterns
                r"\b(pain|ache|hurt|sore|tender|burning|throbbing|sharp|dull|stabbing|cramping)\b",
                # Fever and temperature
                r"\b(fever|temperature|hot|chills|sweating|feverish|burning up)\b",
                # Gastrointestinal
                r"\b(nausea|vomiting|sick|queasy|throwing up|stomach ache|belly pain)\b",
                # Neurological
                r"\b(headache|migraine|head pain|dizzy|dizziness|lightheaded|vertigo)\b",
                # Respiratory
                r"\b(shortness of breath|breathless|gasping|wheezing|cough|coughing)\b",
                # Cardiac
                r"\b(chest pain|chest tightness|chest pressure|heart pain|palpitations)\b",
                # General
                r"\b(fatigue|tired|exhausted|weakness|weak|swelling|swollen|bloated)\b",
            ],
            "body_parts": [
                r"\b(head|neck|shoulder|arm|elbow|wrist|hand|finger|thumb)\b",
                r"\b(chest|back|spine|abdomen|stomach|belly|pelvis)\b",
                r"\b(hip|leg|knee|ankle|foot|toe|thigh|calf)\b",
                r"\b(heart|lung|liver|kidney|brain|throat|nose|ear|eye)\b",
            ],
            "conditions": [
                r"\b(diabetes|hypertension|asthma|arthritis|depression|anxiety)\b",
                r"\b(covid|flu|cold|pneumonia|bronchitis|infection)\b",
                r"\b(cancer|tumor|stroke|heart attack|migraine)\b",
            ],
            "medications": [
                r"\b(ibuprofen|paracetamol|aspirin|acetaminophen|tylenol|advil)\b",
                r"\b(antibiotic|insulin|inhaler|steroid|medication|medicine|pill|tablet)\b",
            ],
            "temporal": [
                r"\b(today|yesterday|last week|few days|hours ago|minutes ago)\b",
                r"\b(sudden|gradual|chronic|acute|persistent|intermittent)\b",
                r"\b(morning|evening|night|during sleep|after eating)\b",
            ],
            "severity": [
                r"\b(severe|excruciating|unbearable|intense|terrible|awful|extreme)\b",
                r"\b(moderate|noticeable|uncomfortable|bothersome|manageable)\b",
                r"\b(mild|slight|little|minor|small|barely noticeable)\b",
            ],
        }

    def extract_entities(self, text: str) -> Dict:
        """Extract comprehensive medical entities from text"""
        text_lower = text.lower()
        entities = {
            "symptoms": [],
            "body_parts": [],
            "conditions": [],
            "medications": [],
            "temporal": [],
            "severity": self._extract_severity(text_lower),
            "duration": self._extract_duration(text_lower),
            "urgency_indicators": self._extract_urgency_indicators(text_lower),
        }

        for category, patterns in self.medical_patterns.items():
            if category != "severity":  # Already handled separately
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    entities[category].extend(matches)

        # Remove duplicates while preserving order
        for key in entities:
            if isinstance(entities[key], list):
                entities[key] = list(dict.fromkeys(entities[key]))

        return entities

    def _extract_severity(self, text: str) -> str:
        """Extract pain/symptom severity indicators"""
        severity_patterns = {
            "severe": r"\b(severe|excruciating|unbearable|intense|terrible|awful|extreme|worst)\b",
            "moderate": r"\b(moderate|noticeable|uncomfortable|bothersome|manageable|medium)\b",
            "mild": r"\b(mild|slight|little|minor|small|barely|light)\b",
        }

        for severity, pattern in severity_patterns.items():
            if re.search(pattern, text):
                return severity
        return "unspecified"

    def _extract_duration(self, text: str) -> str:
        """Extract symptom duration"""
        duration_patterns = {
            "acute": r"\b(sudden|minutes|hour|hours|today|just now|right now)\b",
            "subacute": r"\b(days|few days|week|yesterday)\b",
            "chronic": r"\b(weeks|months|years|long time|always|chronic)\b",
        }

        for duration, pattern in duration_patterns.items():
            if re.search(pattern, text):
                return duration
        return "unspecified"

    def _extract_urgency_indicators(self, text: str) -> List[str]:
        """Extract urgency indicators"""
        urgency_patterns = [
            r"\b(emergency|urgent|immediate|help|911|hospital|emergency room)\b",
            r"\b(can't breathe|chest pain|heart attack|stroke|bleeding)\b",
            r"\b(severe pain|unbearable|excruciating|passing out)\b",
        ]

        indicators = []
        for pattern in urgency_patterns:
            matches = re.findall(pattern, text)
            indicators.extend(matches)

        return list(set(indicators))


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class SymptomExtractor:
    """Advanced symptom extraction powered by Kaggle Symptom2Disease dataset"""

    def __init__(self):
        self.symptom_database = {}
        self.vectorizer = None
        self.disease_texts = []
        self.disease_labels = []
        self.is_trained = False
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the dataset on first use"""
        if self._loaded:
            return
        self._loaded = True
        self.load_kaggle_dataset()

    @property
    def symptom_count(self) -> int:
        """Get count of loaded symptoms (triggers lazy load)"""
        self._ensure_loaded()
        return len(self.symptom_database)

    def load_kaggle_dataset(self):
        """Load and train on the Kaggle Symptom2Disease dataset"""
        import os

        csv_path = os.path.join(
            os.path.dirname(__file__), "data", "kaggle_symptom2disease.csv"
        )

        try:
            # Prefer EMR dataset if it exists, fallback to NLP dataset
            emr_path = os.path.join(
                os.path.dirname(__file__), "data", "sample_emr_dataset.csv"
            )

            if os.path.exists(emr_path):
                print("🔄 Detected EMR tabular dataset. Serializing to text...")
                df = pd.read_csv(emr_path)

                # Auto-detect diagnosis column based on Kaggle/MIMIC formats
                if "apacheadmissiondx" in df.columns:
                    label_col = "apacheadmissiondx"
                elif "Diagnosis_Label" in df.columns:
                    label_col = "Diagnosis_Label"
                else:
                    label_col = df.columns[0]

                # Drop rows where diagnosis is empty
                df = df.dropna(subset=[label_col])

                for disease, group in df.groupby(label_col):
                    serialized_texts = []
                    for _, row in group.iterrows():
                        # Convert the tabular row into a natural language paragraph
                        features = []
                        for col in df.columns:
                            if col != label_col and pd.notna(row[col]):
                                features.append(f"{col} is {row[col]}")
                        paragraph = f"Patient presents with: {', '.join(features)}."
                        serialized_texts.append(paragraph)

                    combined_text = " ".join(serialized_texts)
                    self.disease_labels.append(disease)
                    self.disease_texts.append(combined_text)

                    self.symptom_database[disease] = {
                        "text": combined_text,
                        "urgency": "high"
                        if disease in ["Heart Disease"]
                        else "moderate",
                        "follow_up_questions": [
                            f"Do you have a history of {disease}?",
                            "Have these symptoms worsened recently?",
                        ],
                        "common_causes": [disease],
                    }

                self.tfidf_matrix = self.vectorizer.fit_transform(self.disease_texts)
                self.is_trained = True
                print(
                    f"✅ EMR Dataset Serialized & Loaded: {len(self.disease_labels)} diseases"
                )

            elif os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                # Group by disease to build comprehensive symptom profiles
                for disease, group in df.groupby("label"):
                    combined_text = " ".join(group["text"].tolist())
                    self.disease_labels.append(disease)
                    self.disease_texts.append(combined_text)

                    # Generate urgency and followups based on disease
                    urgency = (
                        "high"
                        if disease
                        in ["Heart attack", "Covid", "Hypertension", "Asthma"]
                        else "moderate"
                    )
                    if disease in ["Common Cold", "Acne", "Allergy"]:
                        urgency = "low"

                    self.symptom_database[disease] = {
                        "text": combined_text,
                        "urgency": urgency,
                        "follow_up_questions": [
                            f"Are you experiencing any other symptoms of {disease}?",
                            "How long have you felt this way?",
                        ],
                        "common_causes": [
                            disease
                        ],  # Disease itself forms the cause bucket
                    }

                # Fit TF-IDF on all symptom texts
                self.tfidf_matrix = self.vectorizer.fit_transform(self.disease_texts)
                self.is_trained = True
                print(
                    f"✅ Kaggle Symptom2Disease Dataset Loaded: {len(self.disease_labels)} diseases"
                )
            else:
                print(f"⚠️ Dataset not found. Falling back to default.")

        except Exception as e:
            print(f"❌ Error loading Kaggle dataset: {e}")

    def extract_symptoms(self, text: str) -> List[ExtractedSymptom]:
        """Extract and analyze symptoms by comparing against the Kaggle dataset text"""
        self._ensure_loaded()
        if not self.is_trained or not text.strip():
            return []

        found_symptoms = []

        # Vectorize user input
        user_vec = self.vectorizer.transform([text.lower()])

        # Calculate cosine similarity with all disease symptom profiles
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(user_vec, self.tfidf_matrix)[0]

        # Get top matches (threshold > 0.1)
        top_indices = np.argsort(similarities)[::-1]

        for idx in top_indices:
            confidence = similarities[idx]
            if confidence > 0.1:
                disease = self.disease_labels[idx]
                data = self.symptom_database[disease]

                # Extract the distinct symptom keywords matching from user text
                user_words = set(text.lower().replace(",", "").replace(".", "").split())
                disease_words = set(self.vectorizer.get_feature_names_out())
                matched_keywords = list(user_words.intersection(disease_words))

                if not matched_keywords:
                    # Fallback if no direct overlap but vectorizer matched
                    matched_keywords = ["general symptoms"]

                found_symptoms.append(
                    ExtractedSymptom(
                        symptom=f"Symptoms matching {disease}",
                        confidence=float(confidence),
                        matched_text=matched_keywords[:3],
                        related_context=[],
                        urgency=data["urgency"],
                        possible_causes=[disease],
                    )
                )

        # Return top 3 matches
        return sorted(found_symptoms, key=lambda x: x.confidence, reverse=True)[:3]


import sqlite3


class ConversationMemory:
    """Advanced conversation memory with medical context tracking backed by SQLite"""

    def __init__(self):
        self.sessions = {}  # session_id -> conversation_data
        self.max_history_length = 10  # Keep last 10 interactions
        self.db_path = os.path.join(
            os.path.dirname(__file__), "data", "medical_chats.db"
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._load_all()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS sessions 
                            (session_id TEXT PRIMARY KEY, data TEXT)""")

    def _load_all(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT session_id, data FROM sessions")
            for row in cursor:
                try:
                    data = json.loads(row[1])
                    # Reconstruct Sets
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
                except Exception as e:
                    pass

    def _save_session(self, session_id):
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

        # Create interaction record
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

        # Maintain history limit
        if len(session["conversation_history"]) > self.max_history_length:
            session["conversation_history"] = session["conversation_history"][
                -self.max_history_length :
            ]

        # Update accumulated medical information
        self._update_accumulated_info(session, extracted_info)

        # Update conversation state
        self._update_conversation_state(session, extracted_info)

        self._save_session(session_id)

    def get_context(self, session_id: str) -> Dict:
        """Get comprehensive conversation context"""

        if session_id not in self.sessions:
            return self._create_new_session_context()

        session = self.sessions[session_id]

        return {
            "conversation_history": session["conversation_history"][
                -5:
            ],  # Last 5 interactions
            "accumulated_symptoms": list(session["accumulated_symptoms"]),
            "accumulated_conditions": list(session["accumulated_conditions"]),
            "accumulated_medications": list(session["accumulated_medications"]),
            "patient_profile": session["patient_profile"],
            "conversation_state": session["conversation_state"].value,
            "conversation_summary": self._generate_conversation_summary(session),
            "urgency_level": session["urgency_level"],
            "last_topic": session["last_topic"],
            "session_start_time": session["session_start_time"],
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

        # Add symptoms
        if "symptoms" in extracted_info:
            for symptom_obj in extracted_info["symptoms"]:
                if isinstance(symptom_obj, dict):
                    session["accumulated_symptoms"].add(symptom_obj.get("symptom", ""))
                else:
                    session["accumulated_symptoms"].add(str(symptom_obj))

        # Add entities
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

        # Check for emergency indicators
        urgency_indicators = entities.get("urgency_indicators", [])
        if urgency_indicators or any(
            s.get("urgency") == "critical" for s in symptoms if isinstance(s, dict)
        ):
            session["conversation_state"] = ConversationState.EMERGENCY
            session["urgency_level"] = "critical"
            return

        # Normal state progression
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

        if session["urgency_level"] != "low":
            summary_parts.append(f"Urgency level: {session['urgency_level']}")

        return ". ".join(summary_parts) + "."


class ContextBuilder:
    """Build enriched context for LLM prompts"""

    def build_context(
        self,
        current_input: str,
        entities: Dict,
        symptoms: List[ExtractedSymptom],
        conversation_context: Dict,
    ) -> Dict:
        """Build comprehensive context for prompt enrichment"""

        return {
            "current_input": current_input,
            "current_entities": entities,
            "current_symptoms": [asdict(s) for s in symptoms],
            "conversation_context": conversation_context,
            "medical_urgency": self._assess_medical_urgency(symptoms, entities),
            "conversation_flow": self._analyze_conversation_flow(conversation_context),
            "follow_up_suggestions": self._generate_follow_up_suggestions(
                symptoms, conversation_context
            ),
        }

    def _assess_medical_urgency(
        self, symptoms: List[ExtractedSymptom], entities: Dict
    ) -> str:
        """Assess overall medical urgency"""

        # Check for emergency indicators
        if entities.get("urgency_indicators"):
            return "critical"

        # Check symptom urgency levels
        urgency_levels = [s.urgency for s in symptoms]
        if "critical" in urgency_levels:
            return "critical"
        elif "high" in urgency_levels:
            return "high"
        elif "moderate" in urgency_levels:
            return "moderate"
        else:
            return "low"

    def _analyze_conversation_flow(self, context: Dict) -> Dict:
        """Analyze conversation flow and progression"""

        return {
            "state": context.get("conversation_state", "initial"),
            "progression": self._determine_conversation_progression(context),
            "gaps": self._identify_information_gaps(context),
            "next_logical_steps": self._suggest_next_steps(context),
        }

    def _determine_conversation_progression(self, context: Dict) -> str:
        """Determine how the conversation is progressing"""

        interactions = context.get("total_interactions", 0)
        symptoms = len(context.get("accumulated_symptoms", []))

        if interactions == 0:
            return "conversation_start"
        elif interactions < 3 and symptoms == 0:
            return "information_gathering"
        elif symptoms > 0 and interactions < 5:
            return "symptom_exploration"
        else:
            return "analysis_and_guidance"

    def _identify_information_gaps(self, context: Dict) -> List[str]:
        """Identify missing information that should be gathered"""

        gaps = []
        symptoms = context.get("accumulated_symptoms", [])

        if symptoms:
            if not any("duration" in str(s).lower() for s in symptoms):
                gaps.append("symptom_duration")
            if not any("severity" in str(s).lower() for s in symptoms):
                gaps.append("symptom_severity")
            if not any("location" in str(s).lower() for s in symptoms):
                gaps.append("symptom_location")

        return gaps

    def _suggest_next_steps(self, context: Dict) -> List[str]:
        """Suggest logical next steps in conversation"""

        state = context.get("conversation_state", "initial")
        urgency = context.get("urgency_level", "low")

        if urgency == "critical":
            return ["emergency_guidance", "immediate_action_required"]
        elif state == "initial":
            return ["gather_chief_complaint", "build_rapport"]
        elif state == "symptom_gathering":
            return ["clarify_symptoms", "assess_severity", "gather_timeline"]
        elif state == "symptom_analysis":
            return ["provide_analysis", "suggest_next_steps", "offer_reassurance"]
        else:
            return ["continue_support", "monitor_progress"]

    def _generate_follow_up_suggestions(
        self, symptoms: List[ExtractedSymptom], context: Dict
    ) -> List[str]:
        """Generate relevant follow-up questions based on symptoms"""

        suggestions = []

        for symptom in symptoms[:2]:  # Focus on top 2 symptoms
            # Access follow_up_questions from symptom database if available
            # This is a simplified version - in practice, you'd have this data
            if "chest pain" in symptom.symptom.lower():
                suggestions.extend(
                    [
                        "When did the chest pain start?",
                        "Is the pain radiating to your arm, jaw, or back?",
                        "Are you experiencing shortness of breath?",
                    ]
                )
            elif "headache" in symptom.symptom.lower():
                suggestions.extend(
                    [
                        "Is this headache different from your usual headaches?",
                        "Do you have sensitivity to light or sound?",
                        "Any nausea or vomiting with the headache?",
                    ]
                )

        return list(set(suggestions))  # Remove duplicates


class MedicalRAGEnrichmentEngine:
    """Main RAG engine that coordinates all components"""

    def __init__(self):
        self.conversation_memory = ConversationMemory()
        self.medical_ner = MedicalEntityRecognizer()
        self.symptom_extractor = SymptomExtractor()
        self.context_builder = ContextBuilder()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._cache: Dict[str, Dict] = {}
        self._cache_max_size = 100

    def _get_cache_key(self, user_input: str, session_id: str) -> str:
        """Generate cache key from input"""
        return f"{session_id}:{user_input[:50]}"

    def _get_cached_result(self, user_input: str, session_id: str) -> Optional[Dict]:
        """Get cached result if available"""
        key = self._get_cache_key(user_input, session_id)
        return self._cache.get(key)

    def _cache_result(self, user_input: str, session_id: str, result: Dict):
        """Cache the result"""
        if len(self._cache) >= self._cache_max_size:
            # Simple FIFO: remove first item
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        key = self._get_cache_key(user_input, session_id)
        self._cache[key] = result

    def process_user_input(self, user_input: str, session_id: str) -> Dict:
        """Main RAG processing pipeline"""

        # Check cache first
        cached = self._get_cached_result(user_input, session_id)
        if cached:
            cached["from_cache"] = True
            return cached

        # 1. Extract medical entities and symptoms in parallel
        entities_future = self._executor.submit(
            self.medical_ner.extract_entities, user_input
        )
        symptoms_future = self._executor.submit(
            self.symptom_extractor.extract_symptoms, user_input
        )

        entities = entities_future.result()
        symptoms = symptoms_future.result()

        # 2. Get conversation context
        conversation_context = self.conversation_memory.get_context(session_id)

        # 3. Build enriched context
        enriched_context = self.context_builder.build_context(
            current_input=user_input,
            entities=entities,
            symptoms=symptoms,
            conversation_context=conversation_context,
        )

        # 4. Create enriched prompt for LLM
        enriched_prompt = self._create_enriched_prompt(enriched_context)

        result = {
            "enriched_prompt": enriched_prompt,
            "context": enriched_context,
            "entities": entities,
            "symptoms": [asdict(s) for s in symptoms],
            "conversation_context": conversation_context,
            "confidence_score": self._calculate_confidence_score(
                entities, symptoms, conversation_context
            ),
        }

        # Cache the result
        self._cache_result(user_input, session_id, result)

        return result

    def _create_enriched_prompt(self, context: Dict) -> str:
        """Create comprehensive enriched prompt for LLM"""

        current_input = context["current_input"]
        conversation_context = context["conversation_context"]
        current_symptoms = context["current_symptoms"]
        current_entities = context["current_entities"]
        medical_urgency = context["medical_urgency"]

        # Build system prompt based on conversation state and urgency
        system_prompt = self._build_system_prompt(conversation_context, medical_urgency)

        # Build conversation history context
        conversation_history = self._build_conversation_history_context(
            conversation_context
        )

        # Build medical context
        medical_context = self._build_medical_context(
            current_symptoms, current_entities, conversation_context
        )

        # Build current query context
        current_query_context = self._build_current_query_context(
            current_input, current_symptoms, current_entities
        )

        # Build guidance for response
        response_guidance = self._build_response_guidance(context)

        full_prompt = f"""
{system_prompt}

{conversation_history}

{medical_context}

{current_query_context}

{response_guidance}

Please provide a thoughtful, contextual response that demonstrates understanding of our conversation history and the patient's current concerns.
"""

        return full_prompt

    def _build_system_prompt(self, conversation_context: Dict, urgency: str) -> str:
        """Build adaptive system prompt"""

        state = conversation_context.get("conversation_state", "initial")
        interactions = conversation_context.get("total_interactions", 0)

        base_prompt = "You are an empathetic AI medical assistant engaged in an ongoing conversation with a patient."

        if urgency == "critical":
            return f"{base_prompt} URGENT SITUATION DETECTED - Provide immediate guidance while recommending emergency care."

        state_prompts = {
            "initial": "The patient is starting to share their health concerns. Focus on building rapport and gathering initial information.",
            "symptom_gathering": "You're gathering detailed symptom information. Ask specific follow-up questions to understand the complete picture.",
            "symptom_analysis": "You have symptom information and are providing analysis. Connect current symptoms with previously discussed information.",
            "treatment_discussion": "You're discussing next steps and treatment options. Reference the full context of symptoms discussed.",
            "follow_up": "This is a follow-up conversation. Check on previously discussed symptoms and progress.",
        }

        state_specific = state_prompts.get(
            state, "Continue the supportive medical conversation."
        )

        return f"{base_prompt} {state_specific} This is interaction #{interactions + 1} in this conversation."

    def _build_conversation_history_context(self, context: Dict) -> str:
        """Build conversation history for context"""

        history = context.get("conversation_history", [])
        if not history:
            return "CONVERSATION CONTEXT: This is the beginning of a new conversation with this patient."

        parts = [
            "CONVERSATION CONTEXT:",
            f"- Total interactions: {len(history)}",
            f"- Conversation state: {context.get('conversation_state', 'unknown')}",
            f"- Session duration: Started {context.get('session_start_time', 'unknown')}",
        ]

        # Add accumulated information
        symptoms = context.get("accumulated_symptoms", [])
        if symptoms:
            parts.append(f"- Symptoms discussed: {', '.join(symptoms)}")

        conditions = context.get("accumulated_conditions", [])
        if conditions:
            parts.append(f"- Conditions mentioned: {', '.join(conditions)}")

        # Add recent interaction
        if history:
            recent = history[-1]
            parts.append(f'- Last user input: "{recent["user_input"]}"')
            parts.append(f'- Your last response: "{recent["ai_response"][:150]}..."')

        return "\n".join(parts)

    def _build_medical_context(
        self, symptoms: List[Dict], entities: Dict, conversation_context: Dict
    ) -> str:
        """Build medical context section"""

        parts = ["MEDICAL CONTEXT:"]

        # Current symptoms
        if symptoms:
            parts.append("Current symptoms detected:")
            for symptom in symptoms:
                confidence = symptom.get("confidence", 0)
                urgency = symptom.get("urgency", "unknown")
                parts.append(
                    f"  - {symptom['symptom']} (confidence: {confidence:.2f}, urgency: {urgency})"
                )

        # Current medical entities
        entity_types = ["body_parts", "conditions", "medications", "temporal"]
        for entity_type in entity_types:
            items = entities.get(entity_type, [])
            if items:
                parts.append(
                    f"- {entity_type.replace('_', ' ').title()}: {', '.join(items)}"
                )

        # Severity and duration
        if entities.get("severity") != "unspecified":
            parts.append(f"- Severity: {entities['severity']}")
        if entities.get("duration") != "unspecified":
            parts.append(f"- Duration: {entities['duration']}")

        # Urgency assessment
        urgency = conversation_context.get("urgency_level", "low")
        if urgency != "low":
            parts.append(f"- Overall urgency level: {urgency}")

        return "\n".join(parts)

    def _build_current_query_context(
        self, user_input: str, symptoms: List[Dict], entities: Dict
    ) -> str:
        """Build current query context"""

        return f"""CURRENT USER INPUT: "{user_input}"

EXTRACTED FROM CURRENT INPUT:
- Number of symptoms detected: {len(symptoms)}
- Medical entities found: {sum(len(v) if isinstance(v, list) else 0 for v in entities.values())}
- Urgency indicators: {len(entities.get("urgency_indicators", []))}
"""

    def _build_response_guidance(self, context: Dict) -> str:
        """Build guidance for LLM response"""

        urgency = context["medical_urgency"]
        gaps = context["conversation_flow"]["gaps"]
        next_steps = context["conversation_flow"]["next_logical_steps"]
        follow_ups = context["follow_up_suggestions"]

        guidance_parts = ["RESPONSE GUIDANCE:"]

        if urgency == "critical":
            guidance_parts.append(
                "- PRIORITY: Address urgent medical situation immediately"
            )
            guidance_parts.append(
                "- Recommend emergency care while providing immediate guidance"
            )
        else:
            guidance_parts.append("- Maintain empathetic, supportive tone")
            guidance_parts.append(
                "- Reference relevant information from conversation history"
            )

        if gaps:
            guidance_parts.append(f"- Information gaps to address: {', '.join(gaps)}")

        if next_steps:
            guidance_parts.append(f"- Logical next steps: {', '.join(next_steps)}")

        if follow_ups:
            guidance_parts.append(
                f"- Consider asking: {follow_ups[0] if follow_ups else 'N/A'}"
            )

        return "\n".join(guidance_parts)

    def _calculate_confidence_score(
        self, entities: Dict, symptoms: List, context: Dict
    ) -> float:
        """Calculate overall confidence score for the interaction"""

        # Base confidence on entity extraction quality
        entity_score = min(
            1.0,
            sum(len(v) if isinstance(v, list) else 0 for v in entities.values()) * 0.1,
        )

        # Add symptom confidence
        symptom_score = 0
        if symptoms:
            symptom_confidences = [
                s.get("confidence", 0) for s in symptoms if isinstance(s, dict)
            ]
            if symptom_confidences:
                symptom_score = sum(symptom_confidences) / len(symptom_confidences)

        # Context quality score
        context_score = min(1.0, context.get("total_interactions", 0) * 0.1)

        # Weighted average
        final_score = entity_score * 0.3 + symptom_score * 0.5 + context_score * 0.2

        return round(min(1.0, final_score), 2)


if __name__ == "__main__":
    # Test the RAG engine
    engine = MedicalRAGEnrichmentEngine()

    # Test conversation
    session_id = "test_session_123"

    # First interaction
    result1 = engine.process_user_input("I have chest pain", session_id)
    print("=== First Interaction ===")
    print(f"Symptoms detected: {len(result1['symptoms'])}")
    print(f"Confidence: {result1['confidence_score']}")
    print(f"Prompt length: {len(result1['enriched_prompt'])} characters")

    # Add to memory (simulating AI response)
    engine.conversation_memory.add_interaction(
        session_id=session_id,
        user_input="I have chest pain",
        extracted_info={
            "symptoms": result1["symptoms"],
            "entities": result1["entities"],
        },
        ai_response="I understand you're experiencing chest pain. This is concerning. Can you tell me when this started and if you have any other symptoms?",
        confidence_score=result1["confidence_score"],
    )

    # Second interaction
    result2 = engine.process_user_input(
        "It started an hour ago and I feel nauseous", session_id
    )
    print("\n=== Second Interaction ===")
    print(f"Symptoms detected: {len(result2['symptoms'])}")
    print(
        f"Conversation state: {result2['conversation_context']['conversation_state']}"
    )
    print(
        f"Total accumulated symptoms: {len(result2['conversation_context']['accumulated_symptoms'])}"
    )
    print(f"Prompt preview: {result2['enriched_prompt'][:200]}...")
