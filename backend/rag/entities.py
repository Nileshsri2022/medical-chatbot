"""
Medical Entity Recognition Module
Extracts medical entities (symptoms, body parts, conditions, medications) from user input

Optimized with:
- Pre-compiled regex patterns for faster matching
- Local variable caching
- LRU cache for common inputs
"""

import re
from typing import Dict, List, Optional
from functools import lru_cache


class MedicalEntityRecognizer:
    """Advanced medical entity recognition using pattern matching - OPTIMIZED"""

    def __init__(self):
        self.medical_patterns = {
            "symptoms": [
                r"\b(pain|ache|hurt|sore|tender|burning|throbbing|sharp|dull|stabbing|cramping)\b",
                r"\b(fever|temperature|hot|chills|sweating|feverish|burning up)\b",
                r"\b(nausea|vomiting|sick|queasy|throwing up|stomach ache|belly pain)\b",
                r"\b(headache|migraine|head pain|dizzy|dizziness|lightheaded|vertigo)\b",
                r"\b(shortness of breath|breathless|gasping|wheezing|cough|coughing)\b",
                r"\b(chest pain|chest tightness|chest pressure|heart pain|palpitations)\b",
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

        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._severity_patterns: Dict[str, re.Pattern] = {}
        self._duration_patterns: Dict[str, re.Pattern] = {}
        self._urgency_patterns: List[re.Pattern] = []

        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile all regex patterns for better performance"""
        for category, patterns in self.medical_patterns.items():
            self._compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        severity_patterns = {
            "severe": r"\b(severe|excruciating|unbearable|intense|terrible|awful|extreme|worst)\b",
            "moderate": r"\b(moderate|noticeable|uncomfortable|bothersome|manageable|medium)\b",
            "mild": r"\b(mild|slight|little|minor|small|barely|light)\b",
        }
        self._severity_patterns = {
            k: re.compile(v) for k, v in severity_patterns.items()
        }

        duration_patterns = {
            "acute": r"\b(sudden|minutes|hour|hours|today|just now|right now)\b",
            "subacute": r"\b(days|few days|week|yesterday)\b",
            "chronic": r"\b(weeks|months|years|long time|always|chronic)\b",
        }
        self._duration_patterns = {
            k: re.compile(v) for k, v in duration_patterns.items()
        }

        urgency_patterns = [
            r"\b(emergency|urgent|immediate|help|911|hospital|emergency room)\b",
            r"\b(can't breathe|chest pain|heart attack|stroke|bleeding)\b",
            r"\b(severe pain|unbearable|excruciating|passing out)\b",
        ]
        self._urgency_patterns = [re.compile(p) for p in urgency_patterns]

    @lru_cache(maxsize=512)
    def extract_entities(self, text: str) -> Dict:
        """Extract comprehensive medical entities from text - OPTIMIZED with LRU cache"""
        if not text:
            return self._empty_entities()

        text_lower = text.lower()

        severity = self._extract_severity_fast(text_lower)
        duration = self._extract_duration_fast(text_lower)
        urgency = self._extract_urgency_fast(text_lower)

        entities = {
            "symptoms": self._extract_category(
                text_lower, self._compiled_patterns["symptoms"]
            ),
            "body_parts": self._extract_category(
                text_lower, self._compiled_patterns["body_parts"]
            ),
            "conditions": self._extract_category(
                text_lower, self._compiled_patterns["conditions"]
            ),
            "medications": self._extract_category(
                text_lower, self._compiled_patterns["medications"]
            ),
            "temporal": self._extract_category(
                text_lower, self._compiled_patterns["temporal"]
            ),
            "severity": severity,
            "duration": duration,
            "urgency_indicators": urgency,
        }

        return entities

    def _extract_category(self, text: str, patterns: List[re.Pattern]) -> List[str]:
        """Extract matches for a category using pre-compiled patterns"""
        matches = []
        for pattern in patterns:
            found = pattern.findall(text)
            matches.extend(found)
        return list(dict.fromkeys(matches)) if matches else []

    def _extract_severity_fast(self, text: str) -> str:
        """Extract severity using pre-compiled patterns"""
        for severity, pattern in self._severity_patterns.items():
            if pattern.search(text):
                return severity
        return "unspecified"

    def _extract_duration_fast(self, text: str) -> str:
        """Extract duration using pre-compiled patterns"""
        for duration, pattern in self._duration_patterns.items():
            if pattern.search(text):
                return duration
        return "unspecified"

    def _extract_urgency_fast(self, text: str) -> List[str]:
        """Extract urgency indicators using pre-compiled patterns"""
        indicators = []
        for pattern in self._urgency_patterns:
            matches = pattern.findall(text)
            indicators.extend(matches)
        return list(set(indicators)) if indicators else []

    def _empty_entities(self) -> Dict:
        return {
            "symptoms": [],
            "body_parts": [],
            "conditions": [],
            "medications": [],
            "temporal": [],
            "severity": "unspecified",
            "duration": "unspecified",
            "urgency_indicators": [],
        }

    def clear_cache(self):
        """Clear the LRU cache"""
        self.extract_entities.cache_clear()
