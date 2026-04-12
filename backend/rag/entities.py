"""
Medical Entity Recognition Module
Extracts medical entities (symptoms, body parts, conditions, medications) from user input
"""

import re
from typing import Dict, List


class MedicalEntityRecognizer:
    """Advanced medical entity recognition using pattern matching"""

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
            if category != "severity":
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    entities[category].extend(matches)

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
