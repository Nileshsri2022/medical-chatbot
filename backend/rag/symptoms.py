"""
Symptom Extraction Module
Uses ML (TF-IDF) to match user symptoms against disease database
"""

import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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


class SymptomExtractor:
    """Advanced symptom extraction powered by Kaggle Symptom2Disease dataset"""

    SIMILARITY_THRESHOLD = 0.12
    MAX_SYMPTOMS = 5

    HIGH_URGENCY_CONDITIONS = {
        "Heart Disease",
        "Heart attack",
        "Covid",
        "Hypertension",
        "Asthma",
    }
    LOW_URGENCY_CONDITIONS = {"Common Cold", "Acne", "Allergy"}

    def __init__(self):
        self.symptom_database: Dict = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.disease_texts: List[str] = []
        self.disease_labels: List[str] = []
        self.is_trained: bool = False
        self._loaded: bool = False
        self._direct_symptom_phrases = [
            "chest pain",
            "shortness of breath",
            "sore throat",
            "runny nose",
            "body aches",
            "stomach pain",
            "abdominal pain",
            "headache",
            "migraine",
            "dizziness",
            "nausea",
            "vomiting",
            "diarrhea",
            "constipation",
            "fatigue",
            "weakness",
            "fever",
            "chills",
            "cough",
            "wheezing",
            "palpitations",
            "rash",
            "swelling",
            "pain",
            "ache",
            "cold",
        ]

    def _ensure_loaded(self):
        """Lazy load the dataset on first use"""
        if self._loaded:
            return
        self._loaded = True
        self._load_dataset()

    @property
    def symptom_count(self) -> int:
        """Get count of loaded symptoms (triggers lazy load)"""
        self._ensure_loaded()
        return len(self.symptom_database)

    def _load_dataset(self):
        """Load and train on the Kaggle Symptom2Disease dataset"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
        )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        kaggle_path = os.path.join(base_dir, "data", "kaggle_symptom2disease.csv")
        emr_path = os.path.join(base_dir, "data", "sample_emr_dataset.csv")

        try:
            loaded = False
            if os.path.exists(emr_path):
                try:
                    self._load_emr_dataset(emr_path)
                    loaded = True
                except Exception as e:
                    print(f"⚠️ EMR load failed: {e}")

            if not loaded and os.path.exists(kaggle_path):
                try:
                    self._load_kaggle_dataset(kaggle_path)
                    loaded = True
                except Exception as e:
                    print(f"⚠️ Kaggle load failed: {e}")

            if not loaded:
                print("⚠️ No dataset found, using fallback symptom database")
                self._load_fallback()
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self._load_fallback()

    def _load_emr_dataset(self, path: str):
        """Load EMR tabular dataset"""
        df = pd.read_csv(path)

        if "apacheadmissiondx" in df.columns:
            label_col = "apacheadmissiondx"
        elif "Diagnosis_Label" in df.columns:
            label_col = "Diagnosis_Label"
        else:
            label_col = df.columns[0]

        df = df.dropna(subset=[label_col])

        for disease, group in df.groupby(label_col):
            serialized_texts = []
            for _, row in group.iterrows():
                features = []
                for col in df.columns:
                    if col != label_col and pd.notna(row[col]):
                        features.append(f"{col} is {row[col]}")
                paragraph = f"Patient presents with: {', '.join(features)}."
                serialized_texts.append(paragraph)

            combined_text = " ".join(serialized_texts)
            self.disease_labels.append(disease)
            self.disease_texts.append(combined_text)

            disease_keywords = set(re.findall(r"[a-z]{3,}", str(disease).lower()))

            self.symptom_database[disease] = {
                "text": combined_text,
                "urgency": "high"
                if disease in self.HIGH_URGENCY_CONDITIONS
                else "moderate",
                "keywords": disease_keywords,
                "follow_up_questions": [
                    f"Do you have a history of {disease}?",
                    "Have these symptoms worsened recently?",
                ],
                "common_causes": [disease],
            }

        self.tfidf_matrix = self.vectorizer.fit_transform(self.disease_texts)
        self.is_trained = True
        print(f"✅ EMR Dataset Loaded: {len(self.disease_labels)} diseases")

    def _load_kaggle_dataset(self, path: str):
        """Load Kaggle Symptom2Disease dataset"""
        df = pd.read_csv(path)

        for disease, group in df.groupby("label"):
            combined_text = " ".join(group["text"].tolist())
            self.disease_labels.append(disease)
            self.disease_texts.append(combined_text)

            urgency = (
                "high"
                if disease in self.HIGH_URGENCY_CONDITIONS
                else "low"
                if disease in self.LOW_URGENCY_CONDITIONS
                else "moderate"
            )

            self.symptom_database[disease] = {
                "text": combined_text,
                "urgency": urgency,
                "follow_up_questions": [
                    f"Are you experiencing any other symptoms of {disease}?",
                    "How long have you felt this way?",
                ],
                "common_causes": [disease],
            }

        self.tfidf_matrix = self.vectorizer.fit_transform(self.disease_texts)
        self.is_trained = True
        print(f"✅ Kaggle Dataset Loaded: {len(self.disease_labels)} diseases")

    def _load_fallback(self):
        """Load fallback symptom database"""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words="english",
                ngram_range=(1, 2),
            )

        fallback_conditions = [
            ("Common Cold", ["runny nose", "sore throat", "cough", "sneezing"], "low"),
            ("Flu", ["fever", "body aches", "fatigue", "cough"], "moderate"),
            ("Headache", ["head pain", "tension", "pressure"], "low"),
            (
                "Migraine",
                ["severe headache", "nausea", "light sensitivity"],
                "moderate",
            ),
            ("Chest Pain", ["chest tightness", "pressure", "radiating pain"], "high"),
        ]

        for disease, symptoms, urgency in fallback_conditions:
            text = " ".join(symptoms)
            self.disease_labels.append(disease)
            self.disease_texts.append(text)
            self.symptom_database[disease] = {
                "text": text,
                "urgency": urgency,
                "common_causes": [disease],
            }

        if self.disease_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.disease_texts)
            self.is_trained = True

    def extract_symptoms(self, text: str) -> List[ExtractedSymptom]:
        """Extract and analyze symptoms by comparing against the dataset"""
        self._ensure_loaded()
        if (
            not self.is_trained
            or not text.strip()
            or self.vectorizer is None
            or self.tfidf_matrix is None
        ):
            return []

        text_lower = text.lower().strip()
        found_symptoms: List[ExtractedSymptom] = []

        user_vec = self.vectorizer.transform([text_lower])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:5]
        top_causes = [
            self.disease_labels[idx]
            for idx in top_indices
            if similarities[idx] >= self.SIMILARITY_THRESHOLD
        ]
        top_confidence = (
            float(similarities[top_indices[0]]) if len(top_indices) else 0.0
        )

        direct_matches = []
        for phrase in self._direct_symptom_phrases:
            if phrase in text_lower:
                direct_matches.append(phrase)

        if direct_matches:
            seen = set()
            for phrase in direct_matches:
                if phrase in seen:
                    continue
                seen.add(phrase)
                found_symptoms.append(
                    ExtractedSymptom(
                        symptom=phrase,
                        confidence=max(0.55, top_confidence),
                        matched_text=[phrase],
                        related_context=[],
                        urgency="moderate",
                        possible_causes=top_causes[:3],
                    )
                )
        else:
            # Fallback to weak lexical extraction so we still ground on user wording.
            user_words = set(re.findall(r"[a-z]{3,}", text_lower))
            noisy_terms = {
                "patient",
                "presents",
                "with",
                "admit",
                "admitted",
                "unit",
                "hospital",
                "room",
                "floor",
                "male",
                "female",
                "years",
                "year",
                "old",
            }
            cleaned_words = [w for w in user_words if w not in noisy_terms]
            for word in cleaned_words[:3]:
                found_symptoms.append(
                    ExtractedSymptom(
                        symptom=word,
                        confidence=max(0.35, top_confidence),
                        matched_text=[word],
                        related_context=[],
                        urgency="low",
                        possible_causes=top_causes[:2],
                    )
                )

        return sorted(found_symptoms, key=lambda x: x.confidence, reverse=True)[
            : self.MAX_SYMPTOMS
        ]
