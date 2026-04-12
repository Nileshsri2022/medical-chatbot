"""
Symptom Extraction Module
Uses ML (TF-IDF) to match user symptoms against disease database
"""

import os
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

    def __init__(self):
        self.symptom_database: Dict = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.disease_texts: List[str] = []
        self.disease_labels: List[str] = []
        self.is_trained: bool = False
        self._loaded: bool = False

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
            if os.path.exists(emr_path):
                self._load_emr_dataset(emr_path)
            elif os.path.exists(kaggle_path):
                self._load_kaggle_dataset(kaggle_path)
            else:
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

            self.symptom_database[disease] = {
                "text": combined_text,
                "urgency": "high" if disease in ["Heart Disease"] else "moderate",
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
                if disease in ["Heart attack", "Covid", "Hypertension", "Asthma"]
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
                "common_causes": [disease],
            }

        self.tfidf_matrix = self.vectorizer.fit_transform(self.disease_texts)
        self.is_trained = True
        print(f"✅ Kaggle Dataset Loaded: {len(self.disease_labels)} diseases")

    def _load_fallback(self):
        """Load fallback symptom database"""
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
        if not self.is_trained or not text.strip():
            return []

        found_symptoms = []

        user_vec = self.vectorizer.transform([text.lower()])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[::-1]

        for idx in top_indices:
            confidence = similarities[idx]
            if confidence > 0.1:
                disease = self.disease_labels[idx]
                data = self.symptom_database[disease]

                user_words = set(text.lower().replace(",", "").replace(".", "").split())
                disease_words = set(self.vectorizer.get_feature_names_out())
                matched_keywords = list(user_words.intersection(disease_words))

                if not matched_keywords:
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

        return sorted(found_symptoms, key=lambda x: x.confidence, reverse=True)[:3]
