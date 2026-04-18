"""
Main RAG Engine
Coordinates all modular RAG components
"""

from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from .entities import MedicalEntityRecognizer
from .symptoms import ExtractedSymptom, SymptomExtractor
from .memory import ConversationMemory
from .context import ContextBuilder


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

    @property
    def symptom_extractor_symptom_count(self) -> int:
        return self.symptom_extractor.symptom_count

    def _get_cache_key(self, user_input: str, session_id: str) -> str:
        return f"{session_id}:{user_input[:50]}"

    def _get_cached_result(self, user_input: str, session_id: str) -> Optional[Dict]:
        key = self._get_cache_key(user_input, session_id)
        return self._cache.get(key)

    def _cache_result(self, user_input: str, session_id: str, result: Dict):
        if len(self._cache) >= self._cache_max_size:
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        key = self._get_cache_key(user_input, session_id)
        self._cache[key] = result

    def process_user_input(self, user_input: str, session_id: str) -> Dict:
        """Main RAG processing pipeline"""
        cached = self._get_cached_result(user_input, session_id)
        if cached:
            cached["from_cache"] = True
            return cached

        entities_future = self._executor.submit(
            self.medical_ner.extract_entities, user_input
        )
        symptoms_future = self._executor.submit(
            self.symptom_extractor.extract_symptoms, user_input
        )

        entities = entities_future.result()
        symptoms = symptoms_future.result()

        conversation_context = self.conversation_memory.get_context(session_id)

        enriched_context = self.context_builder.build_context(
            current_input=user_input,
            entities=entities,
            symptoms=symptoms,
            conversation_context=conversation_context,
        )

        enriched_prompt = self._create_enriched_prompt(enriched_context)

        result = {
            "enriched_prompt": enriched_prompt,
            "context": enriched_context,
            "entities": entities,
            "symptoms": [asdict(s) if hasattr(s, "__dict__") else s for s in symptoms],
            "conversation_context": conversation_context,
            "confidence_score": self._calculate_confidence_score(
                entities, symptoms, conversation_context
            ),
        }

        self._cache_result(user_input, session_id, result)
        return result

    def _create_enriched_prompt(self, context: Dict) -> str:
        """Create comprehensive enriched prompt for LLM"""
        current_input = context["current_input"]
        conversation_context = context["conversation_context"]
        current_symptoms = context["current_symptoms"]
        current_entities = context["current_entities"]
        medical_urgency = context["medical_urgency"]

        system_prompt = self._build_system_prompt(conversation_context, medical_urgency)
        conversation_history = self._build_conversation_history_context(
            conversation_context
        )
        medical_context = self._build_medical_context(
            current_symptoms, current_entities, conversation_context
        )
        current_query_context = self._build_current_query_context(
            current_input, current_symptoms, current_entities
        )
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
        history = context.get("conversation_history", [])
        if not history:
            return "CONVERSATION CONTEXT: This is the beginning of a new conversation with this patient."

        parts = [
            "CONVERSATION CONTEXT:",
            f"- Total interactions: {len(history)}",
            f"- Conversation state: {context.get('conversation_state', 'unknown')}",
            f"- Session duration: Started {context.get('session_start_time', 'unknown')}",
        ]

        symptoms = context.get("accumulated_symptoms", [])
        if symptoms:
            parts.append(f"- Symptoms discussed: {', '.join(symptoms)}")

        conditions = context.get("accumulated_conditions", [])
        if conditions:
            parts.append(f"- Conditions mentioned: {', '.join(conditions)}")

        if history:
            recent = history[-1]
            parts.append(f'- Last user input: "{recent["user_input"]}"')
            parts.append(f'- Your last response: "{recent["ai_response"][:150]}..."')

        return "\n".join(parts)

    def _build_medical_context(
        self, symptoms: List[Dict], entities: Dict, conversation_context: Dict
    ) -> str:
        parts = ["MEDICAL CONTEXT:"]

        user_reported_symptoms = entities.get("symptoms", [])
        if user_reported_symptoms:
            parts.append(
                "User-reported symptoms from current input: "
                + ", ".join(user_reported_symptoms)
            )

        if symptoms:
            parts.append("Extractor output grounded in current input:")
            for symptom in symptoms:
                symptom_str = (
                    symptom.get("symptom", "Unknown")
                    if isinstance(symptom, dict)
                    else str(symptom)
                )
                confidence = (
                    symptom.get("confidence", 0) if isinstance(symptom, dict) else 0
                )
                urgency = (
                    symptom.get("urgency", "unknown")
                    if isinstance(symptom, dict)
                    else "unknown"
                )
                parts.append(
                    f"  - {symptom_str} (confidence: {confidence:.2f}, urgency: {urgency})"
                )
                possible_causes = (
                    symptom.get("possible_causes", [])
                    if isinstance(symptom, dict)
                    else []
                )
                if possible_causes:
                    parts.append(
                        "    Dataset-informed possible causes: "
                        + ", ".join(possible_causes[:3])
                    )

        entity_types = ["body_parts", "conditions", "medications", "temporal"]
        for entity_type in entity_types:
            items = entities.get(entity_type, [])
            if items:
                parts.append(
                    f"- {entity_type.replace('_', ' ').title()}: {', '.join(items)}"
                )

        if entities.get("severity") != "unspecified":
            parts.append(f"- Severity: {entities['severity']}")
        if entities.get("duration") != "unspecified":
            parts.append(f"- Duration: {entities['duration']}")

        urgency = conversation_context.get("urgency_level", "low")
        if urgency != "low":
            parts.append(f"- Overall urgency level: {urgency}")

        return "\n".join(parts)

    def _build_current_query_context(
        self, user_input: str, symptoms: List[Dict], entities: Dict
    ) -> str:
        return f"""CURRENT USER INPUT: "{user_input}"

EXTRACTED FROM CURRENT INPUT:
- Number of symptoms detected: {len(symptoms)}
- Medical entities found: {sum(len(v) if isinstance(v, list) else 0 for v in entities.values())}
- Urgency indicators: {len(entities.get("urgency_indicators", []))}

INSTRUCTION:
- Ground symptom identification primarily on CURRENT USER INPUT.
- Use any dataset-informed causes only as secondary hints, not as confirmed symptoms.
"""

    def _build_response_guidance(self, context: Dict) -> str:
        urgency = context["medical_urgency"]
        gaps = context["conversation_flow"].get("gaps", [])
        next_steps = context["conversation_flow"].get("next_logical_steps", [])
        follow_ups = context.get("follow_up_suggestions", [])

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
        entity_score = min(
            1.0,
            sum(len(v) if isinstance(v, list) else 0 for v in entities.values()) * 0.1,
        )

        symptom_score = 0
        if symptoms:
            for s in symptoms:
                if hasattr(s, "confidence"):
                    symptom_score += s.confidence
                elif isinstance(s, dict):
                    symptom_score += s.get("confidence", 0)
            symptom_score = min(1.0, symptom_score / 3)

        context_score = 0.3 if context.get("total_interactions", 0) > 0 else 0.1

        return round(
            (entity_score * 0.3 + symptom_score * 0.5 + context_score * 0.2), 3
        )

    def add_interaction(
        self,
        session_id: str,
        user_input: str,
        extracted_info: Dict,
        ai_response: str,
        confidence_score: float = 0.8,
    ):
        """Add interaction to conversation memory"""
        self.conversation_memory.add_interaction(
            session_id=session_id,
            user_input=user_input,
            extracted_info=extracted_info,
            ai_response=ai_response,
            confidence_score=confidence_score,
        )

    def clear_session(self, session_id: str):
        """Clear a session"""
        self.conversation_memory.clear_session(session_id)

    def get_context(self, session_id: str) -> Dict:
        """Get conversation context for a session"""
        return self.conversation_memory.get_context(session_id)

    def __del__(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
