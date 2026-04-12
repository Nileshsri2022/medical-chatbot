"""
Context Builder Module
Builds enriched context for LLM prompts
"""

from typing import Dict, List, Any
from dataclasses import asdict


class ContextBuilder:
    """Build enriched context for LLM prompts"""

    def build_context(
        self,
        current_input: str,
        entities: Dict,
        symptoms: List[Any],
        conversation_context: Dict,
    ) -> Dict:
        """Build comprehensive context for prompt enrichment"""

        return {
            "current_input": current_input,
            "current_entities": entities,
            "current_symptoms": [
                asdict(s) if hasattr(s, "__dict__") else s for s in symptoms
            ],
            "conversation_context": conversation_context,
            "medical_urgency": self._assess_medical_urgency(symptoms, entities),
            "conversation_flow": self._analyze_conversation_flow(conversation_context),
            "follow_up_suggestions": self._generate_follow_up_suggestions(
                symptoms, conversation_context
            ),
        }

    def _assess_medical_urgency(self, symptoms: List[Any], entities: Dict) -> str:
        """Assess overall medical urgency"""
        if entities.get("urgency_indicators"):
            return "critical"

        urgency_levels = []
        for s in symptoms:
            if hasattr(s, "urgency"):
                urgency_levels.append(s.urgency)
            elif isinstance(s, dict):
                urgency_levels.append(s.get("urgency", "low"))

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
        self, symptoms: List[Any], context: Dict
    ) -> List[str]:
        """Generate relevant follow-up questions based on symptoms"""
        suggestions = []

        for symptom in symptoms[:2]:
            symptom_str = (
                symptom.symptom
                if hasattr(symptom, "symptom")
                else symptom.get("symptom", "")
            )

            if "chest pain" in symptom_str.lower():
                suggestions.extend(
                    [
                        "When did the chest pain start?",
                        "Is the pain radiating to your arm, jaw, or back?",
                        "Are you experiencing shortness of breath?",
                    ]
                )
            elif "headache" in symptom_str.lower():
                suggestions.extend(
                    [
                        "Is this headache different from your usual headaches?",
                        "Do you have sensitivity to light or sound?",
                        "Any nausea or vomiting with the headache?",
                    ]
                )

        return list(set(suggestions)) if suggestions else []
