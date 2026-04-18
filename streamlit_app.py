"""
Medical RAG Chatbot - Streamlit UI
A conversational AI assistant for medical symptom analysis.
Converts the FastAPI backend to Streamlit for easy UI.
"""

import os
import sys
import json
import time
import logging
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any

import streamlit as st
import pandas as pd
import httpx
from openai import OpenAI

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "moonshotai/kimi-k2-instruct")
LLM_SERVER_PORT = int(os.getenv("LLM_SERVER_PORT", "8001"))
CONTENT_SAFETY_MODEL = os.getenv(
    "CONTENT_SAFETY_MODEL", "nvidia/nemotron-3-content-safety"
)

# ─── Constants ────────────────────────────────────────────────────────────────────────

MEDICAL_DISCLAIMER = """
⚠️ **Important Disclaimer**: This AI assistant is for informational purposes only and is **NOT a substitute for professional medical advice, diagnosis, or treatment**. Always consult with a qualified healthcare provider for medical concerns.
"""

URGENT_WARNING = """
🚨 **Seek Immediate Medical Attention** if you're experiencing:
- Chest pain, pressure, or tightness
- Difficulty breathing
- Severe bleeding
- Sudden confusion or loss of consciousness
- Signs of stroke (face drooping, arm weakness, speech difficulty)
- Severe injury
"""

# ─── Streamlit Setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "session_" + str(int(time.time()))

if "entities" not in st.session_state:
    st.session_state.entities = {}

if "symptoms" not in st.session_state:
    st.session_state.symptoms = []

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {}

if "llm_client" not in st.session_state:
    st.session_state.llm_client = None

# ─── Backend Components (Adapted) ────────────────────────────────────────────


class StreamlitRAGEngine:
    """Streamlit-compatible RAG engine using backend components"""

    def __init__(self):
        from backend.rag.engine import MedicalRAGEnrichmentEngine

        self.engine = MedicalRAGEnrichmentEngine()

    def process_input(self, user_input: str, session_id: str) -> Dict:
        """Process user input and return enriched context"""
        result = self.engine.process_user_input(user_input, session_id)
        return result

    def add_interaction(
        self,
        session_id: str,
        user_input: str,
        extracted_info: Dict,
        ai_response: str,
        confidence_score: float = 0.8,
    ):
        """Add interaction to conversation memory"""
        self.engine.add_interaction(
            session_id=session_id,
            user_input=user_input,
            extracted_info=extracted_info,
            ai_response=ai_response,
            confidence_score=confidence_score,
        )

    def get_context(self, session_id: str) -> Dict:
        """Get conversation context"""
        return self.engine.get_context(session_id)

    def clear_session(self, session_id: str):
        """Clear a session"""
        self.engine.clear_session(session_id)


# ─── LLM Client ──────────────────────────────────────────────────────────────────────────────


def get_llm_client() -> Optional[OpenAI]:
    """Get or create OpenAI client"""
    if not LLM_API_KEY:
        st.warning("LLM_API_KEY not set. Set it in .env file.")
        return None

    try:
        client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        return None


def check_content_safety(
    client: OpenAI, prompt: str, image_data: Optional[str] = None
) -> Dict:
    """Check content safety using NVIDIA Nemotron 3 Content Safety model"""
    try:
        messages = []

        if image_data:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=CONTENT_SAFETY_MODEL,
            messages=messages,
            max_tokens=100,
            temperature=0.3,
        )

        content = response.choices[0].message.content.lower()

        return {
            "is_safe": "unsafe" not in content or "safe" in content,
            "raw_response": content,
        }
    except Exception as e:
        logger.warning(f"Content safety check failed: {e}")
        return {"is_safe": True, "raw_response": "", "error": str(e)}


MEDICAL_SYSTEM_PROMPT = """You are a medical symptom analysis AI. Your job is to:
1. Identify symptoms from the patient's description
2. Suggest possible conditions/illnesses based on those symptoms

IMPORTANT: You MUST respond ONLY with valid JSON in exactly this format, no markdown, no extra text:
{
  "symptoms": ["symptom1", "symptom2"],
  "illnesses": [
    {
      "name": "Condition Name",
      "illness_coverage": 75,
      "condition_coverage": 60
    }
  ]
}

Rules:
- "symptoms" is a list of identified symptom strings (use standard medical terminology)
- "illnesses" is a list of possible conditions, each with:
  - "name": condition name
  - "illness_coverage": percentage (0-100) of that illness's typical symptoms that match
  - "condition_coverage": percentage (0-100) of the patient's symptoms explained by this condition
- List up to 5 most likely conditions, sorted by relevance
- Be thorough in symptom identification
- Include both common and serious possibilities
- Always include a disclaimer-worthy common condition and any serious red flags

Respond ONLY with the JSON object. No other text."""


def call_llm(
    client: OpenAI, description: str, max_tokens: int = 200, temperature: float = 0.7
) -> Dict:
    """Call LLM for diagnosis"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze these symptoms: {description}"},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def format_diagnosis_response(llm_response: str, enriched_context: Dict) -> str:
    """Format LLM response into user-friendly message"""
    try:
        parsed = json.loads(llm_response)
    except json.JSONDecodeError:
        try:
            import re

            json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                return llm_response
        except:
            return llm_response

    symptoms = parsed.get("symptoms", [])
    illnesses = parsed.get("illnesses", [])

    lines = [
        "Thanks for sharing your symptoms. Based on your details, here is a preliminary analysis:",
        "",
    ]

    if symptoms:
        lines.append("**Possible symptoms identified:** " + ", ".join(symptoms[:8]))

    if illnesses:
        lines.append("")
        lines.append("**Possible conditions to discuss with a clinician:**")
        for idx, illness in enumerate(illnesses[:3], start=1):
            name = illness.get("name", "Unspecified condition")
            illness_cov = illness.get("illness_coverage", 0)
            condition_cov = illness.get("condition_coverage", 0)
            lines.append(
                f"{idx}. {name} (match quality: illness {illness_cov}%, symptom fit {condition_cov}%)"
            )
    else:
        lines.append("No high-confidence condition match was returned by the model.")

    lines.append("")
    lines.append(
        "⚠️ **If symptoms are severe, sudden, worsening, or include chest pain, breathing trouble, severe headache, confusion, or fainting, seek emergency care immediately.**"
    )

    return "\n".join(lines)


# ─── Initialize RAG Engine ───────────────────────────────────────────────────


@st.cache_resource
def get_rag_engine():
    """Get RAG engine instance"""
    return StreamlitRAGEngine()


rag_engine = get_rag_engine()


# ─── UI Functions ───────────────────────────────────────────────────────────


def check_urgency(entities: Dict, symptoms: List) -> bool:
    """Check if there's urgent medical situation"""
    if entities.get("urgency_indicators"):
        return True

    for s in symptoms:
        if isinstance(s, dict) and s.get("urgency") == "critical":
            return True
        if hasattr(s, "urgency") and s.urgency == "critical":
            return True

    return False


def render_extracted_info(entities: Dict, symptoms: List):
    """Render extracted medical information"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Extracted Entities")
        if entities:
            for key, value in entities.items():
                if value and value != "unspecified" and value != []:
                    if isinstance(value, list):
                        st.markdown(
                            f"**{key.replace('_', ' ').title()}:** {', '.join(value)}"
                        )
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

    with col2:
        st.markdown("### 🩺 Detected Symptoms")
        if symptoms:
            for s in symptoms:
                if isinstance(s, dict):
                    symptom_name = s.get("symptom", "Unknown")
                    confidence = s.get("confidence", 0)
                    urgency = s.get("urgency", "low")
                    st.markdown(
                        f"- **{symptom_name}** (confidence: {confidence:.2f}, urgency: {urgency})"
                    )
                elif hasattr(s, "symptom"):
                    st.markdown(
                        f"- **{s.symptom}** (confidence: {s.confidence:.2f}, urgency: {s.urgency})"
                    )


def render_conversation_context(context: Dict):
    """Render conversation context summary"""
    st.markdown("### 📊 Conversation Context")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Interactions", context.get("total_interactions", 0))
    with col2:
        st.metric("Symptoms Discussed", len(context.get("accumulated_symptoms", [])))
    with col3:
        st.metric(
            "Conditions Mentioned", len(context.get("accumulated_conditions", []))
        )

    if context.get("conversation_state"):
        st.markdown(f"**Conversation State:** {context.get('conversation_state')}")

    if context.get("urgency_level") and context.get("urgency_level") != "low":
        st.warning(f"⚠️ Urgency Level: {context.get('urgency_level')}")


# ─── Main App ──────────────────────────────────────────────────────────────


def main():
    # Sidebar
    st.sidebar.title("🏥 Medical RAG Chatbot")
    st.sidebar.markdown("---")

    # New Chat Button
    if st.sidebar.button("🔄 New Chat", use_container_width=True):
        rag_engine.clear_session(st.session_state.session_id)
        st.session_state.session_id = "session_" + str(int(time.time()))
        st.session_state.messages = []
        st.session_state.entities = {}
        st.session_state.symptoms = []
        st.session_state.conversation_context = {}
        st.rerun()

    # Quick Actions - Common Symptoms
    st.sidebar.markdown("### 💡 Quick Symptoms")
    st.sidebar.markdown("*Tap to add to chat*")

    common_symptoms = [
        "Chest pain",
        "Headache",
        "Fever",
        "Cough",
        "Fatigue",
        "Stomach pain",
        "Shortness of breath",
        "Dizziness",
    ]

    cols = st.sidebar.columns(2)
    for i, symptom in enumerate(common_symptoms):
        if cols[i % 2].button(symptom, use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": f"I have {symptom.lower()}"}
            )
            st.rerun()

    st.sidebar.markdown("---")

    # Session Info
    st.sidebar.markdown("### 📋 Session Info")
    st.sidebar.markdown(f"**Session:** `{st.session_state.session_id[:16]}...`")
    st.sidebar.markdown(f"**Messages:** {len(st.session_state.messages)}")

    if st.session_state.conversation_context:
        interactions = st.session_state.conversation_context.get(
            "total_interactions", 0
        )
        st.sidebar.markdown(f"**Interactions:** {interactions}")

    # Show configuration
    with st.sidebar.expander("⚙️ Configuration"):
        st.markdown(
            f"**LLM Provider:** {LLM_BASE_URL.split('/')[2] if '//' in LLM_BASE_URL else 'Unknown'}"
        )
        st.markdown(f"**Model:** `{LLM_MODEL}`")
        st.markdown(f"**Safety Model:** `{CONTENT_SAFETY_MODEL.split('/')[-1]}`")

    st.sidebar.markdown("---")

    # Medical Tips
    with st.sidebar.expander("💊 Medical Tips"):
        st.markdown("""
        • **Stay hydrated** - Drink 8 glasses of water daily
        • **Get enough sleep** - 7-9 hours recommended
        • **Exercise regularly** - 30 mins most days
        • **Eat balanced meals** - Include fruits & vegetables
        """)

    # Emergency Info
    with st.sidebar.expander("🚨 Emergency"):
        st.markdown("""
        **Call emergency services if:**
        - Chest pain or pressure
        - Severe bleeding
        - Difficulty breathing
        - Signs of stroke
        - Loss of consciousness
        """)

    # Main content
    st.title("🏥 Medical RAG Chatbot")
    st.markdown("*AI-powered symptom analysis assistant*")
    st.markdown("---")

    # Display disclaimer
    st.info(MEDICAL_DISCLAIMER)

    # Chat messages with custom styling
    st.markdown("### 💬 Conversation")

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(
            message["role"], avatar="👨‍⚕️" if message["role"] == "assistant" else "👤"
        ):
            content = message["content"]
            if message["role"] == "assistant":
                st.markdown(content)
            else:
                if content.startswith("[Image:"):
                    st.markdown(f"📎 {content}")
                else:
                    st.markdown(content)

    # Chat input
    with st.container():
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            prompt = st.chat_input("Describe your symptoms...")
        with col2:
            uploaded_file = st.file_uploader(
                "📷", type=["png", "jpg", "jpeg", "gif"], label_visibility="collapsed"
            )
        with col3:
            if st.button("🗑️", help="Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Handle image upload first
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        with st.chat_message("user", avatar="👤"):
            st.image(uploaded_file, caption=uploaded_file.name)

        client = get_llm_client()
        if client:
            with st.spinner("Checking content safety..."):
                safety_result = check_content_safety(
                    client, "Medical image for analysis", image_base64
                )

            if not safety_result.get("is_safe", True):
                st.error(
                    "⚠️ Content safety check failed. Please upload a different image."
                )
            else:
                st.session_state.messages.append(
                    {"role": "user", "content": f"[Image: {uploaded_file.name}]"}
                )

                with st.spinner("Analyzing image..."):
                    try:
                        # Create a multimodal prompt for the LLM
                        multimodal_response = client.chat.completions.create(
                            model=LLM_MODEL,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a medical image analysis assistant. Describe what you see in the medical image related to symptoms or conditions. Focus on visible symptoms, anomalies, or medical findings.",
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Analyze this medical image and describe any visible symptoms or conditions.",
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{image_base64}"
                                            },
                                        },
                                    ],
                                },
                            ],
                            max_tokens=300,
                            temperature=0.7,
                        )
                        image_description = multimodal_response.choices[
                            0
                        ].message.content

                        # Now use this description in the RAG pipeline
                        result = rag_engine.process_input(
                            image_description, st.session_state.session_id
                        )

                        # Store extracted info
                        st.session_state.entities = result.get("entities", {})
                        st.session_state.symptoms = result.get("symptoms", [])
                        st.session_state.conversation_context = result.get(
                            "conversation_context", {}
                        )

                        is_urgent = check_urgency(
                            st.session_state.entities, st.session_state.symptoms
                        )
                        if is_urgent:
                            st.error(URGENT_WARNING)

                        # Get diagnosis from main LLM
                        llm_response = call_llm(
                            client, result.get("enriched_prompt", image_description)
                        )
                        formatted_response = format_diagnosis_response(
                            llm_response, result
                        )
                        formatted_response = (
                            f"{formatted_response}\n\n{MEDICAL_DISCLAIMER}"
                        )

                        rag_engine.add_interaction(
                            session_id=st.session_state.session_id,
                            user_input=f"[Image: {uploaded_file.name}] {image_description}",
                            extracted_info={
                                "entities": st.session_state.entities,
                                "symptoms": st.session_state.symptoms,
                            },
                            ai_response=formatted_response,
                            confidence_score=result.get("confidence_score", 0.0),
                        )

                        st.session_state.messages.append(
                            {"role": "assistant", "content": formatted_response}
                        )
                        with st.chat_message("assistant"):
                            st.markdown(formatted_response)

                    except Exception as e:
                        logger.error(f"Image analysis failed: {e}")
                        st.error(
                            f"⚠️ This model does not support image input. Please use a vision-capable model or upload text descriptions instead.\n\nError: {str(e)}"
                        )
        else:
            st.error("LLM service not configured. Please set LLM_API_KEY in .env")

        uploaded_file = None

    elif prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Process with RAG engine
        with st.spinner("Analyzing your symptoms..."):
            try:
                # Get RAG-enriched context
                result = rag_engine.process_input(prompt, st.session_state.session_id)

                # Store extracted info
                st.session_state.entities = result.get("entities", {})
                st.session_state.symptoms = result.get("symptoms", [])
                st.session_state.conversation_context = result.get(
                    "conversation_context", {}
                )

                # Check for urgent situation
                is_urgent = check_urgency(
                    st.session_state.entities, st.session_state.symptoms
                )

                if is_urgent:
                    st.error(URGENT_WARNING)

                # Call LLM for diagnosis
                client = get_llm_client()
                if client:
                    try:
                        llm_response = call_llm(
                            client, result.get("enriched_prompt", prompt)
                        )
                        formatted_response = format_diagnosis_response(
                            llm_response, result
                        )
                    except Exception as e:
                        logger.error(f"LLM call failed: {e}")
                        formatted_response = f"I processed your input but encountered an error calling the diagnosis service. Error: {str(e)}\n\nPlease consult a healthcare professional for proper medical advice."
                else:
                    formatted_response = "LLM service is not configured. Please set LLM_API_KEY in the .env file."

                # Add AI response
                formatted_response = f"{formatted_response}\n\n{MEDICAL_DISCLAIMER}"

                rag_engine.add_interaction(
                    session_id=st.session_state.session_id,
                    user_input=prompt,
                    extracted_info={
                        "entities": st.session_state.entities,
                        "symptoms": st.session_state.symptoms,
                    },
                    ai_response=formatted_response,
                    confidence_score=result.get("confidence_score", 0.0),
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": formatted_response}
                )

                with st.chat_message("assistant", avatar="👨‍⚕️"):
                    st.markdown(formatted_response)

                # Show extracted info in expandable section
                with st.expander("🔍 View Extracted Medical Information"):
                    if st.session_state.entities or st.session_state.symptoms:
                        render_extracted_info(
                            st.session_state.entities, st.session_state.symptoms
                        )
                    else:
                        st.info("No medical entities or symptoms detected.")

                # Show conversation context
                if (
                    st.session_state.conversation_context.get("total_interactions", 0)
                    > 0
                ):
                    with st.expander("📊 View Conversation Context"):
                        render_conversation_context(
                            st.session_state.conversation_context
                        )

            except Exception as e:
                logger.error(f"Error processing input: {e}")
                error_msg = f"I apologize, but I encountered an error processing your message: {str(e)}. Please try again or consult a healthcare professional."
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                with st.chat_message("assistant"):
                    st.markdown(error_msg)


if __name__ == "__main__":
    main()
