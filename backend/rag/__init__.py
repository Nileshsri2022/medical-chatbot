"""
Medical RAG Module
Modular RAG components for medical chatbot
"""

from .engine import MedicalRAGEnrichmentEngine
from .entities import MedicalEntityRecognizer
from .symptoms import SymptomExtractor
from .memory import ConversationMemory
from .context import ContextBuilder


def get_vector_store():
    """Lazy import for vector store"""
    from .vector_store import get_vector_store as _gs

    return _gs()


def get_hybrid_retriever(alpha: float = 0.5):
    """Lazy import for hybrid retriever"""
    from .vector_store import get_hybrid_retriever as _gr

    return _gr(alpha)


def load_medical_documents():
    """Lazy import for document loader"""
    from .document_loader import load_medical_documents as _ld

    return _ld()


__all__ = [
    "MedicalRAGEnrichmentEngine",
    "MedicalEntityRecognizer",
    "SymptomExtractor",
    "ConversationMemory",
    "ContextBuilder",
    "get_vector_store",
    "get_hybrid_retriever",
    "load_medical_documents",
]
