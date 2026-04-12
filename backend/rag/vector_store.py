"""
Vector Store Module for Medical RAG
Provides semantic search using ChromaDB and sentence embeddings
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

import chromadb
from chromadb.config import Settings


@dataclass
class SearchResult:
    """Vector search result"""

    text: str
    score: float
    metadata: Dict[str, Any]
    source: str


class MedicalVectorStore:
    """Vector store for medical knowledge with ChromaDB"""

    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        persist_directory: str = None,
    ):
        self.collection_name = collection_name

        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(base_dir, "data", "vector_store")

        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(self.collection_name)
            except Exception:
                self._collection = self.client.create_collection(
                    self.collection_name,
                    metadata={"description": "Medical knowledge base"},
                )
        return self._collection

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None,
    ):
        """Add documents to vector store"""
        if not texts:
            return

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata,
            )

            search_results = []
            if results["documents"] and results["documents"][0]:
                for i, text in enumerate(results["documents"][0]):
                    search_results.append(
                        SearchResult(
                            text=text,
                            score=1 - results["distances"][0][i],
                            metadata=results["metadatas"][0][i]
                            if results["metadatas"]
                            else {},
                            source=results["metadatas"][0][i].get("source", "unknown")
                            if results["metadatas"]
                            else "unknown",
                        )
                    )

            return search_results
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def get_count(self) -> int:
        """Get number of documents in store"""
        return self.collection.count()

    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection_name)
        self._collection = None


class HybridRetriever:
    """Combines dense (vector) and sparse (BM25) retrieval"""

    def __init__(
        self,
        vector_store: MedicalVectorStore = None,
        alpha: float = 0.5,
    ):
        self.vector_store = vector_store or MedicalVectorStore()
        self.alpha = alpha
        self._bm25_index = None
        self._documents = []

    def _build_bm25_index(self, documents: List[str]):
        """Build BM25 index"""
        try:
            from rank_bm25 import BM25Okapi

            tokenized = [doc.lower().split() for doc in documents]
            self._bm25_index = BM25Okapi(tokenized)
            self._documents = documents
        except ImportError:
            pass

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None,
    ):
        """Add documents to both vector store and BM25 index"""
        self.vector_store.add_documents(texts, metadatas)

        self._documents.extend(texts)
        if len(self._documents) > 0:
            self._build_bm25_index(self._documents)

    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[SearchResult]:
        """Hybrid search combining vector and BM25"""
        vector_results = self.vector_store.search(query, n_results * 2)

        bm25_results = []
        if self._bm25_index and self._documents:
            try:
                tokenized_query = query.lower().split()
                scores = self._bm25_index.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[::-1][: n_results * 2]

                for idx in top_indices:
                    bm25_results.append(
                        SearchResult(
                            text=self._documents[idx],
                            score=scores[idx],
                            metadata={},
                            source="bm25",
                        )
                    )
            except Exception:
                pass

        combined = self._merge_results(vector_results, bm25_results, n_results)
        return combined

    def _merge_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_n: int,
    ) -> List[SearchResult]:
        """Merge and rerank results from both methods"""
        seen = set()
        merged = []

        for v_result in vector_results:
            if v_result.text not in seen:
                seen.add(v_result.text)
                merged.append(v_result)

        for b_result in bm25_results:
            if b_result.text not in seen:
                seen.add(b_result.text)
                merged.append(b_result)

        normalized = []
        if merged:
            max_score = max(r.score for r in merged)
            for r in merged:
                normalized_score = r.score / max_score if max_score > 0 else 0
                normalized.append(
                    SearchResult(
                        text=r.text,
                        score=normalized_score,
                        metadata=r.metadata,
                        source=r.source,
                    )
                )

        normalized.sort(key=lambda x: x.score, reverse=True)
        return normalized[:top_n]


_medical_vector_store = None
_hybrid_retriever = None


def get_vector_store() -> MedicalVectorStore:
    global _medical_vector_store
    if _medical_vector_store is None:
        _medical_vector_store = MedicalVectorStore()
    return _medical_vector_store


def get_hybrid_retriever(alpha: float = 0.5) -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(alpha=alpha)
    return _hybrid_retriever
