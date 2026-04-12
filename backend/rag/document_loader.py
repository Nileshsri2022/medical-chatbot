"""
Medical Document Loader
Loads and processes PDF documents for RAG context
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk from a document"""

    text: str
    page_number: int
    source: str
    metadata: Dict[str, Any]


class MedicalDocumentLoader:
    """Load and process medical PDFs"""

    def __init__(self, documents_dir: str = None):
        if documents_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            documents_dir = os.path.join(base_dir, "data", "documents")

        self.documents_dir = documents_dir
        os.makedirs(documents_dir, exist_ok=True)
        self._chunks = []

    def load_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Load a PDF file and extract text chunks"""
        try:
            from pypdf import PdfReader

            chunks = []
            reader = PdfReader(file_path)

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_chunks = self._split_into_chunks(text, page_num, file_path)
                    chunks.extend(text_chunks)

            return chunks
        except ImportError:
            logger.warning("pypdf not installed, cannot load PDFs")
            return []
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []

    def _split_into_chunks(
        self,
        text: str,
        page_number: int,
        source: str,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> List[DocumentChunk]:
        """Split text into chunks with overlap"""
        chunks = []
        lines = text.split("\n")
        current_chunk = []
        current_size = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_size = len(line)

            if current_size + line_size > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        page_number=page_number,
                        source=source,
                        metadata={"chunk_size": len(current_chunk)},
                    )
                )

                overlap_lines = current_chunk[-max(1, overlap // 20) :]
                current_chunk = overlap_lines
                current_size = sum(len(l) for l in current_chunk)

            current_chunk.append(line)
            current_size += line_size + 1

        if current_chunk:
            chunks.append(
                DocumentChunk(
                    text=" ".join(current_chunk),
                    page_number=page_number,
                    source=source,
                    metadata={"chunk_size": len(current_chunk)},
                )
            )

        return chunks

    def load_all_pdfs(self) -> List[DocumentChunk]:
        """Load all PDFs from documents directory"""
        all_chunks = []

        if not os.path.exists(self.documents_dir):
            return all_chunks

        for filename in os.listdir(self.documents_dir):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.documents_dir, filename)
                chunks = self.load_pdf(file_path)
                all_chunks.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {filename}")

        self._chunks = all_chunks
        return all_chunks

    def load_csv_as_documents(self, csv_path: str) -> List[DocumentChunk]:
        """Load CSV file as documents"""
        try:
            import pandas as pd

            chunks = []
            df = pd.read_csv(csv_path)

            for idx, row in df.iterrows():
                text = " ".join(
                    [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                )
                chunks.append(
                    DocumentChunk(
                        text=text,
                        page_number=1,
                        source=csv_path,
                        metadata={"row": idx, "source_type": "csv"},
                    )
                )

            return chunks
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return []

    def load_medical_datasets(self) -> List[DocumentChunk]:
        """Load existing medical datasets as documents"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")

        chunks = []

        csv_files = ["kaggle_symptom2disease.csv", "sample_emr_dataset.csv"]

        for filename in csv_files:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                file_chunks = self.load_csv_as_documents(file_path)
                chunks.extend(file_chunks)
                logger.info(f"Loaded {len(file_chunks)} chunks from {filename}")

        self._chunks.extend(chunks)
        return chunks

    def get_chunks(self) -> List[DocumentChunk]:
        """Get all loaded chunks"""
        return self._chunks

    def get_chunks_as_text(self) -> List[str]:
        """Get all chunks as text list"""
        return [chunk.text for chunk in self._chunks]

    def get_chunks_with_metadata(self) -> List[Dict[str, Any]]:
        """Get chunks with metadata for vector store"""
        return [
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page_number,
            }
            for chunk in self._chunks
        ]


def load_medical_documents() -> List[DocumentChunk]:
    """Convenience function to load all medical documents"""
    loader = MedicalDocumentLoader()

    loader.load_medical_datasets()

    pdf_chunks = loader.load_all_pdfs()

    return loader.get_chunks()
