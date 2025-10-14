"""
RAG (Retrieval Augmented Generation) Module

Provides semantic search over chip design documentation to enhance
LLM responses with factual, up-to-date information.

Components:
- VectorStore: ChromaDB-based document storage and retrieval
- Embedder: Sentence embedding for semantic search
- DocumentLoader: Load and chunk EDA documentation
- Retriever: High-level interface for retrieval
"""

from .embedder import Embedder
from .vector_store import VectorStore
from .document_loader import DocumentLoader
from .retriever import RAGRetriever

__all__ = ['Embedder', 'VectorStore', 'DocumentLoader', 'RAGRetriever']
