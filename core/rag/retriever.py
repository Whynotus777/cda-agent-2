"""
RAG Retriever Module

High-level interface for retrieval augmented generation.
Combines vector store, embedder, and document loader.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

from .vector_store import VectorStore
from .embedder import Embedder
from .document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    High-level interface for RAG system.

    Handles document indexing, retrieval, and context formatting.
    """

    def __init__(
        self,
        persist_directory: str = "./data/rag/chroma_db",
        collection_name: str = "chip_design_docs"
    ):
        """
        Initialize RAG retriever.

        Args:
            persist_directory: Directory for vector store
            collection_name: Name of document collection
        """
        self.vector_store = VectorStore(persist_directory, collection_name)
        self.embedder = Embedder()
        self.doc_loader = DocumentLoader()

        logger.info("RAG Retriever initialized")

    def index_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> int:
        """
        Index all documents in a directory.

        Args:
            directory: Path to directory
            extensions: File extensions to include
            recursive: Search subdirectories

        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing directory: {directory}")

        # Load documents
        chunks = self.doc_loader.load_directory(directory, extensions, recursive)

        if not chunks:
            logger.warning(f"No documents found in {directory}")
            return 0

        # Add to vector store
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{chunk.metadata.get('source', 'doc')}_{i}" for i, chunk in enumerate(chunks)]

        self.vector_store.add_documents(documents, metadatas, ids)

        logger.info(f"Indexed {len(chunks)} document chunks")
        return len(chunks)

    def index_file(self, file_path: str) -> int:
        """
        Index a single file.

        Args:
            file_path: Path to file

        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing file: {file_path}")

        chunks = self.doc_loader.load_file(file_path)

        if not chunks:
            return 0

        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).name}_{i}" for i in range(len(chunks))]

        self.vector_store.add_documents(documents, metadatas, ids)

        return len(chunks)

    def index_url(self, url: str) -> int:
        """
        Index content from a URL.

        Args:
            url: Web URL to fetch and index

        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing URL: {url}")

        chunks = self.doc_loader.load_web_content(url)

        if not chunks:
            return 0

        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"web_{hash(url)}_{i}" for i in range(len(chunks))]

        self.vector_store.add_documents(documents, metadatas, ids)

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of retrieved documents with metadata
        """
        return self.vector_store.search(query, top_k, filter_metadata)

    def retrieve_and_format(
        self,
        query: str,
        top_k: int = 3,
        max_context_length: int = 4000
    ) -> str:
        """
        Retrieve documents and format as context for LLM.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            max_context_length: Maximum total context length

        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k)

        if not results:
            return ""

        # Format context
        context_parts = ["# Relevant Documentation\n"]

        total_length = 0
        for i, result in enumerate(results, 1):
            doc_text = result['document']
            metadata = result.get('metadata', {})

            # Format source info
            source = metadata.get('source', 'Unknown')
            source_type = metadata.get('type', 'document')

            header = f"\n## Source {i}: {source} ({source_type})\n"
            content = f"{doc_text}\n"

            # Check length
            chunk_length = len(header) + len(content)
            if total_length + chunk_length > max_context_length:
                # Truncate
                remaining = max_context_length - total_length - len(header)
                if remaining > 100:
                    content = doc_text[:remaining] + "...\n"
                else:
                    break

            context_parts.append(header)
            context_parts.append(content)
            total_length += chunk_length

        formatted_context = "".join(context_parts)

        logger.debug(f"Generated context of {len(formatted_context)} characters from {len(results)} documents")

        return formatted_context

    def get_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        return self.vector_store.get_stats()

    def clear_index(self):
        """Clear all indexed documents (use with caution!)"""
        self.vector_store.delete_collection()
        logger.info("Cleared document index")
