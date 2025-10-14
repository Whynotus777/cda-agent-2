"""
Vector Store Module

Persistent storage and semantic search over document embeddings.
Uses ChromaDB for efficient similarity search.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database for storing and retrieving document embeddings.

    Uses ChromaDB for persistent storage and fast similarity search.
    """

    def __init__(
        self,
        persist_directory: str = "./data/rag/chroma_db",
        collection_name: str = "chip_design_docs"
    ):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.collection = None
        self.client = None

        # Create directory if needed
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings

            logger.info(f"Initializing ChromaDB at {self.persist_directory}")

            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Chip design documentation and knowledge base"}
            )

            logger.info(f"Collection '{self.collection_name}' ready, {self.collection.count()} documents")

        except ImportError:
            logger.error(
                "chromadb not installed. Install with: pip install chromadb"
            )
            raise

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents (auto-generated if not provided)
        """
        if not documents:
            return

        # Generate IDs if not provided
        if ids is None:
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        # Add to collection (ChromaDB handles embedding internally)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Added {len(documents)} documents to vector store")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of dicts with 'document', 'metadata', 'distance', 'id'
        """
        if self.collection.count() == 0:
            logger.warning("Vector store is empty")
            return []

        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0,
                    'id': results['ids'][0][i]
                })

        logger.debug(f"Search returned {len(formatted_results)} results")
        return formatted_results

    def delete_collection(self):
        """Delete the entire collection (use with caution!)"""
        if self.client and self.collection:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'collection_name': self.collection_name,
            'document_count': self.collection.count() if self.collection else 0,
            'persist_directory': str(self.persist_directory)
        }
