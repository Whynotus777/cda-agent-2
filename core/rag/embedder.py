"""
Embedder Module

Converts text into dense vector embeddings for semantic search.
Uses sentence-transformers for high-quality embeddings.
"""

import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generate embeddings for text using sentence-transformers.

    Uses all-MiniLM-L6-v2 by default (fast, good quality, 384 dimensions).
    Falls back to simple averaging if sentence-transformers unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder.

        Args:
            model_name: Name of sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM

        self._load_model()

    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded, dimension: {self.embedding_dim}")

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            logger.warning("Using fallback simple embeddings (not recommended)")
            self.model = None

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.

        Args:
            text: Single string or list of strings

        Returns:
            numpy array of embeddings, shape (n, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]

        if self.model is not None:
            # Use sentence-transformers
            embeddings = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings
        else:
            # Fallback: simple character-based embeddings (not good, but works)
            return self._fallback_embed(text)

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """
        Fallback embedding using character frequencies.

        This is NOT good for semantic search, but allows the system
        to function without sentence-transformers.
        """
        logger.warning("Using fallback embeddings - install sentence-transformers for better results")

        embeddings = []
        for text in texts:
            # Simple character frequency vector
            vec = np.zeros(self.embedding_dim)
            for i, char in enumerate(text.lower()[:self.embedding_dim]):
                vec[i] = ord(char) / 255.0
            embeddings.append(vec)

        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        return self.embed_text(query)[0]
