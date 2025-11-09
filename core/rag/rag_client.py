"""
RAG-Enhanced API Client

Architectural wrapper that automatically injects RAG context into all API calls.
This ensures RAG is NEVER lost from the pipeline - it's built into the client layer.

Usage:
    from core.rag.rag_client import RAGEnhancedClient

    client = RAGEnhancedClient(
        anthropic_api_key="...",
        rag_persist_dir="./data/rag/chroma_db"
    )

    # RAG context automatically injected
    response = await client.generate_with_rag(
        user_prompt="Design a Moore FSM...",
        system_prompt="You are an expert...",
        query_for_rag="Moore FSM design patterns"
    )
"""

import logging
import os
from typing import Optional, Dict, List
from anthropic import AsyncAnthropic, Anthropic
from .retriever import RAGRetriever

logger = logging.getLogger(__name__)


class RAGEnhancedClient:
    """
    RAG-Enhanced Anthropic API Client

    Wraps AsyncAnthropic client and automatically injects RAG context
    into prompts. This architectural pattern ensures RAG is permanently
    integrated and cannot be "lost" from the pipeline.
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        rag_persist_dir: str = "./data/rag/chroma_db",
        rag_collection: str = "chip_design_docs",
        enable_rag: bool = True
    ):
        """
        Initialize RAG-enhanced client.

        Args:
            anthropic_api_key: API key (defaults to env var)
            rag_persist_dir: Vector store directory
            rag_collection: Collection name
            enable_rag: Enable RAG injection (default True)
        """
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not provided or set in environment")

        self.client_async = AsyncAnthropic(api_key=api_key)
        self.client_sync = Anthropic(api_key=api_key)

        self.enable_rag = enable_rag

        if self.enable_rag:
            try:
                self.rag = RAGRetriever(
                    persist_directory=rag_persist_dir,
                    collection_name=rag_collection
                )
                logger.info("RAG retriever initialized successfully")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}. Continuing without RAG.")
                self.enable_rag = False
                self.rag = None
        else:
            self.rag = None
            logger.info("RAG disabled by configuration")

    def _build_rag_context(
        self,
        query: str,
        top_k: int = 3,
        max_context_length: int = 2000
    ) -> str:
        """
        Build RAG context from vector store.

        Args:
            query: Search query for RAG
            top_k: Number of documents to retrieve
            max_context_length: Max context length

        Returns:
            Formatted RAG context string
        """
        if not self.enable_rag or self.rag is None:
            return ""

        try:
            context = self.rag.retrieve_and_format(
                query=query,
                top_k=top_k,
                max_context_length=max_context_length
            )

            if context:
                logger.debug(f"Retrieved RAG context ({len(context)} chars)")
            else:
                logger.debug("No RAG context retrieved")

            return context
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return ""

    def _inject_rag_into_prompt(
        self,
        user_prompt: str,
        rag_context: str
    ) -> str:
        """
        Inject RAG context into user prompt.

        Args:
            user_prompt: Original user prompt
            rag_context: RAG context to inject

        Returns:
            Enhanced prompt with RAG context
        """
        if not rag_context:
            return user_prompt

        enhanced_prompt = f"""{rag_context}

---

{user_prompt}"""

        return enhanced_prompt

    async def generate_with_rag(
        self,
        user_prompt: str,
        query_for_rag: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8000,
        temperature: float = 1.0,
        rag_top_k: int = 3,
        rag_max_context: int = 2000,
        **kwargs
    ) -> Dict:
        """
        Generate response with automatic RAG context injection (async).

        Args:
            user_prompt: User prompt
            query_for_rag: Query for RAG retrieval (defaults to user_prompt)
            system_prompt: Optional system prompt
            model: Claude model to use
            max_tokens: Max response tokens
            temperature: Sampling temperature
            rag_top_k: RAG top-k documents
            rag_max_context: RAG max context length
            **kwargs: Additional API parameters

        Returns:
            API response dict
        """
        # Build RAG context
        rag_query = query_for_rag or user_prompt
        rag_context = self._build_rag_context(
            query=rag_query,
            top_k=rag_top_k,
            max_context_length=rag_max_context
        )

        # Inject RAG into prompt
        enhanced_prompt = self._inject_rag_into_prompt(user_prompt, rag_context)

        # Build messages
        messages = [{"role": "user", "content": enhanced_prompt}]

        # Build API call parameters
        api_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            **kwargs
        }

        if system_prompt:
            api_params["system"] = system_prompt

        # Call API
        logger.info(f"Calling Claude API with RAG (enabled={self.enable_rag})")
        response = await self.client_async.messages.create(**api_params)

        return response

    def generate_with_rag_sync(
        self,
        user_prompt: str,
        query_for_rag: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8000,
        temperature: float = 1.0,
        rag_top_k: int = 3,
        rag_max_context: int = 2000,
        **kwargs
    ) -> Dict:
        """
        Generate response with automatic RAG context injection (sync).

        Same as generate_with_rag() but synchronous.
        """
        # Build RAG context
        rag_query = query_for_rag or user_prompt
        rag_context = self._build_rag_context(
            query=rag_query,
            top_k=rag_top_k,
            max_context_length=rag_max_context
        )

        # Inject RAG into prompt
        enhanced_prompt = self._inject_rag_into_prompt(user_prompt, rag_context)

        # Build messages
        messages = [{"role": "user", "content": enhanced_prompt}]

        # Build API call parameters
        api_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            **kwargs
        }

        if system_prompt:
            api_params["system"] = system_prompt

        # Call API
        logger.info(f"Calling Claude API with RAG (enabled={self.enable_rag})")
        response = self.client_sync.messages.create(**api_params)

        return response

    def index_training_data(
        self,
        dataset_path: str,
        sample_limit: int = 100
    ) -> int:
        """
        Index training data into RAG vector store.

        Args:
            dataset_path: Path to JSONL training data
            sample_limit: Max examples to index

        Returns:
            Number of chunks indexed
        """
        if not self.enable_rag or self.rag is None:
            logger.warning("RAG not enabled, cannot index data")
            return 0

        import json

        logger.info(f"Indexing training data from {dataset_path}")

        documents = []
        metadatas = []

        with open(dataset_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx >= sample_limit:
                    break

                example = json.loads(line)

                # Create document from example
                instruction = example.get('instruction', '')
                output = example.get('output', '')
                hierarchy = example.get('hierarchy', {})

                # Format as searchable document
                doc_text = f"""INSTRUCTION: {instruction}

CODE:
{output}

HIERARCHY: L1={hierarchy.get('l1', 'Unknown')}, L2={hierarchy.get('l2', 'Unknown')}, L3={hierarchy.get('l3', 'Unknown')}"""

                documents.append(doc_text)
                metadatas.append({
                    'source': dataset_path,
                    'type': 'training_example',
                    'l1': hierarchy.get('l1'),
                    'l2': hierarchy.get('l2'),
                    'l3': hierarchy.get('l3'),
                    'index': idx
                })

        if not documents:
            logger.warning("No documents to index")
            return 0

        # Add to vector store
        ids = [f"train_{i}" for i in range(len(documents))]
        self.rag.vector_store.add_documents(documents, metadatas, ids)

        logger.info(f"Indexed {len(documents)} training examples")
        return len(documents)

    def get_stats(self) -> Dict:
        """Get RAG statistics"""
        if not self.enable_rag or self.rag is None:
            return {"enabled": False}

        return {
            "enabled": True,
            **self.rag.get_stats()
        }
