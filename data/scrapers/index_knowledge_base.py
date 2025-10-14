#!/usr/bin/env python3
"""
Index Knowledge Base into RAG System

Indexes all scraped EDA documentation into the RAG vector store.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from pathlib import Path
from core.rag import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def index_knowledge_base(knowledge_base_dir: str = "./data/knowledge_base"):
    """
    Index all documentation from knowledge base directory.

    Args:
        knowledge_base_dir: Directory containing scraped documentation
    """
    kb_path = Path(knowledge_base_dir)

    if not kb_path.exists():
        logger.error(f"Knowledge base directory not found: {knowledge_base_dir}")
        logger.info("Run data/scrapers/eda_doc_scraper.py first to download documentation")
        return

    logger.info("="*60)
    logger.info("Indexing Knowledge Base into RAG System")
    logger.info("="*60)

    # Initialize RAG retriever
    rag = RAGRetriever()

    # Get current stats
    stats_before = rag.get_stats()
    logger.info(f"\nCurrent index: {stats_before['document_count']} documents")

    # Index all documentation
    logger.info(f"\nIndexing directory: {knowledge_base_dir}")
    total_indexed = rag.index_directory(
        str(kb_path),
        extensions=['.md', '.txt', '.rst', '.html'],
        recursive=True
    )

    # Get new stats
    stats_after = rag.get_stats()

    logger.info("\n" + "="*60)
    logger.info("Indexing Complete!")
    logger.info("="*60)
    logger.info(f"Total chunks indexed: {total_indexed}")
    logger.info(f"Total documents in index: {stats_after['document_count']}")
    logger.info(f"Vector store location: {stats_after['persist_directory']}")

    # Test retrieval
    logger.info("\n" + "="*60)
    logger.info("Testing Retrieval")
    logger.info("="*60)

    test_queries = [
        "How do I synthesize a design with Yosys?",
        "What is placement in chip design?",
        "How does DREAMPlace work?",
    ]

    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = rag.retrieve(query, top_k=2)

        if results:
            for i, result in enumerate(results, 1):
                source = result['metadata'].get('source', 'Unknown')
                distance = result.get('distance', 0.0)
                preview = result['document'][:150] + "..."

                logger.info(f"  Result {i} (distance={distance:.3f}):")
                logger.info(f"    Source: {source}")
                logger.info(f"    Preview: {preview}")
        else:
            logger.warning("  No results found")

    logger.info("\n✅ Knowledge base indexed successfully!")
    logger.info("\nThe agent can now use this documentation to answer queries.")


if __name__ == "__main__":
    index_knowledge_base()
