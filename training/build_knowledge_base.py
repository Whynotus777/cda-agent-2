#!/usr/bin/env python3
"""
Build RAG Knowledge Base from Specialist Corpus
Convert PLACEMENT_REFORMED_V2.jsonl into vector database
"""

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def load_corpus(corpus_path: str):
    """Load specialist corpus"""
    print(f"Loading corpus: {corpus_path}")

    documents = []
    with open(corpus_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # Extract fields
            doc = {
                'id': f"doc_{idx}",
                'prompt': obj.get('prompt', ''),
                'response': obj.get('response', ''),
                'category': obj.get('category', 'unknown'),
                'source': obj.get('source', 'unknown'),
                'metadata': {
                    'phase': obj.get('phase', ''),
                    'complexity_score': obj.get('complexity_score', 0),
                    'total_score': obj.get('total_score', 0)
                }
            }

            documents.append(doc)

    print(f"✓ Loaded {len(documents)} documents")
    return documents

def build_vector_db(documents, db_path="./knowledge_base"):
    """Build ChromaDB vector database"""
    print(f"\nBuilding vector database: {db_path}")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    # Create or get collection
    collection_name = "placement_specialist_kb"
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Placement and timing analysis specialist knowledge base"}
    )

    # Prepare documents for embedding
    ids = []
    texts = []
    metadatas = []

    for doc in documents:
        ids.append(doc['id'])

        # Combine prompt + response for embedding
        # This allows retrieval based on both question patterns and answer content
        text = f"QUESTION: {doc['prompt']}\n\nANSWER: {doc['response']}"
        texts.append(text)

        metadatas.append({
            'category': doc['category'],
            'source': doc['source'],
            'prompt': doc['prompt'][:500],  # Truncate for metadata storage
            'phase': doc['metadata']['phase'],
            'complexity_score': doc['metadata']['complexity_score']
        })

    # Add to collection (ChromaDB will handle embedding automatically)
    print(f"Adding {len(ids)} documents to collection...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    print(f"✓ Vector database built: {len(ids)} documents indexed")
    return collection

def test_retrieval(collection, query, top_k=3):
    """Test retrieval"""
    print(f"\n{'='*80}")
    print(f"TEST QUERY: {query}")
    print(f"{'='*80}\n")

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    print(f"Top {top_k} retrieved documents:\n")
    for idx, (doc_id, distance, metadata) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    ), 1):
        print(f"{idx}. Document: {doc_id} (distance: {distance:.3f})")
        print(f"   Category: {metadata['category']}")
        print(f"   Source: {metadata['source']}")
        print(f"   Prompt preview: {metadata['prompt'][:200]}...")
        print()

def main():
    print("="*80)
    print("BUILDING RAG KNOWLEDGE BASE")
    print("="*80)
    print()

    # Load corpus
    corpus_path = "../data/training/PLACEMENT_REFORMED_V2.jsonl"
    documents = load_corpus(corpus_path)

    # Build vector database
    collection = build_vector_db(documents)

    # Test retrievals
    print("\n" + "="*80)
    print("TESTING RETRIEVAL")
    print("="*80)

    test_queries = [
        "Why is wire delay so high on this net?",
        "Clock skew is causing timing violations",
        "How do I fix setup timing failures?",
    ]

    for query in test_queries:
        test_retrieval(collection, query, top_k=2)

    print("="*80)
    print("✓ KNOWLEDGE BASE READY")
    print("="*80)
    print(f"\nLocation: ./knowledge_base")
    print(f"Collection: placement_specialist_kb")
    print(f"Documents: {len(documents)}")

if __name__ == "__main__":
    main()
