#!/usr/bin/env python3
"""
Create Reasoning Training Corpus for RAG Architecture

Format: (Retrieved Context + Question) → (Synthesized Diagnostic Answer)

This trains the model ONLY on synthesis, not memorization.
"""

import json
import chromadb

def retrieve_context(collection, query, top_k=2):
    """Retrieve relevant context from knowledge base"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    contexts = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        contexts.append({
            'text': doc,
            'source': metadata['source'],
            'category': metadata['category']
        })

    return contexts

def format_training_example(query, contexts, target_response):
    """Format as training example: Input = Context + Query, Output = Synthesized Answer"""

    # Build context section
    context_str = "RETRIEVED KNOWLEDGE:\n\n"
    for idx, ctx in enumerate(contexts, 1):
        context_str += f"[Document {idx}] Source: {ctx['source']}\n"
        context_str += f"{ctx['text']}\n\n"

    # Build input
    input_text = f"{context_str}USER QUESTION:\n{query}\n\nDIAGNOSTIC ANALYSIS:"

    # Target is the synthesized answer
    output_text = target_response

    return {
        'input': input_text,
        'output': output_text,
        'query': query,
        'num_contexts': len(contexts)
    }

def main():
    print("="*80)
    print("CREATING REASONING TRAINING CORPUS")
    print("="*80)
    print()

    # Load knowledge base
    print("Loading knowledge base...")
    client = chromadb.PersistentClient(path="./knowledge_base")
    collection = client.get_collection("placement_specialist_kb")
    print(f"✓ Knowledge base loaded\n")

    # Load diagnostic corpus (contains the target reasoning chains)
    print("Loading diagnostic corpus...")
    diagnostic_examples = []
    with open('DIAGNOSTIC_CORPUS.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                diagnostic_examples.append(json.loads(line))
    print(f"✓ Loaded {len(diagnostic_examples)} diagnostic examples\n")

    # Create reasoning training examples
    reasoning_corpus = []

    for idx, example in enumerate(diagnostic_examples, 1):
        query = example['prompt']
        target_response = example['response']

        print(f"Example {idx}/{len(diagnostic_examples)}")
        print(f"Query: {query[:100]}...")

        # Retrieve relevant context
        contexts = retrieve_context(collection, query, top_k=2)
        print(f"Retrieved {len(contexts)} context documents")

        # Format as reasoning training example
        training_ex = format_training_example(query, contexts, target_response)

        reasoning_corpus.append({
            'text': f"Q: {training_ex['input']}\n\nA: {training_ex['output']}",
            'metadata': {
                'query': query[:200],
                'num_contexts': len(contexts),
                'source': 'reasoning_synthesis'
            }
        })

        print(f"✓ Formatted training example\n")

    # Save reasoning corpus
    output_path = "REASONING_CORPUS.jsonl"
    print(f"Saving reasoning corpus: {output_path}")

    with open(output_path, 'w') as f:
        for example in reasoning_corpus:
            f.write(json.dumps(example) + '\n')

    print(f"✓ Saved {len(reasoning_corpus)} reasoning training examples")

    # Print sample
    print("\n" + "="*80)
    print("SAMPLE TRAINING EXAMPLE")
    print("="*80)
    print()
    print("INPUT (Context + Query):")
    print("-" * 80)
    print(reasoning_corpus[0]['text'].split("A:")[0][:800] + "...")
    print()
    print("OUTPUT (Synthesized Answer):")
    print("-" * 80)
    print(reasoning_corpus[0]['text'].split("A:")[1][:800] + "...")

    print("\n" + "="*80)
    print("✓ REASONING CORPUS READY FOR TRAINING")
    print("="*80)
    print(f"\nCorpus: {output_path}")
    print(f"Examples: {len(reasoning_corpus)}")
    print(f"\nThis corpus trains the model on:")
    print("  - Integrating retrieved context with user questions")
    print("  - Synthesizing diagnostic reasoning chains")
    print("  - NOT memorizing facts (facts come from retrieval)")

if __name__ == "__main__":
    main()
