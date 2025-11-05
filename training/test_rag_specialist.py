#!/usr/bin/env python3
"""
Test RAG-Augmented Reasoning Specialist

Architecture:
1. User Question → Vector DB Retrieval (top-k relevant knowledge)
2. Retrieved Context + Question → Reasoning Model
3. Reasoning Model → Synthesized Diagnostic Answer
"""

import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time

# Mission test prompts
PROMPTS = [
    {
        "id": 1,
        "difficulty": "8/10",
        "scenario": "High wire delay dominance",
        "prompt": "I am analyzing a critical path timing report for a mobile SoC. The path starts at the CPU register file and ends at the memory controller. The total path delay is 500ps, but the delay breakdown shows that a single net, net_cpu_to_mem_data[31], contributes 430ps of that delay. The rest of the path consists of standard logic cells with minimal delay. What is the most likely root cause of this timing violation?"
    },
    {
        "id": 2,
        "difficulty": "9/10",
        "scenario": "Clock skew masking",
        "prompt": "The second critical path in my design is in the instruction decoder. The data path delay is high, but it is only failing setup timing by 35ps. I've noticed in the report that the clock path to the endpoint flip-flop has a latency of 110ps, while the clock path to the start point flip-flop is only 40ps. Given this information, what is the most probable true source of the timing failure?"
    },
    {
        "id": 3,
        "difficulty": "10/10",
        "scenario": "Clock domain crossing",
        "prompt": "The third critical path is a control signal that travels from the 2GHz main CPU clock domain to the 500MHz peripheral bus clock domain. The timing tool is reporting a catastrophic setup violation of -1500ps, and no amount of buffer insertion seems to fix it. What is the most likely architectural or constraints-related reason for this massive, unfixable timing error?"
    }
]

def load_rag_system(model_path: str, base_model: str, kb_path: str):
    """Load RAG system: knowledge base + reasoning model"""

    print("="*80)
    print("LOADING RAG SYSTEM")
    print("="*80)
    print(f"Knowledge Base: {kb_path}")
    print(f"Reasoning Model Base: {base_model}")
    print(f"Reasoning Model Adapters: {model_path}")
    print()

    # Load knowledge base
    print("Loading knowledge base...")
    kb_client = chromadb.PersistentClient(path=kb_path)
    collection = kb_client.get_collection("placement_specialist_kb")
    print(f"✓ Knowledge base loaded ({collection.count()} documents)")
    print()

    # Load reasoning model
    print("Loading reasoning model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    model = PeftModel.from_pretrained(base, model_path)
    model.eval()

    print("✓ Reasoning model loaded")
    print()

    return collection, model, tokenizer

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
            'source': metadata['source']
        })

    return contexts

def format_rag_prompt(query, contexts):
    """Format prompt with retrieved context"""
    context_str = "RETRIEVED KNOWLEDGE:\n\n"
    for idx, ctx in enumerate(contexts, 1):
        context_str += f"[Document {idx}] Source: {ctx['source']}\n"
        context_str += f"{ctx['text']}\n\n"

    prompt = f"{context_str}USER QUESTION:\n{query}\n\nDIAGNOSTIC ANALYSIS:"
    return prompt

def test_rag_specialist(collection, model, tokenizer, prompts):
    """Test RAG specialist on mission prompts"""

    print("="*80)
    print("MISSION TEST: RAG-AUGMENTED REASONING SPECIALIST")
    print("="*80)
    print(f"Prompts: {len(prompts)}")
    print()

    results = []

    for p in prompts:
        print("="*80)
        print(f"SCENARIO {p['id']}: {p['scenario']} (Difficulty: {p['difficulty']})")
        print("="*80)
        print(f"\nQUESTION:")
        print(f"{p['prompt']}")
        print()

        # Retrieve context
        print("→ Retrieving relevant knowledge...")
        contexts = retrieve_context(collection, p['prompt'], top_k=2)
        for idx, ctx in enumerate(contexts, 1):
            print(f"  [{idx}] {ctx['source'][:80]}...")
        print()

        # Format RAG prompt
        rag_prompt = format_rag_prompt(p['prompt'], contexts)

        # Generate
        print("-"*80)
        print("SPECIALIST RESPONSE:")
        print("-"*80)

        formatted = f"Q: {rag_prompt}\n\nA:"

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.3,  # Lower temperature for more focused reasoning
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        elapsed = time.time() - start_time

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "A:" in generated:
            answer = generated.split("A:", 1)[1].strip()
            if "\nQ:" in answer:
                answer = answer.split("\nQ:")[0].strip()
        else:
            answer = generated

        print(answer)
        print()
        print(f"[Response time: {elapsed:.1f}s]")
        print()

        results.append({
            "scenario": p['scenario'],
            "difficulty": p['difficulty'],
            "prompt": p['prompt'],
            "response": answer,
            "time": elapsed,
            "retrieved_sources": [ctx['source'] for ctx in contexts]
        })

    return results

def main():
    base_model = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    reasoning_model_path = "./models/reasoning-specialist-rag-v1"
    knowledge_base_path = "./knowledge_base"

    # Load RAG system
    collection, model, tokenizer = load_rag_system(
        reasoning_model_path,
        base_model,
        knowledge_base_path
    )

    # Run tests
    results = test_rag_specialist(collection, model, tokenizer, PROMPTS)

    # Summary
    print("="*80)
    print("MISSION TEST COMPLETE")
    print("="*80)
    print(f"Scenarios tested: {len(results)}")
    print(f"Total response time: {sum(r['time'] for r in results):.1f}s")
    print()
    print("RAG Architecture:")
    print("  ✓ Knowledge externalized in vector database")
    print("  ✓ Model trained only on reasoning/synthesis")
    print("  ✓ Context retrieved dynamically per question")
    print("="*80)

if __name__ == "__main__":
    main()
