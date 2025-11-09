# RAG Integration Pattern

**CRITICAL**: This document defines the architectural pattern for ensuring RAG is **NEVER LOST** from the pipeline.

## Problem Statement

Previously, RAG was lost from the pipeline during various iterations. This happened because RAG was treated as an optional add-on rather than a core architectural component.

## Solution: Architectural RAG Integration

We've created `core/rag/rag_client.py` - a wrapper around the Anthropic API client that **automatically** injects RAG context into all API calls. This ensures RAG is built into the client layer, not manually added to each tool.

## The RAG-Enhanced Client

### Key Features

1. **Automatic RAG Injection**: RAG context is automatically retrieved and injected into prompts
2. **Graceful Degradation**: If RAG fails or is unavailable, the client continues working without RAG
3. **Configurable**: Can be enabled/disabled, with customizable retrieval parameters
4. **Training Data Indexing**: Built-in method to index training data into the vector store

### Basic Usage

```python
from core.rag import RAGEnhancedClient

# Initialize client (RAG enabled by default)
client = RAGEnhancedClient(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    rag_persist_dir="./data/rag/chroma_db",
    enable_rag=True  # Default: True
)

# Generate with automatic RAG injection
response = await client.generate_with_rag(
    user_prompt="Design a Moore FSM with 4 states...",
    query_for_rag="Moore FSM design patterns",  # Optional: defaults to user_prompt
    system_prompt="You are an expert HDL designer.",
    model="claude-sonnet-4-20250514",
    max_tokens=8000
)

# Extract response
code = response.content[0].text
```

### Advanced Usage

```python
# Custom RAG retrieval parameters
response = await client.generate_with_rag(
    user_prompt="Design a Mealy FSM...",
    query_for_rag="Mealy FSM examples",
    rag_top_k=5,  # Retrieve 5 documents (default: 3)
    rag_max_context=3000,  # Max RAG context length (default: 2000)
)

# Disable RAG for specific call
response = await client.generate_with_rag(
    user_prompt="Hello",
    rag_top_k=0  # Skip RAG retrieval
)

# Check RAG stats
stats = client.get_stats()
print(stats)  # Shows RAG enabled status, document count, etc.
```

## Pattern 1: Data Generation Scripts

### Example: FSM Synthetic Generation

**Location**: `scripts/generate_fsm_synthetic_v5_6.py`

```python
#!/usr/bin/env python3
from core.rag import RAGEnhancedClient

async def main():
    # Initialize RAG-enhanced client
    client = RAGEnhancedClient(enable_rag=True)

    # Index training data into RAG (run once)
    client.index_training_data(
        dataset_path="data/rtl_behavioral_v5_5.jsonl",
        sample_limit=100
    )

    # Generate with RAG
    for category in ["Moore", "Mealy", "Handshake"]:
        response = await client.generate_with_rag(
            user_prompt=f"Design a {category} FSM with...",
            query_for_rag=f"{category} FSM design patterns",
            system_prompt=SYSTEM_PROMPT
        )

        code = extract_code(response.content[0].text)
        # ... validation ...
```

**Key Points**:
- Use `RAGEnhancedClient` instead of `AsyncAnthropic`
- Index training data once at startup
- Use `query_for_rag` parameter to specify RAG query

## Pattern 2: Benchmarking Scripts

### Example: Model Benchmarking with RAG

**Location**: `scripts/benchmark_v5_4.py` (to be updated)

```python
#!/usr/bin/env python3
from core.rag import RAGEnhancedClient

async def benchmark_with_rag(model_path: str, test_dataset: str):
    # Initialize RAG-enhanced client
    client = RAGEnhancedClient(enable_rag=True)

    # Index reference examples
    client.index_training_data(
        dataset_path="data/rtl_behavioral_v5_5.jsonl",
        sample_limit=100
    )

    results = []

    for example in load_test_dataset(test_dataset):
        # Get model prediction (NO RAG - just inference)
        model_output = run_model_inference(model_path, example['instruction'])

        # Get Claude baseline WITH RAG
        claude_response = await client.generate_with_rag(
            user_prompt=example['instruction'],
            query_for_rag=extract_query_from_instruction(example),
            system_prompt="You are an expert HDL designer."
        )
        claude_output = extract_code(claude_response.content[0].text)

        # Compare
        model_correct = validate_output(model_output, example['expected'])
        claude_correct = validate_output(claude_output, example['expected'])

        results.append({
            'model_correct': model_correct,
            'claude_correct': claude_correct,
            'category': example.get('hierarchy', {}).get('l2')
        })

    return analyze_results(results)
```

**Key Points**:
- Use RAG for Claude baseline comparisons
- Index reference examples at startup
- Extract meaningful RAG queries from instructions

## Pattern 3: Interactive Tools

### Example: Web UI / API Server

```python
from fastapi import FastAPI
from core.rag import RAGEnhancedClient

app = FastAPI()

# Initialize RAG client at startup (singleton)
rag_client = None

@app.on_event("startup")
async def startup():
    global rag_client
    rag_client = RAGEnhancedClient(enable_rag=True)

    # Index training data
    rag_client.index_training_data(
        dataset_path="data/rtl_behavioral_v5_5.jsonl",
        sample_limit=100
    )

@app.post("/generate")
async def generate_hdl(request: GenerateRequest):
    response = await rag_client.generate_with_rag(
        user_prompt=request.prompt,
        query_for_rag=request.prompt,
        system_prompt=request.system_prompt
    )

    return {
        "code": extract_code(response.content[0].text),
        "rag_stats": rag_client.get_stats()
    }
```

## Indexing Training Data

The RAG client can automatically index training data into the vector store:

```python
from core.rag import RAGEnhancedClient

client = RAGEnhancedClient()

# Index V5.5 training data (100 examples)
num_indexed = client.index_training_data(
    dataset_path="data/rtl_behavioral_v5_5.jsonl",
    sample_limit=100
)

print(f"Indexed {num_indexed} examples into RAG")
```

**What gets indexed:**
- Instruction text
- Generated code
- Hierarchy metadata (L1, L2, L3)

**How to query:**
- Use specific keywords from the instruction
- Reference hierarchy categories (e.g., "Moore FSM examples")
- Reference design patterns (e.g., "counter with enable")

## RAG Query Best Practices

### Good RAG Queries

```python
# Specific FSM type
query_for_rag = "Moore FSM design patterns"

# Specific functionality
query_for_rag = "counter with parallel load and enable"

# Specific hierarchy
query_for_rag = "Sequential FSM Handshake examples"

# Specific dialect
query_for_rag = "SystemVerilog Moore FSM always_ff"
```

### Bad RAG Queries

```python
# Too generic
query_for_rag = "FSM"  # Too broad

# Too specific (overfitting)
query_for_rag = "4-bit counter with asynchronous reset at address 0x100"  # Too narrow
```

## Disabling RAG (Emergency Only)

If RAG causes issues, it can be disabled:

```python
# Disable at initialization
client = RAGEnhancedClient(enable_rag=False)

# Or disable in environment
export DISABLE_RAG=1
```

**NOTE**: Disabling RAG should be a last resort. If RAG is causing issues, fix the root cause rather than disabling it.

## Migration Guide

### Before (Direct Anthropic Client)

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = await client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    messages=[{"role": "user", "content": prompt}]
)
```

### After (RAG-Enhanced Client)

```python
from core.rag import RAGEnhancedClient

client = RAGEnhancedClient()  # RAG enabled by default

response = await client.generate_with_rag(
    user_prompt=prompt,
    query_for_rag=extract_query(prompt),  # Add RAG query
    model="claude-sonnet-4-20250514",
    max_tokens=8000
)
```

## Checklist for New Tools

When creating a new tool that uses the Claude API:

- [ ] Import `RAGEnhancedClient` instead of `AsyncAnthropic`
- [ ] Initialize client with `enable_rag=True` (default)
- [ ] Index relevant training data at startup
- [ ] Use `generate_with_rag()` instead of `messages.create()`
- [ ] Provide meaningful `query_for_rag` parameter
- [ ] Add RAG stats to logging/debugging output

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│         Your Tool/Script                │
│  (generation, benchmarking, etc.)       │
└──────────────┬──────────────────────────┘
               │
               │ uses
               ▼
┌─────────────────────────────────────────┐
│       RAGEnhancedClient                 │
│  core/rag/rag_client.py                 │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ 1. Retrieve RAG context         │   │
│  │    from vector store            │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ 2. Inject context into prompt   │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ 3. Call Anthropic API           │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │
               │ calls
               ▼
┌─────────────────────────────────────────┐
│    Anthropic API (Claude)               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│      RAG Vector Store                   │
│   (ChromaDB - persistent)               │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Training Examples (indexed)    │   │
│  │  - Instructions                 │   │
│  │  - Code                         │   │
│  │  - Hierarchy metadata           │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Testing RAG Integration

### Smoke Test

```bash
cd ~/cda-agent-2C1

# Test RAG client
python3 << 'EOF'
import asyncio
from core.rag import RAGEnhancedClient

async def test():
    client = RAGEnhancedClient(enable_rag=True)

    # Index some data
    num_indexed = client.index_training_data(
        "data/rtl_behavioral_v5_5.jsonl",
        sample_limit=10
    )
    print(f"Indexed: {num_indexed} examples")

    # Test generation
    response = await client.generate_with_rag(
        user_prompt="Design a simple 4-bit counter",
        query_for_rag="counter design examples"
    )

    print(f"Response length: {len(response.content[0].text)} chars")
    print(f"RAG stats: {client.get_stats()}")

asyncio.run(test())
EOF
```

## Summary

**NEVER** use `AsyncAnthropic` or `Anthropic` directly in new tools. **ALWAYS** use `RAGEnhancedClient` to ensure RAG is permanently integrated into the pipeline.

This architectural pattern ensures:
1. RAG is built into the client layer
2. RAG cannot be "forgotten" or "lost"
3. RAG works consistently across all tools
4. RAG can be disabled if needed (emergency only)
5. All tools benefit from RAG improvements automatically
