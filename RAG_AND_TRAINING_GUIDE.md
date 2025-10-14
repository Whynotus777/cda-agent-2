# RAG and Fine-tuning Guide

This guide explains how to use the **RAG (Retrieval Augmented Generation)** system and **Fine-tuning pipeline** to make your CDA agent truly expert-level at chip design.

---

## 🎯 Quick Start

### Option A: RAG Only (Fastest - 30 minutes)

```bash
cd /home/quantumc1/cda-agent-claude

# 1. Install RAG dependencies
./venv/bin/pip install chromadb sentence-transformers beautifulsoup4 lxml PyPDF2

# 2. Scrape EDA documentation
python3 data/scrapers/eda_doc_scraper.py

# 3. Index into RAG system
python3 data/scrapers/index_knowledge_base.py

# 4. Done! Agent now uses documentation to answer
```

### Option B: Full System (Best - 2-3 hours)

```bash
# 1. Install all dependencies
./venv/bin/pip install -r requirements.txt

# 2. Scrape documentation AND code
python3 data/scrapers/eda_doc_scraper.py
python3 data/scrapers/verilog_github_scraper.py

# 3. Index documentation for RAG
python3 data/scrapers/index_knowledge_base.py

# 4. Prepare training data
python3 training/data_preparation/prepare_training_data.py

# 5. Fine-tune 8B model (optional, takes 1-2 hours)
python3 training/finetune_8b_chipdesign.py

# 6. Done! You now have RAG + specialized model
```

---

## 📚 What is RAG?

**RAG (Retrieval Augmented Generation)** enhances LLM responses by retrieving relevant documentation before answering.

### How It Works:

```
User asks: "How do I synthesize with Yosys?"
    ↓
1. RAG retrieves relevant Yosys documentation
    ↓
2. LLM receives query + retrieved docs
    ↓
3. LLM answers based on actual documentation
    ↓
User gets accurate, cited answer
```

### Benefits:
- ✅ **Factual accuracy** - Uses actual documentation
- ✅ **Up-to-date** - Can update docs without retraining
- ✅ **Transparent** - Can cite sources
- ✅ **Fast** - Works immediately, no training needed

---

## 🏗️ RAG System Architecture

```
Data Sources → Document Loader → Chunking → Embeddings → Vector Store
                                                                ↓
User Query → Embedding → Semantic Search → Retrieve Top-K → LLM Context
```

### Components:

1. **VectorStore** (`core/rag/vector_store.py`)
   - Uses ChromaDB for persistence
   - Stores document embeddings
   - Fast similarity search

2. **Embedder** (`core/rag/embedder.py`)
   - sentence-transformers: all-MiniLM-L6-v2
   - 384-dimensional embeddings
   - Optimized for semantic similarity

3. **DocumentLoader** (`core/rag/document_loader.py`)
   - Supports: .md, .txt, .pdf, .html, code files
   - Smart chunking (1000 chars with 200 overlap)
   - Preserves context

4. **RAGRetriever** (`core/rag/retriever.py`)
   - High-level interface
   - Retrieves top-K relevant docs
   - Formats context for LLM

---

## 📥 Data Sources

### 1. EDA Tool Documentation

**Scraper**: `data/scrapers/eda_doc_scraper.py`

**Sources**:
- Yosys synthesis documentation
- OpenROAD placement/routing docs
- DREAMPlace GPU placement docs
- OpenLane flow documentation
- Magic VLSI layout tool docs

**Run**:
```bash
python3 data/scrapers/eda_doc_scraper.py
```

**Output**: `data/knowledge_base/{tool}/*.md`

---

### 2. Verilog/SystemVerilog Repositories

**Scraper**: `data/scrapers/verilog_github_scraper.py`

**Repositories**:
- RISC-V cores (BOOM, Rocket Chip, VexRiscv, Ibex)
- ARM implementations
- NVIDIA DLA (Deep Learning Accelerator)
- SoC platforms (PULPissimo)
- SkyWater 130nm PDK

**Run**:
```bash
python3 data/scrapers/verilog_github_scraper.py
```

**Output**: `data/training/verilog_repos/*/`

---

## 🔍 Using RAG

### Indexing Documents

```bash
# Index all scraped documentation
python3 data/scrapers/index_knowledge_base.py
```

This creates a vector database at: `data/rag/chroma_db/`

### Testing RAG

```python
from core.rag import RAGRetriever

rag = RAGRetriever()

# Retrieve relevant docs
results = rag.retrieve("How do I synthesize with Yosys?", top_k=3)

for result in results:
    print(f"Source: {result['metadata']['source']}")
    print(f"Content: {result['document'][:200]}...")
```

### RAG in Agent

RAG is **automatically used** when the agent answers QUERY actions:

```
You: What is placement in chip design?
Agent: [Retrieves placement documentation]
Agent: [Answers using retrieved context]
```

The agent will:
1. Check if RAG index has documents
2. Retrieve top-3 relevant docs
3. Format as context for LLM
4. Answer using documentation

**Logs show**:
```
INFO - Using RAG for query: What is placement...
DEBUG - Retrieved 2847 chars of context
```

---

## 🎓 Fine-tuning

Fine-tuning creates a **specialized model** trained on chip design data.

### Why Fine-tune?

| RAG Only | Fine-tuned + RAG |
|----------|------------------|
| ✅ Fast to setup | ⏳ Takes 1-2 hours |
| ✅ Easy to update | 🔄 Need to retrain for updates |
| ⚠️  Relies on retrieval quality | ✅ Deep understanding |
| ⚠️  Can't reason about novel problems | ✅ Better reasoning |

**Recommendation**: Use both! RAG for facts, fine-tuned model for understanding.

---

### Training Data Preparation

**Script**: `training/data_preparation/prepare_training_data.py`

**Sources**:
1. EDA documentation → Q&A pairs
2. Verilog code → Code examples with comments
3. Synthetic Q&A → Generated templates

**Run**:
```bash
python3 training/data_preparation/prepare_training_data.py
```

**Output**: `data/training/chip_design_training.jsonl`

**Format**:
```jsonl
{"prompt": "What is synthesis?", "response": "Synthesis is...", "metadata": {...}}
{"prompt": "Show me a RISC-V module", "response": "module riscv_core...", "metadata": {...}}
```

---

### Fine-tuning with Ollama

**Script**: `training/finetune_8b_chipdesign.py`

**Process**:
1. Checks prerequisites (training data, Ollama, base model)
2. Creates Modelfile with training data
3. Runs `ollama create` to fine-tune
4. Tests the new model

**Run**:
```bash
python3 training/finetune_8b_chipdesign.py
```

**Creates**: `llama3:8b-chipdesign`

**Time**: 30 minutes - 2 hours depending on data size and hardware

---

### Using Fine-tuned Model

Update `configs/default_config.yaml`:

```yaml
llm:
  model_name: "llama3:8b-chipdesign"  # ← Use fine-tuned model
```

Or for triage routing:

```yaml
llm:
  triage:
    models:
      triage: "llama3.2:3b"
      moderate: "llama3:8b-chipdesign"  # ← Fine-tuned for moderate queries
      complex: "llama3:70b"
```

---

## 🚀 Complete Workflow

### Day 1: Setup RAG (30 minutes)

```bash
# Install RAG dependencies
./venv/bin/pip install chromadb sentence-transformers beautifulsoup4 lxml PyPDF2

# Scrape documentation
python3 data/scrapers/eda_doc_scraper.py

# Index for RAG
python3 data/scrapers/index_knowledge_base.py

# Test
python3 agent.py
> What is synthesis in Yosys?  # Should use RAG documentation
```

### Day 2: Collect Training Data (1-2 hours)

```bash
# Clone Verilog repositories
python3 data/scrapers/verilog_github_scraper.py

# Prepare training dataset
python3 training/data_preparation/prepare_training_data.py

# Review training data quality
cat data/training/chip_design_training.jsonl | head -10
```

### Day 3: Fine-tune Model (1-2 hours)

```bash
# Fine-tune 8B model
python3 training/finetune_8b_chipdesign.py

# Test fine-tuned model
ollama run llama3:8b-chipdesign "What is synthesis?"

# Update config to use new model
# Edit configs/default_config.yaml
```

### Day 4+: Continuous Improvement

- Add more documentation sources
- Collect user interaction logs
- Generate synthetic training data with 70B
- Retrain periodically with new data

---

## 📊 Expected Results

### With RAG Only:
```
You: How do I optimize area in Yosys?
Agent: [Retrieves Yosys optimization docs]
Agent: To optimize for area in Yosys, use these synthesis flags:
       - synth -top MODULE -abc9 -area
       - opt_clean
       [Detailed answer from documentation]
```

### With Fine-tuned Model:
```
You: Should I use TSMC 7nm or Samsung 5nm for a robotics SoC?
Agent: [Fine-tuned model reasoning + RAG facts]
Agent: For a robotics SoC, consider these factors:
       - Power efficiency: Both are good, but...
       - Cost: TSMC 7nm is more mature...
       - Availability: ...
       [Deep technical analysis]
```

### With Both (Recommended):
- Fast, accurate answers from RAG
- Deep reasoning from fine-tuned model
- Best of both worlds!

---

## 🛠️ Troubleshooting

### RAG Issues

**Problem**: "RAG index empty"
**Solution**:
```bash
python3 data/scrapers/eda_doc_scraper.py
python3 data/scrapers/index_knowledge_base.py
```

**Problem**: "chromadb not installed"
**Solution**:
```bash
./venv/bin/pip install chromadb sentence-transformers
```

**Problem**: "No results found"
**Solution**: Check if documents were indexed:
```python
from core.rag import RAGRetriever
rag = RAGRetriever()
print(rag.get_stats())  # Should show document_count > 0
```

### Fine-tuning Issues

**Problem**: "Training data not found"
**Solution**:
```bash
python3 training/data_preparation/prepare_training_data.py
```

**Problem**: "Base model not found"
**Solution**:
```bash
ollama pull llama3:8b
```

**Problem**: "Fine-tuning takes too long"
**Solution**: Reduce training data size or use smaller model (3B)

---

## 📁 Directory Structure

```
cda-agent-claude/
├── core/rag/                        # RAG system
│   ├── embedder.py                  # Sentence embeddings
│   ├── vector_store.py              # ChromaDB interface
│   ├── document_loader.py           # Load/chunk documents
│   └── retriever.py                 # High-level RAG interface
│
├── data/
│   ├── knowledge_base/              # Scraped documentation
│   │   ├── yosys/
│   │   ├── openroad/
│   │   └── dreamplace/
│   │
│   ├── training/
│   │   ├── verilog_repos/           # Cloned GitHub repos
│   │   └── chip_design_training.jsonl  # Training data
│   │
│   ├── rag/chroma_db/               # Vector database (auto-created)
│   │
│   └── scrapers/
│       ├── eda_doc_scraper.py       # Documentation scraper
│       ├── verilog_github_scraper.py  # Code scraper
│       └── index_knowledge_base.py  # RAG indexer
│
└── training/
    ├── data_preparation/
    │   └── prepare_training_data.py # Training data prep
    │
    └── finetune_8b_chipdesign.py    # Fine-tuning script
```

---

## 🎯 Next Steps

1. **Start with RAG** (quick win)
2. **Test with queries** about tools you use
3. **Collect more docs** specific to your needs
4. **Fine-tune when ready** (after collecting enough data)
5. **Keep improving** (add more sources, retrain periodically)

---

## 💡 Pro Tips

1. **Update RAG regularly**: Re-run scrapers monthly for latest docs
2. **Monitor quality**: Log which answers come from RAG vs. model knowledge
3. **Domain-specific data**: Add your own company/project documentation
4. **Synthetic data**: Use 70B to generate more training pairs
5. **Phase specialists**: Fine-tune separate models for synthesis, placement, timing, etc.

---

## 🤝 Contributing Data

Want to improve the knowledge base?

1. Add documentation sources to `eda_doc_scraper.py`
2. Add repos to `verilog_github_scraper.py`
3. Create custom training pairs in training data
4. Share with the community!

---

**Built by Claude Code for the CDA Agent project.**
**Making chip design AI truly expert-level! 🚀**
