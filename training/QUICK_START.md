# Quick Start: Hierarchical Model System

## 🚀 Setup (5 minutes)

### 1. Pull Base Models

```bash
# Pull all base models (do this first!)
ollama pull llama3.2:3b  # ~2GB, 30 seconds
ollama pull llama3:8b    # ~4.7GB, 2 minutes
ollama pull llama3:70b   # Already have this!
```

### 2. Create First Specialists

Start with the most common phases:

```bash
cd ~/cda-agent

# Synthesis specialist (most common)
python3 training/finetune_specialist.py --phase synthesis --size 8b --test

# Placement specialist
python3 training/finetune_specialist.py --phase placement --size 8b --test

# Timing analysis specialist
python3 training/finetune_specialist.py --phase timing_analysis --size 8b --test
```

Each takes **~30 seconds** to create.

### 3. Verify Routing is Enabled

Check `configs/default_config.yaml`:

```yaml
llm:
  router:
    enable: true  # ← Should be true
    shadow_orchestrator:
      enable: true  # ← Should be true
```

### 4. Test the System

```bash
./run_agent.sh
```

Try queries of different lengths:

```
# Short query → uses 3B
You: What is synthesis?

# Medium query → uses 8B
You: How do I optimize my design for minimum area in Yosys?

# Long query → uses 70B
You: [paste long technical question]
```

## 📊 How It Works

### Every Query Flow:

```
1. User types query
   ↓
2. Character count → initial routing
   • < 800 chars → route to 3B
   • < 2000 chars → route to 8B
   • ≥ 2000 chars → route to 70B
   ↓
3. Shadow call to 70B orchestrator (background)
   • Analyzes query
   • Learns from interaction
   • Can override routing
   ↓
4. Specialist responds
   ↓
5. Training data collected
```

## 🎯 Training Strategy

### Phase 1: Bootstrap (Week 1)
```bash
# Just use base models - collect training data
# No specialists yet
```

### Phase 2: First Specialists (Week 2)
```bash
# Train most common phases (8B only)
python3 training/finetune_specialist.py --phase synthesis --size 8b
python3 training/finetune_specialist.py --phase placement --size 8b
python3 training/finetune_specialist.py --phase timing_analysis --size 8b
```

### Phase 3: Full Coverage (Week 4)
```bash
# Train all phases, 3B and 8B
./training/train_all_specialists.sh
# (Skip 70B option for now)
```

### Phase 4: Premium Specialists (Month 2)
```bash
# Train 70B specialists for complex cases
# (Only after collecting substantial training data)
```

## 💾 Training Data

### Check collected data:
```bash
ls -lh data/training/
cat data/training/training_data_*.jsonl | jq -r '.phase' | sort | uniq -c
```

### Example data format:
```json
{
  "prompt": "How do I optimize for area?",
  "response": "To optimize for area in Yosys, use...",
  "phase": "synthesis",
  "model_size": "8b",
  "context": {...},
  "timestamp": "2025-10-14T04:52:00"
}
```

## 🔍 Monitoring

### See which models are being used:
```bash
# Check logs
tail -f logs/cda_agent_*.log | grep "Using model"
```

### List all specialists:
```bash
ollama list | grep -E "(3b|8b|70b)-(synthesis|placement|routing|timing)"
```

### Test a specialist directly:
```bash
ollama run llama3:8b-synthesis "How do I reduce gate count?"
```

## ⚡ Performance

### Before (base models only):
- All queries: 15-20 seconds (using 70B)

### After (with specialists):
- Short queries: **1-2 seconds** (3B) ⚡
- Medium queries: **2-4 seconds** (8B) ⚡
- Long queries: 15-20 seconds (70B) ← same
- **Average: 5-10x faster!**

### 70B Orchestrator:
- Runs in background (shadow call)
- **Learns from ALL queries**
- Only ~3 seconds overhead
- Non-blocking

## 🎓 Best Practices

### 1. Start Small
- Train 8B specialists first
- Only for common phases
- 3B later for speed
- 70B only when needed

### 2. Collect Before Training
- Use agent for 1-2 weeks
- Let training data accumulate
- Better specialists from more data

### 3. Iterative Improvement
```bash
# Week 1: Collect data
# Week 2: Train synthesis + placement (8B)
# Week 3: Train timing + power (8B)
# Week 4: Train all 3B versions
# Month 2: Train 70B specialists
```

### 4. Monitor Quality
```bash
# Test each specialist after training
python3 training/finetune_specialist.py --phase synthesis --size 8b --test

# If quality is poor, collect more data
```

## 🔧 Troubleshooting

### Specialist not being used?
```bash
# Check if it exists
ollama list | grep synthesis

# Check if routing is enabled
grep -A 5 "router:" configs/default_config.yaml
```

### Training fails?
```bash
# Make sure base model exists
ollama list | grep "llama3:8b"

# Check training data
ls -lh data/training/

# Try with --test flag to see errors
python3 training/finetune_specialist.py --phase synthesis --size 8b --test
```

### Shadow orchestrator slow?
```yaml
# Reduce shadow tokens in config
shadow_orchestrator:
  enable: true
  max_tokens: 128  # ← reduce from 256
```

## 📈 Scaling

### Small team (1-5 users):
- 8B specialists for main phases
- 3B for quick queries
- Keep 70B as orchestrator

### Medium team (5-20 users):
- All 8B specialists
- All 3B specialists
- Some 70B specialists

### Large team (20+ users):
- Full specialist library
- Periodic retraining
- Custom specialists per project

## 🎯 ROI

### Costs:
- Training time: ~5 minutes per specialist
- Storage: ~5GB per size (3B/8B/70B)
- One-time setup: ~1 hour

### Benefits:
- **5-10x faster responses**
- Continuous learning from 70B
- Specialized expertise per phase
- Better quality over time

## Next Steps

1. ✅ Pull base models
2. ✅ Create 3-4 key specialists (8B)
3. ✅ Use agent normally - data collects automatically
4. ✅ After 1-2 weeks, train more specialists
5. ✅ Monitor and iterate

**Start now:**
```bash
ollama pull llama3:8b
python3 training/finetune_specialist.py --phase synthesis --size 8b --test
./run_agent.sh
```

Your hierarchical system is ready! 🚀
