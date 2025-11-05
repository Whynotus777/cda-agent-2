# 🎯 Chat Interface Guide - Chip Design AI Models

## What You Have Now

**Two AI models** deployed with simple chat interfaces:

1. **GPT-2 Medium** (355M params) - Ready immediately ⚡
2. **Mixtral-8x7B** (47B params) - Better quality, downloads on first run ⭐

Both trained on **~200 examples** from chip design papers, real chips, and EDA tools.

## Quick Start (30 seconds)

```bash
cd /home/quantumc1/cda-agent-2C1
./launch_apprentice_chat.sh
```

Type: **"What is cell placement?"** and press Enter.

## Confirmed Facts

✅ **Training Data:** ~200 curated examples from:
   - Academic papers (DAC, ICCAD, ISPD)
   - Real chips (OpenTitan, TPU, Rocket Chip, etc.)
   - EDA documentation (DREAMPlace, OpenROAD, OpenLane)

✅ **Base Models:** GPT-2 (OpenAI) + Mixtral (Mistral AI)
   - **NOT** Meta's Llama

✅ **Deployment:** Local, on your GPU with CUDA

✅ **Chat Interface:** Simple command-line, ready to use

## Two Ways to Chat

### Option 1: GPT-2 (Instant, Fast)
```bash
./launch_apprentice_chat.sh
```
- 1-3 second responses
- Good for quick factual questions
- Ready immediately

### Option 2: Mixtral (Better Quality)
```bash
./launch_mixtral_chat.sh
```
- 5-15 second responses
- Better for complex reasoning
- Downloads ~94GB on first run (30 min)

## Example Interaction

```bash
$ ./launch_apprentice_chat.sh

==========================================
Chip Design Placement Apprentice
==========================================

Model: placement-apprentice-v2 (GPT-2 Medium)
...

You: What is cell placement?

Apprentice: Cell placement is the process of determining physical locations
for standard cells in a chip design. It aims to optimize wirelength, timing,
and routability while meeting density constraints...

You: How does DREAMPlace work?

Apprentice: DREAMPlace uses GPU acceleration and deep learning optimization
techniques for placement. It achieves 30x speedup over CPU implementations...

You: quit

Goodbye! Thanks for chatting about chip design!
```

## Commands Reference

### Launch Scripts:
```bash
./launch_apprentice_chat.sh     # GPT-2 (fast)
./launch_mixtral_chat.sh        # Mixtral (better quality)
```

### Direct Python Calls:
```bash
# GPT-2 single question
python3 chat_with_apprentice.py --question "What is placement?"

# Mixtral single question
python3 chat_with_specialist.py --4bit --question "What is placement?"

# Interactive GPT-2
python3 chat_with_apprentice.py

# Interactive Mixtral
python3 chat_with_specialist.py --4bit
```

### Within Chat:
- **Type your question** → Get answer
- **help** → Show example questions
- **quit** or **exit** → Leave chat

## Example Questions

### Simple (Good for GPT-2):
1. "What is cell placement?"
2. "Define legalization"
3. "What does DREAMPlace do?"
4. "List main placement algorithms"

### Complex (Best for Mixtral):
1. "Why is wire delay so high? Walk me through diagnosis."
2. "Compare analytical vs simulated annealing placement"
3. "How do I optimize for both timing and power?"
4. "Analyze this placement failure and suggest fixes"

## Model Comparison

| Feature | GPT-2 | Mixtral |
|---------|-------|---------|
| **Response Time** | 1-3 sec | 5-15 sec |
| **Quality** | Good | Excellent |
| **Reasoning** | Basic | Strong |
| **First Load** | Instant | 30 min download |
| **VRAM Usage** | 1-2GB | ~25GB (4-bit) |

## Documentation Map

- **`QUICKSTART.md`** - GPT-2 30-second start
- **`MIXTRAL_QUICKSTART.md`** - Mixtral setup guide
- **`MODEL_COMPARISON.md`** - Detailed comparison
- **`APPRENTICE_CHAT_README.md`** - Complete reference
- **`SETUP_COMPLETE.md`** - Deployment summary

## System Status

✅ **GPU:** CUDA available (PyTorch 2.8.0)
✅ **Disk:** 4.8TB free
✅ **Dependencies:** torch, transformers, peft, bitsandbytes
✅ **Models:** GPT-2 ready, Mixtral adapter ready

## Troubleshooting

### "Model not found"
```bash
ls -lh training/models/placement-apprentice-v2/
ls -lh training/models/wisdom-specialist-v3-mixtral/
```

### "Out of memory" (Mixtral)
```bash
# Always use 4-bit for Mixtral:
python3 chat_with_specialist.py --4bit
```

### "Slow responses"
```bash
# Use GPU (should be automatic)
nvidia-smi  # Check GPU status

# Or use faster GPT-2 model
./launch_apprentice_chat.sh
```

## Training Data Details

**GPT-2:** 198 examples from `PLACEMENT_GOLD_STANDARD.jsonl`
**Mixtral:** 17 examples from `WISDOM_CORPUS.jsonl`

Both include:
- Papers: DREAMPlace (DAC 2019), RePlAce, NTUPlacerDR, etc.
- Chips: OpenTitan, TPU v1, Rocket Chip, BlackParrot, etc.
- Tools: DREAMPlace, OpenROAD, OpenLane documentation

## Next Steps

1. **Start with GPT-2 (instant):**
   ```bash
   ./launch_apprentice_chat.sh
   ```

2. **Try some questions:**
   - "What is cell placement?"
   - "How does DREAMPlace work?"
   - "Explain legalization"

3. **Compare with Mixtral:**
   ```bash
   ./launch_mixtral_chat.sh
   ```
   Ask the same questions and compare!

4. **Read detailed comparison:**
   ```bash
   cat MODEL_COMPARISON.md
   ```

---

## Summary

✅ **GPT-2 Model:** Ready to use, fast (1-3 sec)
⏳ **Mixtral Model:** Better quality, downloads first time (~30 min)
📚 **Training:** ~200 examples from papers and real chips
🎯 **Interface:** Simple chat, just ask questions
📖 **Documentation:** Complete guides created

**Start chatting now:**
```bash
cd /home/quantumc1/cda-agent-2C1 && ./launch_apprentice_chat.sh
```
