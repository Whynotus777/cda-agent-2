# 🚀 Mixtral Specialist - Quick Start Guide

## What's Different Now?

You now have access to **Mixtral-8x7B-Instruct** (47B parameters) with a specialized chip design adapter!

### Model Comparison

| Model | Parameters | Base | Best For |
|-------|------------|------|----------|
| **Mixtral-8x7B** ⭐ | 47B | Mistral AI | Everything - best quality |
| GPT-2 Medium | 355M | OpenAI | Fast inference, simple Q&A |

**Mixtral is 132x larger and significantly more capable!**

## Quick Start

### Interactive Chat (Recommended)
```bash
cd /home/quantumc1/cda-agent-2C1
./launch_mixtral_chat.sh
```

### Single Question
```bash
python3 chat_with_specialist.py --question "What is cell placement?"
```

### With 4-bit Quantization (Saves Memory)
```bash
python3 chat_with_specialist.py --4bit
```

## Important Notes

### First Run
⚠️ **The first run will download Mixtral-8x7B (~94GB)**
- This only happens once
- Requires ~100GB free disk space
- Takes 10-30 minutes depending on internet speed
- Subsequent runs will be instant!

### Memory Requirements
- **Full precision:** ~100GB VRAM (not practical)
- **4-bit quantization:** ~25GB VRAM (recommended) ✓
- **CPU mode:** Works but very slow (~5-10 min per response)

Your system has **GPU with CUDA**, so 4-bit mode should work well!

## Usage Examples

### Interactive Chat
```bash
./launch_mixtral_chat.sh

# Then ask questions:
You: What is cell placement in chip design?
You: How does DREAMPlace achieve GPU acceleration?
You: Explain timing-driven placement
You: quit
```

### Single Questions
```bash
# Quick one-off questions
python3 chat_with_specialist.py --question "What is legalization?"

# Use 4-bit for faster loading
python3 chat_with_specialist.py --4bit --question "Compare DREAMPlace and RePlAce"
```

### Switch Between Models
```bash
# Use Mixtral (default)
python3 chat_with_specialist.py

# Use GPT-2 for faster responses
python3 chat_with_specialist.py --model ./training/models/placement-apprentice-v2

# Use other specialists
python3 chat_with_specialist.py --model ./training/models/wisdom-specialist-v2
```

## Model Details

### Mixtral-8x7B-Instruct
- **Architecture:** Mixture of Experts (8 experts, 7B each)
- **Active Parameters:** ~13B per token (efficient!)
- **Total Parameters:** 47B
- **Context Length:** 32k tokens
- **Training:** Trained by Mistral AI, fine-tuned with LoRA on chip design

### LoRA Adapter (wisdom-specialist-v3)
- **Size:** 53MB (tiny adapter on top of base model)
- **Training:** ~200 examples from:
  - WISDOM_CORPUS.jsonl (chip design wisdom)
  - Academic papers and research
  - Real chip implementations
  - EDA tool documentation

## Expected Performance

### Response Quality
- **Mixtral:** Excellent - nuanced, detailed, accurate
- **GPT-2:** Good - concise, sometimes oversimplified

### Response Speed (with 4-bit quantization)
- **Mixtral:** ~5-15 seconds per response
- **GPT-2:** ~1-3 seconds per response

### Reasoning Capability
- **Mixtral:** Strong reasoning, can handle complex scenarios
- **GPT-2:** Basic reasoning, better for simple Q&A

## Troubleshooting

### Model download too slow
The base Mixtral model is large. Be patient on first run!
- Check disk space: `df -h`
- Check internet speed
- Consider downloading overnight

### Out of memory error
```bash
# Use 4-bit quantization
python3 chat_with_specialist.py --4bit

# Or force CPU (very slow)
python3 chat_with_specialist.py --device cpu
```

### Import errors
```bash
# Install missing dependencies
pip install torch transformers peft accelerate bitsandbytes
```

### Model not found
```bash
# Verify the adapter exists
ls -lh training/models/wisdom-specialist-v3-mixtral/

# If missing, you may need to train it first
```

## What Can Mixtral Do Better?

Compared to GPT-2, Mixtral excels at:

1. **Complex Reasoning:** Multi-step problem solving
2. **Detailed Explanations:** Comprehensive technical answers
3. **Context Understanding:** Better at following conversation history
4. **Technical Accuracy:** More precise with terminology and concepts
5. **Scenario Analysis:** Can handle "what if" questions better

## Example Questions to Try

### For Mixtral (leverage its reasoning)
1. "Why is wire delay so high on this net, and how would you diagnose it?"
2. "Compare the tradeoffs between analytical and simulated annealing placement"
3. "Walk me through the entire placement flow for a modern SoC"
4. "What are the key differences between 7nm and 28nm placement?"
5. "How would you optimize placement for both timing and power?"

### For GPT-2 (quick factual Q&A)
1. "What is cell placement?"
2. "List the main placement algorithms"
3. "What does DREAMPlace do?"
4. "Define legalization"

## Configuration Files

All in `/home/quantumc1/cda-agent-2C1/`:

- ✅ `chat_with_specialist.py` - Main interface (supports both models)
- ✅ `launch_mixtral_chat.sh` - Mixtral launcher
- ✅ `launch_apprentice_chat.sh` - GPT-2 launcher (faster)
- ✅ `training/models/wisdom-specialist-v3-mixtral/` - LoRA adapter
- ✅ `training/models/placement-apprentice-v2/` - GPT-2 model

## Disk Space Check

Before first run:
```bash
# Check available space
df -h /home/quantumc1

# You need ~100GB free for Mixtral base model
```

## Performance Tips

1. **Always use --4bit on GPU** - Saves memory with minimal quality loss
2. **Use GPT-2 for simple questions** - Much faster for basic Q&A
3. **Use Mixtral for complex reasoning** - Better for troubleshooting
4. **Batch your questions** - Stay in interactive mode vs. relaunching

## Next Steps

1. **Test with simple question:**
   ```bash
   python3 chat_with_specialist.py --4bit --question "What is placement?"
   ```

2. **Try interactive chat:**
   ```bash
   ./launch_mixtral_chat.sh
   ```

3. **Compare with GPT-2:**
   ```bash
   # Mixtral answer
   python3 chat_with_specialist.py --question "Explain timing-driven placement"

   # GPT-2 answer
   python3 chat_with_apprentice.py --question "Explain timing-driven placement"
   ```

## Comparison: Mixtral vs GPT-2

Run both and see the difference:

```bash
# Question: "Why is wire delay so high on this net?"

# Mixtral: Will analyze multiple factors (length, fanout, RC delay,
#          routing congestion, technology node effects)

# GPT-2: Will give a brief answer about wire length and resistance
```

---

**Ready to try the more powerful Mixtral model?**

```bash
cd /home/quantumc1/cda-agent-2C1 && ./launch_mixtral_chat.sh
```

*Note: First run downloads ~94GB. Be patient! Subsequent runs are fast.* ⚡
