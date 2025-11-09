# Chip Design Placement Apprentice - Chat Interface

## What is this?

This is a **GPT-2 Medium** language model (355M parameters) that has been fine-tuned on chip design placement knowledge. It can answer questions about:

- Cell placement algorithms and techniques
- EDA tools (DREAMPlace, OpenROAD, OpenLane)
- Placement optimization strategies
- Timing, power, and area tradeoffs
- Real-world chip design challenges

## Model Details

- **Base Model:** GPT-2 Medium (from OpenAI, not Meta's Llama)
- **Parameters:** 354.8M
- **Size:** 1.4GB
- **Training Data:** 198 high-quality examples from the Gold Standard corpus
  - Sources: Academic papers (DAC, ICCAD, ISPD conferences)
  - Fabricated chips (OpenTitan, Rocket Chip, BlackParrot, etc.)
  - EDA tool documentation (DREAMPlace, OpenROAD, OpenLane, Yosys, Magic)
  - Contest benchmarks and real-world tapeouts
- **Hardware:** Runs on both CPU and GPU (CUDA detected: ✓)

## Quick Start

### Option 1: Interactive Chat (Recommended)

```bash
cd /home/quantumc1/cda-agent-2C1
./launch_apprentice_chat.sh
```

Then type your questions and press Enter!

### Option 2: Single Question Mode

```bash
python3 chat_with_apprentice.py --question "How does DREAMPlace work?"
```

### Option 3: Use a Different Model

```bash
# Try the wisdom specialist (if available)
python3 chat_with_apprentice.py --model ./training/models/wisdom-specialist-v2
```

## Example Questions to Ask

1. **Basics:**
   - "What is cell placement in chip design?"
   - "Explain global vs detailed placement"
   - "What is legalization?"

2. **Tools:**
   - "How does DREAMPlace achieve GPU acceleration?"
   - "What placement quality can OpenROAD achieve?"
   - "Compare DREAMPlace and RePlAce"

3. **Optimization:**
   - "How do I reduce wirelength in placement?"
   - "What is timing-driven placement?"
   - "Why does placement become harder above 85% utilization?"

4. **Advanced:**
   - "What are the key innovations in modern placement algorithms?"
   - "How do I handle macro placement in SoC designs?"
   - "What is routability-driven placement?"

5. **Real-World:**
   - "How was OpenTitan placed?"
   - "What placement techniques did Google use for TPU v1?"
   - "What are the differences between 7nm and 28nm placement?"

## Chat Commands

- **Type your question** - Get an answer from the apprentice
- **help** - Show example questions
- **quit** / **exit** / **q** - Exit the chat
- **Ctrl+C** - Force exit

## Available Models

You have several trained models available:

```
training/models/
├── placement-apprentice-v1/         # Earlier version
├── placement-apprentice-v2/         # Latest (default)
├── placement-specialist-5090-v1/    # Optimized for RTX 5090
├── wisdom-specialist-v1/            # General chip design wisdom
├── wisdom-specialist-v2/            # Improved version
└── reasoning-specialist-rag-v1/     # With RAG support
```

## Configuration Options

```bash
# Run on CPU (slower but works without GPU)
python3 chat_with_apprentice.py --device cpu

# Run on GPU (faster)
python3 chat_with_apprentice.py --device cuda

# Use a different model
python3 chat_with_apprentice.py --model ./training/models/wisdom-specialist-v2

# Single question (non-interactive)
python3 chat_with_apprentice.py --question "Your question here"
```

## Performance

- **GPU (CUDA):** ~1-3 seconds per response
- **CPU:** ~10-30 seconds per response
- **Max tokens:** 200 tokens per response (configurable in code)

## Understanding the Responses

The apprentice model was trained on real chip design knowledge, so it will:
- ✓ Reference real tools (DREAMPlace, OpenROAD, Yosys, etc.)
- ✓ Cite real chips (OpenTitan, TPU, Rocket Chip, etc.)
- ✓ Use accurate terminology and metrics
- ✓ Provide practical guidance based on academic papers and production tapeouts

However, remember:
- It's a 355M parameter model (relatively small)
- Best for technical explanations and guidance
- For critical designs, always verify with official documentation
- May occasionally generate plausible-sounding but incorrect details

## Training Data Sources

The model was trained on ~200 curated examples including:

1. **Fabricated Chips (52%):**
   - OpenTitan (Google/lowRISC) - GF 22nm
   - Rocket Chip / BOOM - 16nm TSMC
   - BlackParrot multicore - GF 12nm
   - Google TPU v1 - 28nm
   - Pulpissimo cluster - 65nm
   - And more...

2. **Academic Papers (30%):**
   - DAC, ICCAD, ISPD conferences
   - ISPD placement contests (2005, 2015, etc.)
   - DREAMPlace, RePlAce, NTUPlacerDR papers
   - Timing and routability optimization papers

3. **EDA Tool Documentation (18%):**
   - DREAMPlace GPU acceleration guides
   - OpenROAD flow documentation
   - OpenLane 130nm tapeout guides
   - Yosys synthesis documentation

## Troubleshooting

### Model not found error
```bash
# Make sure you're in the right directory
cd /home/quantumc1/cda-agent-2C1

# Verify model exists
ls -lh training/models/placement-apprentice-v2/
```

### Out of memory (GPU)
```bash
# Force CPU usage
python3 chat_with_apprentice.py --device cpu
```

### Slow on CPU
This is normal. The model has 355M parameters, so CPU inference takes time. Consider:
- Using GPU if available
- Using a smaller model
- Limiting max_new_tokens in the code

### Import errors
```bash
# Install dependencies
pip install torch transformers
```

## Model Comparison

| Model | Base | Parameters | Training Data | Best For |
|-------|------|------------|---------------|----------|
| placement-apprentice-v2 | GPT-2 Medium | 355M | 198 placement examples | Placement questions |
| wisdom-specialist-v2 | GPT-2 Medium | 355M | 17 wisdom examples | General chip design |
| placement-specialist-5090-v1 | GPT-2 Medium | 355M | Diagnostic corpus | Troubleshooting |

## Next Steps

- **Try the chat:** Run `./launch_apprentice_chat.sh`
- **Explore different models:** Change `--model` to try other specialists
- **Customize generation:** Edit `chat_with_apprentice.py` to adjust:
  - `max_new_tokens` - Length of responses
  - `temperature` - Creativity (0.0 = deterministic, 1.0 = creative)
  - `top_p`, `top_k` - Sampling parameters

## Credits

- **Training Script:** `train_placement_apprentice.py`
- **Base Model:** GPT-2 Medium (OpenAI)
- **Training Framework:** HuggingFace Transformers
- **Training Data:** Curated from academic papers, open-source chips, and EDA documentation

## License

Check the original model and training data licenses before commercial use.
