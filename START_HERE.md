# CDA Agent - Claude's Working Copy

This is a **separate copy** of the CDA agent that Claude Code is working on.

Your Cursor agent can continue working on `/home/quantumc1/cda-agent` without conflicts.

## Quick Start

### 1. Run the Agent (Easiest)

```bash
cd /home/quantumc1/cda-agent-claude
./run_claude.sh
```

### 2. Or Run Manually

```bash
cd /home/quantumc1/cda-agent-claude
source venv/bin/activate
python3 agent.py
```

## What's Different in This Copy?

This copy has Claude's latest improvements:

✅ **Triage Router** - Fast 3B model responds first (1-2 sec), escalates to 8B/70B if needed
✅ **QUERY Action Type** - Handles informational questions properly
✅ **Better Query Detection** - Catches typos, question patterns, expanded keywords
✅ **Proper Query Responses** - Actually answers questions instead of asking for clarification
✅ **No More Clarification Loops** - Works with or without triage enabled

### Key Features

**With Triage Enabled (default):**
- You: "what is placement?"
- Agent (1-2 sec): [Actual explanation from 3B]
- Agent (if needed): [Refined answer from 8B/70B]

**Even Without Triage:**
- Heuristics catch informational queries
- Direct LLM-powered answers
- No spurious backend actions

## Configuration

Config file: `configs/default_config.yaml`

Triage is **enabled by default** in this copy:
```yaml
llm:
  triage:
    enable: true
```

To disable triage (use 8B model for everything):
```yaml
llm:
  triage:
    enable: false
```

## Testing

Try these queries:
```
overview of placement
tell me everything about chip design
what are the stages of SoC design
explain synthesis
summary of current design
```

Should get actual answers, not clarification loops!

## Directory Structure

```
/home/quantumc1/cda-agent/          ← Cursor agent's copy
/home/quantumc1/cda-agent-claude/   ← Claude Code's copy (this one)
```

Both are completely independent with separate virtual environments.

## Models Required

Make sure Ollama has these models:
```bash
ollama pull llama3.2:3b    # Triage layer (fast)
ollama pull llama3:8b      # Moderate queries
ollama pull llama3:70b     # Complex queries (optional)
```

## Logs

Logs are saved to: `./logs/cda_agent_YYYYMMDD_HHMMSS.log`

## Support

If something doesn't work:
1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Check models are available: `ollama list`
3. Check logs in `./logs/`
4. Make sure venv is activated: `source venv/bin/activate`

---

**This copy is for Claude Code to work on. Your Cursor agent won't be affected.**
