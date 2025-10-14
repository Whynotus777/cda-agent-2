

# Triage Architecture - Streaming Multi-Layer System

## 🎯 Design Philosophy

**Problem with character count routing:**
- Arbitrary thresholds (800, 2000)
- Doesn't understand conversation depth
- Treats each query independently
- Slow start for complex conversations

**New approach:**
- **Always start fast** (3B, 1-2 seconds)
- Let conversation naturally deepen
- Models escalate intelligently based on need
- User gets immediate feedback

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    User Query                        │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 1: 3B Triage (ALWAYS, ~1-2 seconds)          │
│  ┌──────────────────────────────────────────────┐   │
│  │ 1. Immediate response to user                │   │
│  │ 2. Ask clarifying questions if needed        │   │
│  │ 3. Analyze: SIMPLE/MODERATE/COMPLEX          │   │
│  │ 4. Decide: Can I handle this? Or escalate?   │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
    Handles it            Needs escalation
          │                     │
          ▼                     ▼
    ┌─────────┐        ┌──────────────────┐
    │  Done   │        │ Background call  │
    └─────────┘        │  to 8B or 70B    │
                       └────────┬──────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ LAYER 2: 8B      │
                       │ Moderate         │
                       │ (3-5 seconds)    │
                       └────────┬──────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ LAYER 3: 70B     │
                       │ Complex          │
                       │ (15-20 seconds)  │
                       └──────────────────┘
```

## 📊 Conversation Flow Example

### Turn 1: Simple Query
```
User: "What is synthesis?"

3B (1 sec): "Synthesis is the process of converting RTL code
            into a gate-level netlist. Would you like to know
            about specific synthesis tools or techniques?"

[Analysis: SIMPLE, no escalation needed]
```

### Turn 2: Getting More Specific
```
User: "How do I optimize for area using Yosys?"

3B (1 sec): "Let me help you with Yosys area optimization.
            I'm pulling together detailed guidance..."

[Analysis: MODERATE, escalating to 8B...]

8B (3 sec): "To optimize for area in Yosys:
            1. Use synthesis flags: -abc -opt -opt_clean
            2. Set optimization goals: synth -top MODULE -abc9 -area
            3. ... [detailed technical response]"
```

### Turn 3-7: Building Context
```
[Each turn: 3B responds immediately, evaluates complexity]
[Conversation depth: 7 turns]
[Pattern: Increasingly technical questions]
```

### Turn 8: Proactive Escalation
```
User: "Now I need to choose between TSMC 7nm and Samsung 5nm
       for my robotics SoC with custom NPU..."

3B (1 sec): "This is a complex architecture decision.
            Let me bring in deeper analysis..."

[Analysis: COMPLEX + conversation depth=8]
[Proactive escalation to 70B]

70B (18 sec): [Comprehensive analysis of:
              - Process node trade-offs
              - Power/performance implications
              - Cost considerations
              - NPU integration challenges
              - Specific recommendations]
```

## 🧠 Complexity Assessment

### 3B Triage Analyzes:

1. **Query Characteristics:**
   - Keyword analysis
   - Question complexity
   - Technical depth
   - Ambiguity level

2. **Conversation Context:**
   - Turn number (depth)
   - Topic evolution
   - Previous complexity levels
   - Escalation history

3. **User Intent:**
   - Quick fact lookup → SIMPLE
   - How-to/implementation → MODERATE
   - Architecture/trade-offs → COMPLEX

### Complexity Levels:

#### SIMPLE (3B handles)
- Definitions
- Basic explanations
- Clarifying questions
- Quick facts
- Simple commands

**Example:**
```
"What is DRC?"
"Define setup time"
"Create a new project"
```

#### MODERATE (Escalate to 8B)
- Technical how-to
- Implementation guidance
- Tool usage
- Optimization techniques
- Phase-specific questions

**Example:**
```
"How do I reduce power in my design?"
"What placement density should I use?"
"Optimize this for timing"
```

#### COMPLEX (Escalate to 70B)
- Architecture decisions
- Trade-off analysis
- Multi-factor optimization
- Novel problems
- Strategic planning

**Example:**
```
"Should I use ARM or RISC-V for my robotics SoC?"
"Evaluate TSMC 7nm vs Samsung 5nm for inference workloads"
"Design a power-efficient NPU architecture"
```

## 🔄 Proactive Escalation

The system automatically escalates when:

### 1. Conversation Depth
```python
if conversation_depth > 10:
    # After 10 turns, assume user needs deeper engagement
    proactive_escalate_to_8B()
```

### 2. Sustained Complexity
```python
if last_3_queries_are_moderate():
    # Pattern of technical queries
    proactive_escalate_to_8B()
```

### 3. Increasing Trend
```python
if complexity_trend == [SIMPLE, MODERATE, MODERATE, COMPLEX]:
    # User diving deeper
    proactive_escalate_to_70B()
```

### 4. Topic Shift to Complex Domain
```python
if keywords in ['architecture', 'strategy', 'evaluate', 'choose']:
    # Inherently complex topics
    escalate_to_70B()
```

## ⚡ Performance Comparison

### Old System (Character Count):
```
Query 1 (30 chars): "What is synthesis?"
→ 3B (1 sec) ✓

Query 2 (150 chars): "How do I optimize my design for minimum area..."
→ 3B (1 sec) ✓ [But needs 8B!]

Query 3 (850 chars): [Long query]
→ 8B (4 sec) [Good]

Query 4 (2100 chars): [Very long query]
→ 70B (18 sec) [Good]
```

**Problems:**
- Character count doesn't measure complexity
- Short complex queries mis-routed
- Long simple queries over-routed
- No conversation awareness

### New System (Triage):
```
Query 1: "What is synthesis?"
3B triage (1 sec): [Answers directly, complexity=SIMPLE]
→ Done ✓

Query 2: "How do I optimize for area?"
3B triage (1 sec): "Let me get detailed guidance..."
→ Escalates to 8B (3 sec) → Detailed response ✓

Query 3: [Building on Query 2]
3B triage (1 sec): [Immediate thoughts]
→ Continues with 8B (conversation context maintained) ✓

Query 4: "Should I use ARM or RISC-V?" [After 8 turns]
3B triage (1 sec): "Complex decision, analyzing..."
→ Escalates to 70B (depth=9, complexity=COMPLEX) ✓
```

**Benefits:**
- Always fast initial response (1-2 sec)
- Intelligent complexity assessment
- Conversation-aware routing
- Natural deepening over time

## 📈 Response Time Analysis

### Distribution:

```
3B Only (40% of queries):     1-2 seconds
3B → 8B (45% of queries):     4-6 seconds total
3B → 70B (15% of queries):    18-22 seconds total
```

**User Experience:**
- **Always gets response within 2 seconds**
- Refined answer streams in for complex queries
- No waiting for simple queries
- Natural conversation flow

## 🎛️ Configuration

```yaml
llm:
  triage:
    enable: true

    models:
      triage: "llama3.2:3b"     # Always first
      moderate: "llama3:8b"      # Layer 2
      complex: "llama3:70b"      # Layer 3

    escalation:
      conversation_depth: 10         # Escalate after N turns
      consecutive_moderate: 3        # Escalate if N moderate in a row
      enable_auto_escalation: true   # Smart escalation

    shadow_orchestrator:
      enable: true                   # 70B learns from all
      model: "llama3:70b"
      async: true                    # Non-blocking
```

## 🔍 Monitoring

### Check Routing Stats:
```python
stats = triage_router.get_routing_stats()
print(f"Total turns: {stats['total_turns']}")
print(f"Escalations: {stats['escalations']}")
print(f"Escalation rate: {stats['escalation_rate']:.1%}")
print(f"Avg complexity: {stats['avg_complexity']:.2f}")
```

### Example Output:
```
Total turns: 12
Escalations: 5
Escalation rate: 41.7%
Avg complexity: 1.83  # (1=SIMPLE, 2=MODERATE, 3=COMPLEX)
Recent history:
  Turn 8: MODERATE (escalated to 8B)
  Turn 9: MODERATE (escalated to 8B)
  Turn 10: COMPLEX (escalated to 70B)
  Turn 11: MODERATE (8B)
  Turn 12: SIMPLE (3B only)
```

## 🎯 Best Practices

### 1. Let 3B Be the Face
The 3B model should:
- ✅ Always respond first
- ✅ Be friendly and helpful
- ✅ Ask clarifying questions when unclear
- ✅ Set user expectations ("Let me get detailed info...")

### 2. Escalate Gracefully
When escalating:
- ✅ Tell user you're getting more info
- ✅ Stream initial response immediately
- ✅ Refine with larger model in background

### 3. Track Conversation Depth
- Monitor turn count
- Identify when conversation deepens
- Proactively escalate before user frustrated

### 4. Learn from Patterns
The system learns:
- Which queries need escalation
- Conversation depth patterns
- User expertise level
- Topic complexity evolution

## 🚀 User Experience

### What User Sees:

**Simple Query:**
```
You: What is synthesis?
Agent (1 sec): Synthesis converts RTL to gates. [Done]
```

**Moderate Query:**
```
You: How do I optimize for area?
Agent (1 sec): Let me get detailed guidance...
Agent (3 sec): Here's how to optimize for area in Yosys: [detailed]
```

**Complex Query (Deep Conversation):**
```
[After 8 turns of technical discussion]

You: Should I use TSMC 7nm or Samsung 5nm for my robotics SoC?
Agent (1 sec): This is a complex architecture decision. Analyzing...
Agent (18 sec): Here's a comprehensive evaluation: [detailed analysis]
```

## 📚 Implementation

The triage router is automatically used when:

```python
# In conversation_manager.py

if triage_enabled:
    result = triage_router.route_streaming(user_message, context)

    # User sees immediate response
    print(result['immediate_response'])

    # If escalated, refined response follows
    if result['refined_response']:
        print(result['refined_response'])
```

## Summary

**Old System:**
- Character count thresholds (arbitrary)
- Each query independent
- No conversation awareness

**New System:**
- Always fast (3B first, 1-2 sec)
- Intelligent escalation
- Conversation-aware
- Natural deepening
- Better user experience

The triage system gives users **immediate feedback** while intelligently routing complex queries to appropriate models in the background. 🎯
