# Module 6: Multi-Agent Systems

*Multiple specialized agents working together under an orchestrator.*

---

## Why Multi-Agent?

One agent doing everything hits limits:

```
Single Agent Problem:
  Task: "Research our refund policy, calculate the total for 3 orders
         of $89.50 each, and write a customer email."

  One agent tries to: research + calculate + write
  → Gets confused juggling all three
  → Longer prompts → more hallucination risk
  → Hard to debug (which part failed?)
```

Multi-Agent Solution:
```
Orchestrator reads task → breaks into 3 subtasks

  Subtask 1 → Research Agent   → looks up refund policy
  Subtask 2 → Math Agent       → calculates 3 × $89.50 = $268.50
  Subtask 3 → Writer Agent     → composes the email

  Orchestrator combines results → final output
```

Each agent is an expert at one thing. The orchestrator is the manager.

---

## The Orchestrator + Worker Pattern

```
User Request
     ↓
┌─────────────────────────────────┐
│       ORCHESTRATOR              │
│  - Reads the full request       │
│  - Decides which agents needed  │
│  - Sends subtasks to workers    │
│  - Collects results             │
│  - Combines into final answer   │
└────┬──────────┬──────────┬──────┘
     ↓          ↓          ↓
┌─────────┐ ┌────────┐ ┌─────────┐
│Research │ │  Math  │ │ Writer  │
│ Agent   │ │ Agent  │ │  Agent  │
│         │ │        │ │         │
│Tools:   │ │Tools:  │ │Tools:   │
│-RAG     │ │-calc   │ │(none)   │
│-web srch│ │        │ │         │
└─────────┘ └────────┘ └─────────┘
```

---

## How Agents Communicate

Agents pass messages as plain text. The orchestrator:

1. Sends a **task description** to a worker agent
2. Worker agent processes it (may use tools)
3. Worker returns a **result string**
4. Orchestrator collects all results and synthesizes

```python
# Orchestrator sends this to the Research Agent:
task = "Find the company refund policy. Return only the key facts."

# Research Agent returns:
result = "Refunds within 30 days. Processed in 5-7 days. Contact support@company.com."

# Orchestrator uses this result to form the final answer.
```

No complex frameworks needed — just function calls and strings.

---

## The 3 Specialist Agents We'll Build

```
1. RESEARCH AGENT
   System: "You are a research specialist. Use search_knowledge_base
            and search_web to find accurate information."
   Tools: search_knowledge_base, search_web

2. MATH AGENT
   System: "You are a math specialist. Use the calculator for all
            arithmetic. Return only the numerical result and formula."
   Tools: calculate

3. WRITER AGENT
   System: "You are a professional writer. Given information and
            context, write clear and concise responses."
   Tools: (none — writes from given context)
```

---

## Key Design Decisions

**1. Each agent has its own system prompt**
The system prompt defines the agent's personality, focus, and rules.
A math agent's prompt tells it to ONLY do math — nothing else.

**2. Each agent has its own tool set**
The research agent cannot accidentally call the calculator.
The writer agent has no tools — it just writes.

**3. The orchestrator decides routing**
The orchestrator's system prompt contains routing logic:
- "If the task requires information lookup → use Research Agent"
- "If the task requires math → use Math Agent"
- "For final writing → use Writer Agent"

**4. Results are passed as context, not memory**
Worker agent results are injected into the orchestrator's next prompt.
Workers don't share memory — they only receive their specific task.
