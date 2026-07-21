# Module 5: Building a Real Agent

*Combining Tool Calling + RAG + Memory into one complete working agent.*

---

## What a Real Agent Looks Like

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE AGENT SYSTEM                    │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  AGENT BRAIN (LLM)                    │  │
│  │   Reads context, decides what to do next              │  │
│  └───────────────────────────────────────────────────────┘  │
│              │                                              │
│    ┌─────────┼──────────────────────────┐                  │
│    ↓         ↓                          ↓                  │
│ ┌──────┐ ┌──────────────┐        ┌────────────┐           │
│ │Tools │ │  Knowledge   │        │Conversation│           │
│ │      │ │   Base (RAG) │        │  Memory    │           │
│ │ -calc│ │              │        │            │           │
│ │ -wthr│ │  ChromaDB    │        │  History   │           │
│ │ -srch│ │  Vector DB   │        │  Buffer    │           │
│ └──────┘ └──────────────┘        └────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## The Agent Decision Tree

Every time the agent processes a message, it follows this logic:

```
New user message arrives
        ↓
LLM reads: system prompt + all tools + conversation history + user message
        ↓
LLM decides:
    ├── "I can answer this directly"
    │       ↓
    │   Generate final text response → User
    │
    ├── "I need to look this up in the knowledge base"
    │       ↓
    │   Call search_knowledge_base("...")
    │       ↓
    │   Read retrieved docs → answer grounded in docs
    │
    ├── "I need a live tool (weather, calculator, etc.)"
    │       ↓
    │   Call appropriate tool
    │       ↓
    │   Read tool result → incorporate into answer
    │
    └── "I need multiple things"
            ↓
        Call tool 1 → get result
            ↓
        Call tool 2 → get result
            ↓
        Combine all results → final answer
```

---

## The 4 Tools Our Agent Has

```
1. search_knowledge_base(query)
   → Searches ChromaDB with semantic similarity
   → Returns relevant company document chunks
   → Used for: company policies, internal data, documents

2. get_weather(city)
   → Returns current weather conditions
   → Used for: any weather/temperature question

3. calculate(a, b, operation)
   → Performs arithmetic accurately
   → Used for: any math question

4. search_web(query)
   → Returns web search results
   → Used for: general knowledge questions outside documents
```

---

## Conversation Memory

A real agent remembers the full conversation history:

```
Turn 1:
  User:  "What is the refund policy?"
  Agent: "Refunds are processed in 5-7 business days."

Turn 2:
  User:  "How about shipping?"
  Agent: (LLM sees full history → knows user is asking about our company shipping)
         "Standard shipping takes 3-5 business days."

Turn 3:
  User:  "Can I get both faster?"
  Agent: (LLM sees history → understands "both" = refund + shipping)
         "Express shipping is available for faster delivery.
          Refund processing time is fixed at 5-7 days."
```

Without memory, the agent treats every message as a fresh conversation and loses context.

---

## The System Prompt (The Agent's Identity)

```
You are a helpful assistant for TechCorp customer service.

You have access to the following tools:
- search_knowledge_base: for company policies and documents
- get_weather: for current weather information
- calculate: for math calculations
- search_web: for general knowledge

Rules:
1. ALWAYS use search_knowledge_base before answering company policy questions.
2. NEVER make up company information — use the knowledge base.
3. Use tools when needed. Answer directly when you already know.
4. Be concise and helpful.
```

This system prompt is the foundation of the agent's behavior. Every rule here directly controls how the agent acts.

---

## Why This Architecture Works

```
Problem → Solution

"LLM makes up company data"
→ RAG tool forces it to retrieve before answering

"LLM can't do math reliably"
→ Calculator tool handles all arithmetic

"LLM's knowledge is outdated"
→ Web search tool provides current information

"Agent forgets earlier messages"
→ Conversation history passed in every request

"Agent uses wrong tool"
→ Good tool descriptions + system prompt rules guide selection
```

---

## The Complete Data Flow

```
User: "What is our refund policy and what is 150 * 3.5?"

Message History Sent to LLM:
[
  {"role": "system",    "content": "You are a helpful agent..."},
  {"role": "user",      "content": "What is our refund policy and what is 150 * 3.5?"}
]

LLM responds with TWO tool calls:
  call_1: search_knowledge_base("refund policy")
  call_2: calculate(150, 3.5, "multiply")

Your app executes both:
  result_1: "Refunds are processed in 5-7 business days..."
  result_2: "525.0"

History updated:
[
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "..."},
  {"role": "assistant", "tool_calls": [call_1, call_2]},
  {"role": "tool",      "content": "Refunds are processed..."},
  {"role": "tool",      "content": "525.0"}
]

LLM reads updated history → generates final answer:
  "According to our policy, refunds take 5-7 business days.
   And 150 × 3.5 = 525."
```
