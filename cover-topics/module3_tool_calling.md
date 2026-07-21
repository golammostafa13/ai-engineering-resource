# Module 3: Tool Calling & Function Calling

*The bridge between a text generator and a real AI Agent.*

---

## What Problem Does Tool Calling Solve?

An LLM by itself has two hard limitations:

```
1. KNOWLEDGE CUTOFF
   LLM trained in 2024 knows nothing about events in 2025.
   → We need a "search" tool.

2. NO EXECUTION ABILITY
   LLM can write Python code but cannot run it.
   → We need a "code execution" tool.
```

Tool Calling solves both. The LLM learns to output structured "requests" for tools, and your application executes them.

---

## The Core Concept: LLM Outputs a Request, Not an Answer

**Without Tool Calling:**
```
User  →  "What is 25 * 47?"
LLM   →  "25 * 47 = 1175"   ← might hallucinate the arithmetic!
```

**With Tool Calling:**
```
User     →  "What is 25 * 47?"
LLM      →  { "tool": "calculator", "args": {"a": 25, "b": 47, "op": "multiply"} }
Your App →  runs calculator(25, 47, "multiply") → returns 1175
LLM      →  "25 × 47 = 1175"   ← now factually correct
```

The LLM is not doing the math — it is **delegating** to a real function.

---

## The 5-Step Tool Calling Loop

```
Step 1: DEFINE
  You define tools using a JSON Schema.
  Each schema describes: name, purpose, and parameters.

Step 2: INJECT
  You inject the tool schemas into the system prompt.
  The LLM reads them and knows what it can do.

Step 3: DECIDE
  The LLM reads the user request and decides:
    a) I can answer directly → outputs text normally.
    b) I need a tool → outputs a structured tool call JSON.

Step 4: EXECUTE
  Your application intercepts the tool call.
  It runs the actual Python function.
  It captures the result.

Step 5: FEED BACK
  The result is formatted as a "tool response message"
  and appended to the conversation.
  The LLM reads it and generates the final answer.
```

**Visual Loop:**

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                           │
│                                                         │
│  User Input                                             │
│      ↓                                                  │
│  [System Prompt + Tool Schemas + Chat History]          │
│      ↓                                                  │
│  LLM decides ─────→ "I know the answer"                │
│      │                      ↓                           │
│      │               Final Answer → User                │
│      │                                                   │
│      └──────→ "I need a tool"                           │
│                      ↓                                   │
│            Tool Call JSON output                        │
│      { "tool": "get_weather",                           │
│        "args": {"city": "Dhaka"} }                      │
│                      ↓                                   │
│         Your App runs the function                      │
│            get_weather("Dhaka")                         │
│                      ↓                                   │
│         Result: "32°C, Humid"                           │
│                      ↓                                   │
│         Append result to conversation                   │
│                      ↓                                   │
│         LLM reads result → Final Answer                 │
│                      ↓                                   │
│              Answer → User                              │
└─────────────────────────────────────────────────────────┘
```

---

## Tool Schema: How to Describe a Tool

Every tool is described using a JSON Schema that the LLM reads.

**Example — Weather Tool:**

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given city. Use this when the user asks about weather conditions.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "The name of the city, e.g. Dhaka, London"
      }
    },
    "required": ["city"]
  }
}
```

**The `description` field is the most important part.** The LLM uses it to decide whether to call this tool. A vague description → wrong tool selection. A clear, precise description → reliable tool use.

---

## Message Types in a Tool-Calling Conversation

A tool-calling conversation has 4 message types:

```
1. SYSTEM message
   Role: "system"
   Content: Agent persona + tool descriptions + rules

2. USER message
   Role: "user"
   Content: What the user typed

3. ASSISTANT message with TOOL CALL
   Role: "assistant"
   Content: null
   Tool_calls: [{ "name": "get_weather", "args": {"city": "Dhaka"} }]

4. TOOL RESULT message
   Role: "tool"
   Tool_call_id: (matches the assistant's call)
   Content: "32°C, Humid"
```

The conversation history must preserve all 4 types in order, so the LLM always has full context.

---

## Multi-Step Tool Calling (The Real Power)

Complex tasks require multiple tool calls in sequence:

```
User: "Find me flights from Dhaka to London next week,
       check the weather there, and give me a packing list."

Turn 1:
  LLM → calls search_flights("Dhaka", "London", "next_week")
  Result: "Found 3 flights: Biman at $850, Emirates at $1100..."

Turn 2:
  LLM → calls get_weather("London")
  Result: "15°C, Rainy"

Turn 3:
  LLM → (has enough info) generates final answer:
  "I found 3 flights... London will be 15°C and rainy next week,
   so I recommend packing a raincoat and warm layers."
```

The LLM automatically chained 2 tool calls before generating the answer!

---

## Error Handling in Tool Calls

Real tools fail. Your agent must handle this gracefully:

```
Tool call: get_weather("UnknownCity123")

Tool result: { "error": "City not found" }

LLM reads error → responds:
  "I couldn't find weather data for that location.
   Could you check the city name?"
```

The key: **Always return an error as a tool result, never crash.** The LLM can recover from errors if it sees them as structured feedback.

---

## The 3 Types of Tools Every Agent Needs

```
1. KNOWLEDGE TOOLS
   → Search the web, query a database, read documents
   → Solves: LLM knowledge cutoff

2. EXECUTION TOOLS
   → Run code, call APIs, send emails, create files
   → Solves: LLM can describe but not do

3. MEMORY TOOLS
   → Store/retrieve notes, remember user preferences
   → Solves: LLM forgets between sessions
```

---

## Key Principles to Remember

1. **Descriptions drive decisions.** Write tool descriptions as if explaining to a smart person who has never seen your codebase.

2. **Temperature = 0 for tool calls.** Always use `temperature=0` when you need the LLM to call tools — it must output exact JSON, not creative variations.

3. **Validate before executing.** Always validate tool arguments before running them. The LLM can pass wrong types or missing fields.

4. **One tool per step (usually).** Most frameworks process one tool call per loop iteration. Some advanced setups allow parallel tool calls.

5. **The loop has a max step limit.** Always set a maximum number of iterations (e.g., 10 steps) to prevent infinite loops if the LLM keeps calling tools without converging.
