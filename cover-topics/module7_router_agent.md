# Module 7 Capstone: The Intelligent LLM Router

*Automatically routing user prompts to the most efficient model based on complexity.*

---

## The Concept: Dynamic Model Selection

In standard AI applications, the developer hardcodes the model (e.g., always use GPT-4o or always use Claude 3.5 Sonnet). 
If the user asks "What is 2+2?", using the largest model is a waste of money and time. If the user asks for complex debugging, a small model will fail.

**Dynamic Routing** solves this by inserting a tiny, ultra-fast agent *before* the main generation step.

```
USER PROMPT
     │
     ▼
┌──────────────┐
│ Router Agent │  ← Fast, cheap model analyzing intent
└──────┬───────┘
       │
       ├─────────────── (Decision: Simple query)
       │                       ▼
       │               ┌───────────────┐
       │               │ Small Model   │ ← Fast, 10x cheaper
       │               └───────────────┘
       │
       └─────────────── (Decision: Complex query)
                               ▼
                       ┌───────────────┐
                       │ Large Model   │ ← High reasoning capability
                       └───────────────┘
```

---

## The Router Prompt

The core of the system is the prompt given to the Router Agent. It must be highly structured so the router outputs a consistent decision. We force the output into JSON using function calling mechanisms.

**Example Router Instructions:**
```
Analyze the following user prompt and classify its complexity.

Category 1 (SIMPLE):
- Greetings, small talk, casual conversation
- Basic factual questions (e.g., capitals, dates)
- Simple translation or summarization of short text

Category 2 (COMPLEX):
- Code generation, debugging, or architecture design
- Complex reasoning, logic puzzles, or math
- Analysis of long documents or abstract concepts

Respond in JSON format:
{
  "complexity": "simple" | "complex",
  "reason": "short explanation of why"
}
```

---

## Calculating the Savings

To understand why routing is powerful, we measure two things:
1. **Latency:** How long it takes to get the first token back. Small models are often 3x-5x faster.
2. **Cost:** API providers charge by token. Large models are often 10x-50x more expensive.

If a router correctly sends 60% of traffic to a small model, you can cut your total API bill by over 50% while improving average response speed for users, with nearly zero drop in answer quality.

---

## The Implementation Architecture

In our capstone project, we will use the **Groq API** (which provides access to different open-weight models for free) to simulate this environment.

* **The Router Model:** `llama-3.1-8b-instant` (Very fast, smart enough to classify)
* **The "Small" Model:** `llama-3.1-8b-instant` (Fast, good for simple tasks)
* **The "Large" Model:** `llama-3.3-70b-versatile` (Highly capable, slower, better for complex logic)

We will build an orchestrator that:
1. Passes the user prompt to the Router.
2. Reads the JSON decision.
3. Automatically switches the model endpoint.
4. Generates the final answer.
5. Prints a simulated "Cost & Time Saved" report.
