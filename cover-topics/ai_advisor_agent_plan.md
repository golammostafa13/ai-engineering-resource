# AI Model Advisor Agent: Implementation Plan

## 🎯 Goal
Build an intelligent agent that acts as a consultant. When a user describes a task or problem they want to solve, the agent analyzes the intent, maps it to a strict taxonomy, searches the web for current recommendations, and presents the **top 5 AI models** for that specific job (based on complexity, capability, and security/privacy) — then lets the **user choose** which one they prefer.

**Fully dynamic — no hardcoded model list.** Candidates are discovered from live web search at run time, so the agent stays current as new models ship. A lightweight *grounding check* (each candidate name must actually appear in the search results) is the only guard against the local model inventing names — it is not a fixed allowlist.

**Scope:** free and local only. Single local user running the script directly — no auth, no session/user tracking, no concurrency handling. No paid APIs or API keys required for v1.

**Status:** implemented in [`ai_advisor_agent/advisor.py`](../ai_advisor_agent/advisor.py) (separate folder from the rest of the repo). Design decisions and their rationale are in the "Key Findings" section.

### Companion tool: the Model Router (`router.py`)
Where the **advisor** *recommends* models for a human to pick, the **router** *automatically switches* to the right model at prompt time — for use inside a live coding project / assistant loop, so every prompt runs on the cheapest model that can handle it.

It is a **top layer over many providers**: it switches models across BOTH local Ollama *and* cloud providers (Anthropic / Claude, OpenAI, Groq, ...). The provider layer ([`providers.py`](../ai_advisor_agent/providers.py)) hides where each model actually lives.

*   **Auto-discovers** the routable pool across all providers (`providers.discover_all()`):
    *   Local Ollama is always **live and free** (`/api/tags`).
    *   A cloud provider is discovered **live from its API when its key is set** (e.g. `ANTHROPIC_API_KEY`); otherwise its models come from a small offline **registry** ([`model_registry.json`](../ai_advisor_agent/model_registry.json)) so the router can still route *across* providers with no keys. This is the "auto: live if key, else registry" design.
*   **Tags** each model by tier (Small/Medium/Large — from the param count in the name, on-disk size, or name hints like `opus`/`haiku`) and capability (Coding / Reasoning / Vision / Writing / General). The registry can override tags + carry a **cost** (`Free (local)`, `$ cheap`, `$$$ premium`) for cloud models.
*   **Classifies** the incoming prompt with the same taxonomy — always run on the **smallest *local* general model** ("router brain"), so the routing decision itself is cheap, fast, and never spends a cloud call.
*   **Routes**: a **privacy guard first** — a task classified `security: Private` is restricted to **local** models so sensitive prompts never leave the machine; then capability match (a Reasoning `model_type` wins, then exact capability, then General), then tier by complexity — **Small task → smallest/cheapest model, Large task → largest available** (which may be a cloud flagship). This is the "identify a simple task and send it to a small model, escalate a hard one — but keep private work local" behavior, spanning local *and* cloud.
*   **Runs** the prompt on the chosen model (decide + run). A cloud pick is actually executed **only when its key is present**; otherwise the switch decision is reported and the call is skipped gracefully (staying free/local by default).

Pure routing logic is covered by fast unit tests in [`test_router.py`](../ai_advisor_agent/test_router.py) (no Ollama or network needed) — now including cross-provider cases. This realizes the "Commercial Provider Upgrade Path" section below: adding another provider is one entry in `CLOUD_PROVIDERS` plus (optionally) rows in the registry.

#### Using a Claude Pro/Max **subscription** (no API key)
A subscription is **not** the same as API access — it gives you no API key, and the Anthropic API is billed separately per token. But the subscription *does* authenticate the **`claude` CLI**, which can answer a single prompt non-interactively (`claude -p "<prompt>" --model <haiku|sonnet|opus>`). So the router has a dedicated **`claude-cli` provider** ([`providers.py`](../ai_advisor_agent/providers.py)) that reaches Claude through the CLI via subprocess — **no API key, no per-token billing**, just your existing subscription.

*   It appears in the pool automatically whenever the `claude` binary is installed and logged in (else it's routable-on-paper from the registry).
*   The subscription tiers are registered as `haiku` (Small), `sonnet` (Medium), `opus` (Reasoning/Large) — so a prompt auto-switches: trivial → haiku, hard/reasoning → opus.
*   Run **Claude-only** (switch among just the Claude tiers) with `--claude`, or keep the full cross-provider pool (local Ollama stays cheapest for trivial work) by omitting it:

    ```bash
    # auto-switch among your Claude subscription tiers only
    ../.venv/bin/python router.py --claude "Fix this race condition in my worker pool"
    # or route across everything (local + subscription + any keyed cloud)
    ../.venv/bin/python router.py "Fix this race condition in my worker pool"
    ```
*   The classifier ("router brain") still runs on a **free local** model when one is available, so deciding *which* Claude tier to use doesn't itself spend subscription usage. Private-flagged prompts stay local and never reach the CLI.

#### Exposing the router as an **MCP tool**
The router is also wrapped as an MCP server ([`mcp_server.py`](../ai_advisor_agent/mcp_server.py)) so any MCP-capable agent (Claude Code / Desktop) can call it — see [`MCP.md`](../ai_advisor_agent/MCP.md) for setup. **Important limitation:** an MCP tool *cannot* reach into the host agent and flip its model selector; the protocol doesn't allow changing the host's own model. So it offers two realistic shapes instead:
*   `recommend_model(prompt)` — **advisory**: returns the best model + taxonomy + reasoning, runs nothing.
*   `run_with_best_model(prompt)` — **delegation**: routes *and runs* the prompt on the chosen model, returning the answer. This is real "switching" by offloading a sub-task (a big session pushes a trivial reformat down to a free local model, or a hard one up to opus).

Execution policy: free local Ollama always runs; a paid/subscription pick returns the *decision* but only executes when the caller passes `allow_paid=true` — no surprise spend. Both tools take `only_provider` (e.g. `"claude-cli"`) and `brain`. The decision path is a print-free `router.decide()` so stdout stays clean for the MCP stdio protocol.

**What makes routing capable (beyond the raw classifier):**
*   **Taxonomy refinement (`_refine_taxonomy`).** A tiny local classifier underrates hard tasks. The router post-processes its output with unambiguous prompt signals: **strong** signals (`race condition`, `deadlock`, `distributed`, `architect`, `prove`, `theorem`, `tradeoffs`, …) raise complexity to Large from *both Small and Medium*; **reasoning** signals additionally set `model_type=Reasoning` so proofs/planning reach a reasoning-grade model (opus); **soft** signals (`refactor`, `optimize`, `migrate`, …) raise only Small→Large; and safe capability nudges correct clear mistakes (a code fence → Coding, image words → Vision). It's task understanding, not a model list.
*   **Priority-aware tiering (`_target_tier`).** The `priority` axis now shifts the pick: `Quality` → one tier up, `Cost`/`Speed` → one tier down, `Balanced` follows complexity. So "quality, no matter the cost" reaches a bigger model and "cheapest that works" a smaller one.
*   **Tier escalation.** A Large task never gets stuck on a tiny specialist — if the capability-matched models are all smaller than the target tier, larger general models are pulled in (see step 2b in `route()`).
*   **Weak-classifier override.** When the local brain misclassifies, use `--brain <model>` (e.g. `--brain llama3.2`, or a Claude tier for top-quality routing at a small subscription cost).
*   **`claude -p` is agentic.** The CLI answers as a Claude Code *agent* with repo file access, not a plain completion — so an abstract prompt like *"fix this race condition in my worker pool"* makes it read the current directory and, finding no such code, ask you to point at the file. Run the router **from inside the project that holds the code** (and name the file), or paste the code into the prompt, to get an actual fix.

---

## 🏗️ System Architecture

### 1. The Taxonomy Classifier (System Prompt)
The agent first classifies the developer's prompt into the taxonomy below to guide its search and recommendations. The goal is developer guidance — "this simple task is handled great by *this* (cheaper) model" — not just naming flagships.

*   **Complexity:** Small (formatting/simple chat), Medium (summarization/light logic), Large (complex multi-step logic).
*   **Capability:** General, Reasoning, Coding, **Math**, **Writing** (describing/explaining/summarizing), Creative, Vision.
*   **Security/Privacy:** Public API acceptable, or Private/Local required (sensitive data).
*   **Priority:** **Cost** (cheapest that does the job), **Quality** (best regardless of price), **Speed** (lowest latency), or **Balanced** (default).
*   **Model type:** **Reasoning** (dedicated RL / chain-of-thought models like o3, DeepSeek-R1, QwQ — for hard math/logic/planning) or **Standard** (supervised / instruction-tuned chat models — faster and cheaper, for formatting/chat/writing/simple coding where step-by-step reasoning is overkill).

The **Priority** and **Complexity** axes drive cost-aware guidance: for a Small task or a Cost priority, the agent actively searches for cheaper/smaller/open-weight models and their pricing, rather than pushing an expensive flagship. Each recommendation carries a `cost` field (e.g. `Free (local)`, `$ cheap`, `$$$ premium`, or price-per-1M-tokens when known).

Classification is done via a JSON-mode call to the local LLM: `response_format={"type": "json_object"}`, `temperature=0` (same pattern as `intelligent_router.py`'s `determine_best_model()`). Ollama's JSON-mode support varies by model/version, so parse defensively — `json.loads()` wrapped in a `try/except` that re-prompts once on failure before giving up.

### 2. The Tool Set
The agent needs only one external tool to keep its knowledge current:
*   `search_web(query: str)`: Searches the live web (e.g., LMSYS Chatbot Arena, Artificial Analysis, coding benchmarks, community roundups) using `ddgs` (the renamed `duckduckgo-search` package) — free, no API key, no paid tier. This is where candidate model names come from — the agent has no built-in list.

### 3. The Execution Loop
1.  **Analyze & Classify:** Read user request, identify the taxonomy tags (complexity / capability / security).
2.  **Dynamic Search:** The agent issues one or more web searches shaped by the taxonomy (e.g., `"best local open-weight coding LLMs for security review 2026"`). Raw result text is accumulated into a *corpus*.
3.  **Extract & Rank:** The agent returns up to 5 models it found in the results, ranked best-first, each with a one-line note. Names must come from the search results, not memory. The prompt explicitly excludes non-models (GitHub repos, awesome-lists, tools, frameworks, benchmarks, papers).
4.  **Ground-check & accumulate:** Each returned name is verified against the corpus (`is_grounded()`, normalized alphanumeric substring match) and deduped into a best-first list. If fewer than 5 grounded models are found and steps remain, the agent is asked to search again for more — but **fewer than 5 is acceptable**; the list is never padded with weak/off-topic entries to hit 5. Whatever real models were found are returned ranked best-first (less-good ones after the strong ones).
5.  **User Chooses:** The candidates are printed as a numbered menu; the user picks the one they prefer for their situation. This is the point of the tool — the human makes the final call, not the model.

The cycle is bounded: `for step in range(MAX_STEPS)` with `MAX_STEPS = 6`, so repeated searches can't loop unbounded — same guard convention as module3 and `intelligent_router.py`.

### 4. Key Findings (from testing against locally available models)
Testing against the 3 models already pulled locally (`phi3`, `qwen2.5-coder:1.5b`, `llama3.2`) surfaced two problems, both addressed in the shipped implementation:

*   **Native tool-calling is unreliable on small local models.** `phi3` rejects `tools=[...]` outright (`does not support tools`); `qwen2.5-coder:1.5b` never triggers a tool call at all — it just prints JSON as text; `llama3.2` does trigger a tool call but garbles the arguments (nests the schema itself as the value instead of a plain string). **Fix:** don't use the native `tools=`/`tool_choice` API. Instead, prompt the model to emit a JSON action blob in its plain text reply (`{"action": "search_web", "query": "..."}` or `{"action": "final_answer", "models": [...]}`), parse it manually, and drive the loop off that. This works with any chat model regardless of tool-calling fine-tuning.
*   **Small models hallucinate plausible-sounding but fake model names** (e.g. invented "Claude Mythos 5", "GPT-5.6 Sol") when asked to freely name AI models. **Fix (dynamic, not a fixed list):** require every candidate to come from live search results, then verify each returned name actually appears in the accumulated search corpus (`is_grounded()`, normalized substring match). Ungrounded names are flagged rather than trusted, and because the *user* makes the final choice from the list, a stray flagged entry never becomes a silent wrong recommendation. This keeps the agent fully dynamic — no `MODEL_CATALOG` to maintain — while still catching fabrication.

---

## 🛠️ Implementation Details

### Step 0: The LLM Client — Local Ollama
Use the `openai` SDK's `OpenAI` client pointed at Ollama's OpenAI-compatible endpoint, the same pattern as `local_ide_assistant.py`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
model = "llama3.2"  # whatever chat model is already pulled — no tool-calling support required
```

No API key, no cost, runs entirely on the local machine. Since the agent no longer relies on native tool calling (see Key Findings), any locally pulled chat model works — there's no need to `ollama pull` a specific tool-calling-capable model.

### Step 1: Tool Definition
Define a Python function `search_web(query)` using `ddgs` (free, no API key). It is **not** exposed via the native `tools=[...]` JSON schema — the agent is instructed via its system prompt to request it by emitting a JSON action blob in plain text, which the script parses manually.

### Step 2: The Agent's System Prompt
```text
You are an expert AI Model Advisor.
Given the user's task and its taxonomy: {taxonomy}

Your job is to find which AI models are CURRENTLY best suited for this task, using live
web search — never rely on memory for model names, always ground them in search results.
The current year is {current_year}; include it in your queries so results are current.
If the taxonomy security is "Private", favor local / open-weight models the user can run
on their own machine.

You do NOT have native tool calling. To search the web, reply with ONLY:
{"action": "search_web", "query": "<your search query>"}

You may search a few times to gather enough distinct candidates. When you have enough,
return the best 5 models for this task (ranked best first), based on the taxonomy.
Reply with ONLY:
{"action": "final_answer", "models": [
  {"name": "<model name EXACTLY as it appears in the search results>", "provider": "<company, or 'Local / open-weight'>", "note": "<one-line reason it fits this task>"}
]}

STRICT RULES:
- Return the top 5 candidates, ranked best first for this specific task.
- Every "name" MUST literally have appeared in the search results — never invent or guess.

Always respond with exactly one JSON object, never prose, never markdown fences.
```

After the agent returns, the script ground-checks each name against the search corpus, then prints a numbered menu and lets the **user pick** their preferred model.

### Step 3: Example Flow

**User Input:**
> "I need to build an automated pipeline that reviews our proprietary internal codebase for security vulnerabilities."

**Agent Internal Process:**
*   *Taxonomy detected:* `{"complexity": "Large", "capability": "Coding", "security": "Private"}`
*   *Searches issued (dynamic, taxonomy-driven):* e.g. `search_web("best local open-weight coding LLMs for security vulnerability review 2026")`
*   *Candidates extracted from results, ground-checked against the search corpus.*

**Presented to the user (top 5, then a prompt to choose):**
```
Candidate models found (from live web search):
1. <model> (<provider>): <one-line note>
2. ...
...
5. ...

Which model do you prefer? Enter a number (or blank to skip):
```
The user selects the one that best fits their real-world constraints. (Exact names vary run-to-run since they come from live search — that is the intended dynamic behavior.)

---

## 🔮 Future: Commercial Provider Upgrade Path
Not built now — v1 is Ollama-only. But the code should leave this door open cheaply: wrap client construction in a small factory instead of instantiating `OpenAI(...)` inline.

```python
def get_client(provider="ollama"):
    if provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"), "llama3.1:8b"
    # elif provider == "groq": ...
    # elif provider == "openai": ...
    raise ValueError(f"unknown provider: {provider}")
```

Because the rest of the agent (tool schema, system prompt, execution loop) already speaks the OpenAI-compatible chat/tool-calling format, adding Groq/OpenAI/Anthropic later is just a new `elif` branch here — no rework of the agent logic itself.

## 🚀 Status: Built
Implemented in `ai_advisor_agent/advisor.py`:
1.  ✅ Environment: `ddgs` (free web search) + `openai` SDK (as a generic client against local Ollama) installed in the project's `.venv`.
2.  ✅ Ollama running locally with `llama3.2` (already pulled — no tool-calling-capable model needed, per Key Findings).
3.  ✅ Script implements: taxonomy classifier → dynamic search loop (`MAX_STEPS = 6`) → extract top 5 candidates from search results → ground-check each name against the search corpus → interactive user selection.
4.  ✅ Fully dynamic — no hardcoded model list; recommendations come from live search and stay current automatically.
