"""AI Model Advisor Agent — helps a developer pick the right AI model for a task.

It classifies the task into a taxonomy (complexity, capability, security, and
priority such as cost/quality/speed), searches the web for current models that
fit, and presents a short ranked list — cost included — for the developer to
choose from. The aim is guidance: "this simple task is handled great by this
cheaper model", not just naming flagships.

Fully dynamic: there is no hardcoded model list. Candidates come from live web
search results at run time, so recommendations stay current as new models ship.
A grounding check keeps the local model from inventing names — every candidate
must actually appear in the search results.

Runs fully local and free via Ollama. Single local user, no auth required.
"""
import json
from datetime import datetime

from ddgs import DDGS
from openai import OpenAI

MAX_STEPS = 6


def get_client(provider="ollama"):
    """Factory for the LLM client. Add new providers as new elif branches —
    the rest of the agent only speaks the OpenAI-compatible chat format.
    """
    if provider == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"), "llama3.2"
    raise ValueError(f"unknown provider: {provider}")


def search_web(query, max_results=6):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})")
    return "\n".join(lines)


# Terms that signal a query is actually about LLMs/AI models, not apps or tools.
_MODEL_ANCHORS = ("llm", "language model", "ai model", "gpt", "claude", "llama",
                  "gemini", "mistral", "qwen", "deepseek")


def _anchor_query(query):
    """Keep searches pointed at AI models. If the model's query doesn't already
    mention an LLM/AI-model term, append anchors so results are models, not apps.
    """
    low = query.lower()
    if any(term in low for term in _MODEL_ANCHORS):
        return query
    return f"{query} LLM AI language model"


def _normalize(text):
    """Lowercase, alphanumeric-only — so 'GPT-4o', 'GPT 4o', 'gpt4o' all match."""
    return "".join(c for c in text.lower() if c.isalnum())


def is_grounded(name, corpus):
    """True if the model name actually appears somewhere in the search corpus."""
    return bool(name) and _normalize(name) in _normalize(corpus)


CLASSIFY_PROMPT = """You are a task classifier. Classify the developer's request into a taxonomy.

Respond with ONLY a JSON object, no other text:
{{
  "complexity": "Small" | "Medium" | "Large",
  "capability": "General" | "Reasoning" | "Coding" | "Math" | "Writing" | "Creative" | "Vision",
  "security": "Public" | "Private",
  "priority": "Cost" | "Quality" | "Speed" | "Balanced",
  "model_type": "Standard" | "Reasoning"
}}

Complexity: Small (formatting/simple chat), Medium (summarization/light logic), Large (complex multi-step logic).
Capability — pick the single best-fitting category:
  - General:   everyday chat / assistant
  - Reasoning: multi-step logic, planning, analysis
  - Coding:    writing or reviewing code
  - Math:      calculation, proofs, quantitative problem solving
  - Writing:   describing, explaining, summarizing, documentation
  - Creative:  stories, marketing, ideation
  - Vision:    images, diagrams, screenshots
Security: "Private" if sensitive/proprietary data must stay on the user's machine, otherwise "Public".
Priority — what the developer optimizes for:
  - Cost:     cheapest model that can still do the job well
  - Quality:  best possible result, price is no object
  - Speed:    lowest latency
  - Balanced: reasonable tradeoff (use this if unclear)
Model_type — which KIND of model the task calls for:
  - Reasoning: dedicated reasoning models (RL-trained, chain-of-thought — e.g. o3, DeepSeek-R1,
    QwQ). Choose for hard math, multi-step logic, complex planning, deep analysis.
  - Standard:  regular supervised / instruction-tuned chat models (faster, cheaper). Choose for
    formatting, chat, summarizing, writing, simple coding — where step-by-step reasoning is
    overkill and adds cost/latency.

User request: {request}"""


def classify_request(client, model, user_request):
    prompt = CLASSIFY_PROMPT.format(request=user_request)
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = resp.choices[0].message.content
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            if attempt == 0:
                prompt = CLASSIFY_PROMPT.format(request=user_request) + \
                    "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
                continue
            raise ValueError(f"classifier did not return valid JSON: {content!r}")


AGENT_SYSTEM_PROMPT = """You are an expert AI Model Advisor helping a developer choose the
right model for their task. Task taxonomy: {taxonomy}

Your job is to find which AI models are CURRENTLY best suited for this task, using live
web search — never rely on memory for model names, always ground them in search results.
The current year is {current_year}; include it in your queries so results are current.

GUIDANCE — match the model to the job, don't overspend:
- Every search query MUST be about AI language models / LLMs — always include a term like
  "LLM", "AI model", or "language model". You are recommending MODELS, not apps, SaaS
  products, or libraries. A good query looks like "cheapest small LLM for simple text
  tasks 2026", NOT "tool for reformatting chat messages".
- Shape your search queries around the taxonomy: the capability (e.g. "math", "coding",
  "writing/describing"), the priority, and the complexity.
- For a Small/General task, the answer is usually any cheap general-purpose LLM (e.g. a
  "mini"/"flash"/"haiku"-tier or small open-weight model) — search for those.
- If priority is "Cost" OR complexity is "Small": actively look for cheaper / smaller /
  open-weight models that handle this task well, and search for their pricing. The goal is
  to tell the developer "this simple task is handled great by this cheaper model" instead
  of pushing an expensive flagship.
- If priority is "Quality": favor top-benchmark flagship models.
- If priority is "Speed": favor small/fast/low-latency models.
- If security is "Private": favor local / open-weight models the user can self-host.
- If model_type is "Reasoning": recommend dedicated reasoning models (RL / chain-of-thought,
  e.g. o3, DeepSeek-R1, QwQ, Gemini Thinking) and search specifically for those.
- If model_type is "Standard": recommend regular instruction-tuned chat models and do NOT
  push expensive reasoning models — they add cost/latency this task does not need.

You do NOT have native tool calling. To search the web, reply with ONLY this JSON object
and nothing else:
{{"action": "search_web", "query": "<your search query>"}}

You may search a few times to gather enough distinct candidates. When you have enough,
return the best models for this task (ranked best first). Reply with ONLY this JSON object
and nothing else:
{{"action": "final_answer", "models": [
  {{"name": "<model name EXACTLY as it appears in the search results>", "provider": "<company, or 'Local / open-weight'>", "cost": "<pricing if known: e.g. 'Free (local)', '$ cheap', '$$ mid', '$$$ premium', or price per 1M tokens>", "note": "<one-line reason it fits this task and priority>"}}
]}}

STRICT RULES for the final answer:
- Return UP TO 5 candidates, ranked best first for this specific task. Fewer than 5 is
  fine if that is all the genuinely relevant real models you found — NEVER pad the list
  with weak or off-topic entries just to reach 5. If you found more strong ones, still cap
  at 5. Always rank best first; include less-good ones only after the strong ones.
- Every "name" MUST be a specific, VERSIONED AI model that literally appeared in the
  search results — e.g. "Claude Opus 4.8", "Claude Sonnet 5", "GPT-4o mini", "Gemini 3
  Pro", "Llama 3.1 70B", "DeepSeek-R1", "Qwen2.5-Coder 32B". Include the version/tier,
  never a bare family name like "Claude", "GPT", or "Gemini".
- NEVER include non-models. In particular:
    * NOT products/IDEs/agent platforms that merely USE a model (e.g. Antigravity, Cursor,
      Copilot, Pumble) — recommend the underlying MODEL instead.
    * NOT GitHub repos, "awesome-*" lists, libraries, frameworks, benchmarks, leaderboards,
      companies, or research-paper titles.
- Do NOT invent, rename, or guess names — if you did not see it in the results, omit it.

Always respond with exactly one JSON object, never prose, never markdown fences."""


def _parse_json_action(content):
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _accumulate_grounded(models, corpus, best, seen):
    """Add real, grounded, not-yet-seen models to `best` (order = best-first)."""
    for m in models:
        key = _normalize(m.get("name", ""))
        if key and key not in seen and is_grounded(m.get("name", ""), corpus):
            m["grounded"] = True
            seen.add(key)
            best.append(m)


def run_agent(user_request, provider="ollama"):
    """Return (taxonomy, candidates). Candidates are dicts with name/provider/note
    plus a 'grounded' flag indicating whether the name appeared in search results.
    """
    client, model = get_client(provider)

    taxonomy = classify_request(client, model, user_request)
    print(f"Taxonomy: {taxonomy}")

    system_prompt = AGENT_SYSTEM_PROMPT.format(
        taxonomy=json.dumps(taxonomy),
        current_year=datetime.now().year,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]

    corpus = ""  # accumulated raw search text, used to ground candidate names
    best = []    # grounded candidates found so far, ranked best-first, deduped
    seen = set()

    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
        content = resp.choices[0].message.content

        try:
            action = _parse_json_action(content)
        except (json.JSONDecodeError, TypeError):
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "That was not valid JSON. Reply with ONLY the JSON object described in your instructions.",
            })
            continue

        if action.get("action") == "search_web":
            query = _anchor_query(action.get("query", ""))
            print(f"[step {step + 1}] searching: {query}")
            results = search_web(query)
            corpus += "\n" + results
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Search results:\n{results}"})
            continue

        if action.get("action") == "final_answer":
            models = action.get("models", [])
            if not corpus:
                # Model tried to answer before searching — force a search first.
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "You must search_web at least once before answering. Reply with a search_web action.",
                })
                continue

            # Keep only real models grounded in the corpus (deduped, best-first).
            _accumulate_grounded(models, corpus, best, seen)

            # Enough strong candidates, or out of steps — return the top 5 found.
            if len(best) >= 5 or step == MAX_STEPS - 1:
                return taxonomy, best[:5]

            # Not 5 yet and steps remain — ask for more (fewer is fine, never pad).
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"So far only {len(best)} valid model(s) confirmed. Search again with "
                    "different queries to find more REAL AI models (no repos/tools/papers), "
                    "then return the full ranked list."
                ),
            })
            continue

        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": 'Unrecognized action. Use only "search_web" or "final_answer" as described.',
        })

    # Steps exhausted without a final_answer that met the bar — return best effort.
    return taxonomy, best[:5]


def choose_model(candidates):
    """Present the candidates and let the user pick the one they prefer."""
    if not candidates:
        print("\nNo candidate models were found. Try rephrasing your task.")
        return None

    print("\nCandidate models found (from live web search):")
    for i, m in enumerate(candidates, 1):
        flag = "" if m.get("grounded") else "  ⚠ not confirmed in search results"
        cost = f" [{m.get('cost')}]" if m.get("cost") else ""
        print(f"{i}. {m.get('name')} ({m.get('provider')}){cost}: {m.get('note')}{flag}")

    while True:
        choice = input("\nWhich model do you prefer? Enter a number (or blank to skip): ").strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print(f"Please enter a number between 1 and {len(candidates)}.")


def main():
    user_request = input("Describe the task you need an AI model for: ").strip()
    _, candidates = run_agent(user_request)
    chosen = choose_model(candidates)
    if chosen:
        print(f"\n✅ You chose: {chosen.get('name')} ({chosen.get('provider')})")


if __name__ == "__main__":
    main()
