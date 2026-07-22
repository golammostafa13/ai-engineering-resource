"""Model Router — a top layer over many providers. It classifies an incoming
prompt and switches to the best available model to answer it, across BOTH local
Ollama and cloud providers (Anthropic / Claude, OpenAI, Groq, ...).

Simple tasks are routed to small/cheap models, complex tasks to larger ones;
coding tasks to coding models, reasoning tasks to reasoning models, and so on.
The pool of models is auto-discovered at startup by the provider layer (see
providers.py) — local Ollama is always live and free; a cloud provider is
discovered live when its API key is set, otherwise from an offline registry so
the router can still route across providers. A cloud pick is actually executed
only when its key is present; otherwise the switch decision is reported and the
call is skipped (staying free/local by default).

Meant to sit inside a coding assistant / large project: every prompt gets sent
to the cheapest model that can handle it, instead of always paying for a big one.

Usage:
  .venv/bin/python ai_advisor_agent/router.py "Fix the bug in this function"
  .venv/bin/python ai_advisor_agent/router.py --dry-run "Write a haiku"
"""
import json

import advisor  # reuse the taxonomy classifier
import providers
from providers import tag_model  # re-exported so existing tests keep importing router.tag_model

_TIER_ORDER = {"Small": 0, "Medium": 1, "Large": 2}


# Signals of a genuinely hard task that a tiny classifier often underrates as
# "Small". These bump complexity UP so hard work isn't sent to the weakest model.
# This is task understanding, not a model list — safe to extend.
# Reasoning signals: hard, multi-step thinking. They both raise complexity AND
# mark the task as needing a reasoning-grade model.
_REASONING_SIGNALS = (
    "prove", "theorem", "derive", "step by step", "step-by-step", "reason through",
    "logic puzzle", "weigh the tradeoff", "tradeoffs", "trade-offs", "optimal strategy",
    "mathematically", "rigorous",
)
# Strong signals: genuinely hard tasks. Raise complexity to Large from BOTH Small
# and Medium (a weak classifier often underrates these as Medium).
_STRONG_SIGNALS = _REASONING_SIGNALS + (
    "race condition", "data race", "deadlock", "concurrency", "distributed",
    "architect", "design a system", "multi-service", "scalab", "security vulnerab",
    "security review", "consensus", "thread-saf",
)
# Soft signals: non-trivial but not necessarily reasoning-heavy. Raise complexity
# only from Small (don't over-escalate routine Medium work).
_SOFT_SIGNALS = (
    "refactor", "optimize", "migrat", "memory leak", "multi-step", "end-to-end",
    "vulnerab", "mutex", "performance", "bottleneck",
)
# Safe, unambiguous capability signals — only used to correct a clearly-wrong guess.
_VISION_SIGNALS = ("screenshot", "in this image", "in this photo", "this diagram",
                   "what's in this picture", "attached image", "the picture above")
_CODE_SIGNALS = ("```", "def ", "class ", "function ", "traceback", "stack trace",
                 "compile error", "segfault", "null pointer", "unit test", "async def")

_COMPLEXITY_TIER = {"Small": 0, "Medium": 1, "Large": 2}


def _refine_taxonomy(taxonomy, prompt):
    """Strengthen a weak classifier's output using unambiguous prompt signals:
    raise complexity for hard tasks, flag reasoning tasks as needing a reasoning
    model, and correct clearly-wrong capability guesses. Records what changed in
    taxonomy['_refined'] (a string) rather than printing, so it is safe to call
    from an MCP stdio server (where stdout is the protocol)."""
    low = prompt.lower()
    notes = []
    cx = taxonomy.get("complexity", "Medium")

    strong = next((s for s in _STRONG_SIGNALS if s in low), None)
    soft = next((s for s in _SOFT_SIGNALS if s in low), None)
    if strong and _COMPLEXITY_TIER.get(cx, 1) < 2:
        taxonomy["complexity"] = "Large"
        notes.append(f"complexity→Large ('{strong}')")
    elif soft and cx == "Small":
        taxonomy["complexity"] = "Large"
        notes.append(f"complexity→Large ('{soft}')")

    reason = next((s for s in _REASONING_SIGNALS if s in low), None)
    if reason and taxonomy.get("model_type") != "Reasoning":
        taxonomy["model_type"] = "Reasoning"
        notes.append(f"model_type→Reasoning ('{reason}')")

    if any(s in low for s in _VISION_SIGNALS) and taxonomy.get("capability") != "Vision":
        taxonomy["capability"] = "Vision"
        notes.append("capability→Vision (image reference)")
    elif (("```" in prompt or any(s in low for s in _CODE_SIGNALS))
          and taxonomy.get("capability") not in ("Coding", "Math")):
        taxonomy["capability"] = "Coding"
        notes.append("capability→Coding (code detected)")

    if notes:
        taxonomy["_refined"] = "; ".join(notes)
    return taxonomy


def _loose_json(text):
    """Parse a JSON object even if wrapped in prose / markdown fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in classifier reply: {text!r}")
    return json.loads(text[start:end + 1])


def _classify(brain, user_prompt, ollama_client):
    """Classify the prompt with the chosen 'router brain'. Ollama supports JSON
    mode directly; any other provider is asked (via its normal run path) to reply
    with JSON only, which we then parse loosely."""
    if brain["provider"] == "ollama":
        return advisor.classify_request(ollama_client, brain["name"], user_prompt)
    prompt = advisor.CLASSIFY_PROMPT.format(request=user_prompt) + \
        "\n\nReply with ONLY the JSON object — no prose, no markdown fences."
    raw, note = providers.run_model(brain, prompt, temperature=0)
    if raw is None:
        raise RuntimeError(f"router brain unavailable: {note}")
    return _loose_json(raw)


def get_client():
    # The classifier ("router brain") always runs locally and free.
    return providers.OpenAI(base_url=f"{providers.OLLAMA_HOST}/v1", api_key="ollama")


def discover_models():
    """Every routable model across all providers (see providers.discover_all)."""
    return providers.discover_all()


def pick_router_brain(models):
    """Smallest LOCAL general model — the classification step must be cheap, fast,
    and free, so we never spend a cloud call just to decide where to route."""
    local = [m for m in models if m.get("provider", "ollama") == "ollama"]
    pool = [m for m in local if m["capability"] == "General"] or local or models
    return min(pool, key=lambda m: _TIER_ORDER[m["tier"]])["name"]


def _target_tier(complexity, priority):
    """Which tier index (0=Small..2=Large) to aim for, given task complexity and
    the developer's priority. Quality trades cost for a bigger model; Cost/Speed
    trade quality for a smaller/faster one; Balanced follows complexity."""
    t = _COMPLEXITY_TIER.get(complexity, 1)
    if priority == "Quality":
        t = min(2, t + 1)
    elif priority in ("Cost", "Speed"):
        t = max(0, t - 1)
    return t


def route(taxonomy, models):
    """Pick the best available model for a classified task, across all providers."""
    cap = taxonomy.get("capability", "General")
    complexity = taxonomy.get("complexity", "Medium")
    priority = taxonomy.get("priority", "Balanced")
    wants_reasoning = taxonomy.get("model_type") == "Reasoning"
    target = _target_tier(complexity, priority)

    # (The 'security' axis is intentionally NOT used to force local routing —
    # use the -l/--local scope flag if you want to restrict to local models.)

    # 1) Prefer a dedicated reasoning model when the task wants one.
    if wants_reasoning:
        reasoning = [m for m in models if m["capability"] == "Reasoning"]
        if reasoning:
            reasoning.sort(key=lambda m: _TIER_ORDER[m["tier"]])
            return reasoning[-1]    # hardest thinking → biggest reasoning model
        # No dedicated reasoning model (e.g. Claude-only): use the strongest
        # available general-purpose model rather than the cheapest tier.
        pool = [m for m in models if m["capability"] == cap] or \
               [m for m in models if m["capability"] == "General"] or list(models)
        pool.sort(key=lambda m: _TIER_ORDER[m["tier"]])
        return pool[-1]

    # 2) Otherwise narrow by capability: exact match, then General, then anything.
    candidates = [m for m in models if m["capability"] == cap]
    if not candidates:
        candidates = [m for m in models if m["capability"] == "General"]
    if not candidates:
        candidates = list(models)

    # 2b) Escalation: if the target tier is bigger than the largest capability-
    #     matched model, a tiny specialist (e.g. a 1.5B coder for a large refactor)
    #     is the wrong pick. Widen the pool to larger general-purpose models so the
    #     task can win. Only kicks in when a genuinely bigger model actually exists.
    best_tier = max(_TIER_ORDER[m["tier"]] for m in candidates)
    if target > best_tier:
        bigger = [m for m in models
                  if m["capability"] in (cap, "General")
                  and _TIER_ORDER[m["tier"]] > best_tier]
        candidates = candidates + bigger  # bigger may be empty → unchanged

    # 3) Pick the tier closest to the target (complexity adjusted by priority).
    candidates.sort(key=lambda m: _TIER_ORDER[m["tier"]])
    if target <= 0:
        return candidates[0]        # smallest / cheapest handles simple tasks
    if target >= 2:
        return candidates[-1]       # biggest available for hard tasks
    return candidates[len(candidates) // 2]  # middle tier for medium tasks


_SCOPE_WORDS = {"local", "cloud"}


def _select_pool(full, only_provider=None, scope=None):
    """Narrow the routable pool. only_provider pins one provider (e.g. 'claude-cli');
    scope='local' keeps only Ollama, scope='cloud' keeps everything else.

    Forgiving: 'local'/'cloud' passed as only_provider are treated as a scope (a
    common mistake — they're scopes, not provider names)."""
    if only_provider in _SCOPE_WORDS:      # caller meant a scope, not a provider
        scope, only_provider = only_provider, None
    if only_provider:
        return [m for m in full if m["provider"] == only_provider]
    if scope == "local":
        return [m for m in full if m["provider"] == "ollama"]
    if scope == "cloud":
        return [m for m in full if m["provider"] != "ollama"]
    return full


def decide(user_prompt, only_provider=None, brain_name=None, scope=None, prefer_runnable=False):
    """Pure routing decision — classify + route, NO printing and NO execution.
    Returns {taxonomy, chosen, brain, pool}. Safe to call from an MCP stdio
    server (stdout stays clean for the protocol).

    scope: 'local' | 'cloud' | None. prefer_runnable drops models that can't run
    right now (no API key / CLI not logged in) so a task about to be executed lands
    on something usable — e.g. scope='cloud' resolves to your Claude subscription
    rather than a keyless API model. (Falls back to the full pool if that empties it.)"""
    ollama_client = get_client()
    full = discover_models()  # every provider — used to pick a cheap local brain
    if not full:
        raise RuntimeError(
            "No models found. Start Ollama (`ollama pull llama3.2`), log in to the "
            "`claude` CLI, and/or set a cloud API key (e.g. ANTHROPIC_API_KEY)."
        )

    # The routable pool may be narrowed (--claude / -l / -c), but the classifier
    # brain is always chosen from the FULL pool so it can stay local and free.
    models = _select_pool(full, only_provider, scope)
    if not models:
        provs = sorted({m["provider"] for m in full})
        raise RuntimeError(
            f"no models for provider={only_provider!r}, scope={scope!r}. "
            f"Valid providers: {provs}; or use scope 'local'/'cloud'.")
    if prefer_runnable:
        runnable = [m for m in models if m.get("runnable")]
        models = runnable or models

    chosen_brain = brain_name or pick_router_brain(full)
    brain = next((m for m in full if m["name"] == chosen_brain), None)
    if brain is None:
        raise RuntimeError(f"router brain '{chosen_brain}' not found among discovered models.")
    taxonomy = _classify(brain, user_prompt, ollama_client)
    taxonomy = _refine_taxonomy(taxonomy, user_prompt)
    chosen = route(taxonomy, models)
    return {"taxonomy": taxonomy, "chosen": chosen, "brain": brain, "pool": models}


def _print_decision(d):
    """Verbose print of the routing decision (model listing + taxonomy + route)."""
    taxonomy, chosen, brain, models = d["taxonomy"], d["chosen"], d["brain"], d["pool"]
    print("Available models (auto-discovered across providers):")
    for m in models:
        live = "live" if m.get("live") else "registry"
        print(f"  - {m['name']}  [{m['tier']} · {m['capability']} · {m['cost']}]"
              f"  ({m['provider']}, {live})")
    print(f"\nClassifying with router brain: {brain['name']} ({brain['provider']})")
    if taxonomy.get("_refined"):
        print(f"(refined by prompt signals: {taxonomy['_refined']})")
    print(f"Taxonomy: {taxonomy}")
    print(f"\n→ Routed to: {chosen['name']}  [{chosen['tier']} · {chosen['capability']} · "
          f"{chosen['cost']}]  ({chosen['provider']})  "
          f"(complexity={taxonomy.get('complexity')}, capability={taxonomy.get('capability')}, "
          f"model_type={taxonomy.get('model_type')})")


def run(user_prompt, execute=True, only_provider=None, brain_name=None, scope=None, concise=False):
    """CLI entry: decide (with printing) then optionally execute the prompt on the
    chosen model. execute=False is a fast 'dry run' that stops after the decision.
    concise=True suppresses the model listing/taxonomy and asks for a brief answer."""
    d = decide(user_prompt, only_provider=only_provider, brain_name=brain_name,
               scope=scope, prefer_runnable=execute)
    chosen = d["chosen"]

    if concise:
        print(f"→ {chosen['name']} [{chosen['tier']}·{chosen['capability']}] ({chosen['provider']})")
    else:
        _print_decision(d)

    if not execute:
        print("(dry run)" if concise else "\n(dry run — skipping generation)")
        return chosen, None

    to_run = (user_prompt + "\n\nAnswer concisely and directly, minimal preamble.") if concise else user_prompt
    answer, note = providers.run_model(chosen, to_run)
    if answer is None:
        print(f"({note})" if concise else f"\n(cloud call {note})")
    else:
        print(answer if concise else "\nAnswer:\n" + answer)
    return chosen, answer


def main():
    import sys
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv

    # --claude is shorthand for --provider claude-cli (use the subscription only).
    only_provider = "claude-cli" if "--claude" in argv else None
    if "--provider" in argv:
        i = argv.index("--provider")
        only_provider = argv[i + 1] if i + 1 < len(argv) else only_provider
        argv = argv[:i] + argv[i + 2:]

    # --brain <model> overrides the classifier (e.g. a stronger model than phi3).
    brain_name = None
    if "--brain" in argv:
        i = argv.index("--brain")
        brain_name = argv[i + 1] if i + 1 < len(argv) else None
        argv = argv[:i] + argv[i + 2:]

    # -l/--local restrict to local models, -c/--cloud restrict to cloud/subscription.
    scope = None
    if "-l" in argv or "--local" in argv:
        scope = "local"
    elif "-c" in argv or "--cloud" in argv:
        scope = "cloud"

    # -q/--quiet: concise output (just the routed model + a brief answer).
    concise = "-q" in argv or "--quiet" in argv

    args = [a for a in argv if not a.startswith("-")]
    user_prompt = " ".join(args).strip() or input("Enter your prompt: ").strip()
    run(user_prompt, execute=not dry_run, only_provider=only_provider,
        brain_name=brain_name, scope=scope, concise=concise)


if __name__ == "__main__":
    main()
