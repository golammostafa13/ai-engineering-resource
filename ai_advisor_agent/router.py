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
import advisor  # reuse the taxonomy classifier
import providers
from providers import tag_model  # re-exported so existing tests keep importing router.tag_model

_TIER_ORDER = {"Small": 0, "Medium": 1, "Large": 2}


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
    return sorted(pool, key=lambda m: _TIER_ORDER[m["tier"]])[0]["name"]


def route(taxonomy, models):
    """Pick the best available model for a classified task, across all providers."""
    cap = taxonomy.get("capability", "General")
    complexity = taxonomy.get("complexity", "Medium")
    wants_reasoning = taxonomy.get("model_type") == "Reasoning"

    # 0) Privacy guard: a task flagged Private must never leave the machine, so
    #    restrict the pool to local models. (Fall back to the full pool only if
    #    no local model exists at all, so routing still returns something.)
    if taxonomy.get("security") == "Private":
        local = [m for m in models if m.get("provider", "ollama") == "ollama"]
        models = local or models

    # 1) Narrow by capability. Reasoning flag wins; then exact capability; then
    #    General; then anything, so we always have at least one candidate.
    candidates = []
    if wants_reasoning:
        candidates = [m for m in models if m["capability"] == "Reasoning"]
    if not candidates:
        candidates = [m for m in models if m["capability"] == cap]
    if not candidates:
        candidates = [m for m in models if m["capability"] == "General"]
    if not candidates:
        candidates = list(models)

    # 2) Pick the tier that matches task complexity.
    candidates.sort(key=lambda m: _TIER_ORDER[m["tier"]])
    if complexity == "Small":
        return candidates[0]        # smallest / cheapest handles simple tasks
    if complexity == "Large":
        return candidates[-1]       # biggest available for hard tasks
    return candidates[len(candidates) // 2]  # middle for medium tasks


def run(user_prompt, execute=True):
    """Classify -> route -> (optionally) run. With execute=False it stops after
    the routing decision (a fast 'dry run' for verifying which model is picked)."""
    client = get_client()
    models = discover_models()
    if not models:
        raise RuntimeError(
            "No models found. Start Ollama (`ollama pull llama3.2`) and/or set a "
            "cloud API key (e.g. ANTHROPIC_API_KEY)."
        )

    print("Available models (auto-discovered across providers):")
    for m in models:
        live = "live" if m.get("live") else "registry"
        print(f"  - {m['name']}  [{m['tier']} · {m['capability']} · {m['cost']}]"
              f"  ({m['provider']}, {live})")

    brain = pick_router_brain(models)
    print(f"\nClassifying with router brain: {brain}")
    taxonomy = advisor.classify_request(client, brain, user_prompt)
    print(f"Taxonomy: {taxonomy}")

    chosen = route(taxonomy, models)
    print(f"\n→ Routed to: {chosen['name']}  [{chosen['tier']} · {chosen['capability']} · "
          f"{chosen['cost']}]  ({chosen['provider']})  "
          f"(complexity={taxonomy.get('complexity')}, capability={taxonomy.get('capability')}, "
          f"model_type={taxonomy.get('model_type')})")

    if not execute:
        print("\n(dry run — skipping generation)")
        return chosen, None

    answer, note = providers.run_model(chosen, user_prompt)
    if answer is None:
        print(f"\n(cloud call {note})")
    else:
        print("\nAnswer:\n" + answer)
    return chosen, answer


def main():
    import sys
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    user_prompt = " ".join(args).strip() or input("Enter your prompt: ").strip()
    run(user_prompt, execute=not dry_run)


if __name__ == "__main__":
    main()
