"""MCP server that exposes the model router as tools an agent can call.

Two tools:
  - recommend_model(prompt)        → advisory: which model best fits this task, and why
  - run_with_best_model(prompt)    → delegation: route AND run, returning the answer

WHAT THIS CAN AND CANNOT DO
  An MCP tool cannot reach into the host agent (Claude Code / Desktop) and flip its
  model selector — the protocol doesn't allow a tool to change the host's own model.
  What it CAN do is (a) tell the agent which model to use, and (b) *delegate* a
  sub-task: run it on the chosen model (free local Ollama, the `claude` CLI on your
  subscription, or a keyed cloud API) and hand back the result. That is real
  "switching" by offloading — a big session can push a trivial reformat down to a
  free local model, or a hard sub-task up to opus, through one tool call.

EXECUTION POLICY (run_with_best_model)
  Local Ollama models are free and always run. If routing picks a paid/subscription
  model (claude CLI, or any cloud API), the tool returns the DECISION but does NOT
  execute unless the caller passes allow_paid=true — so there's no surprise spend.

Run standalone:   .venv/bin/python ai_advisor_agent/mcp_server.py
Register with Claude Code:
  claude mcp add model-router -- /ABS/PATH/.venv/bin/python /ABS/PATH/ai_advisor_agent/mcp_server.py
"""
from typing import Optional

from mcp.server.fastmcp import FastMCP

import providers
import router

mcp = FastMCP("model-router")


def _public_taxonomy(tax):
    """Drop internal keys (leading underscore) before returning to the agent."""
    return {k: v for k, v in tax.items() if not k.startswith("_")}


def _why(tax, chosen):
    parts = [f"task classified {tax.get('complexity')}/{tax.get('capability')}"]
    if tax.get("model_type") == "Reasoning":
        parts.append("needs a reasoning model")
    if tax.get("_refined"):
        parts.append(f"refined ({tax['_refined']})")
    return (f"{'; '.join(parts)} → {chosen['name']} "
            f"[{chosen['tier']}·{chosen['capability']}·{chosen['cost']}]")


@mcp.tool()
def recommend_model(
    prompt: str,
    only_provider: Optional[str] = None,
    brain: Optional[str] = None,
    scope: Optional[str] = None,
) -> dict:
    """Recommend the best AI model for a given prompt WITHOUT running anything.

    Use this to decide which model a task should go to: it classifies the prompt
    (complexity, capability, reasoning-vs-standard), routes across every available
    provider (local Ollama, the Claude subscription CLI, keyed cloud APIs), and
    returns the single best model plus its taxonomy and reasoning.

    Args:
        prompt: The task/prompt to route.
        only_provider: Restrict routing to one provider, e.g. "claude-cli" to pick
            among only the Claude subscription tiers. For "cloud"/"local" use `scope`, not this.
        brain: Override the classifier model (default: smallest free local model).
        scope: "local" = only Ollama models, "cloud" = only cloud/subscription
            models, None = all providers.
    """
    d = router.decide(prompt, only_provider=only_provider, brain_name=brain, scope=scope)
    tax, chosen = d["taxonomy"], d["chosen"]
    return {
        "recommended_model": chosen["name"],
        "provider": chosen["provider"],
        "tier": chosen["tier"],
        "capability": chosen["capability"],
        "cost": chosen["cost"],
        "runnable_now": chosen.get("runnable", chosen["provider"] == "ollama"),
        "taxonomy": _public_taxonomy(tax),
        "taxonomy_refined": tax.get("_refined"),
        "classifier_brain": d["brain"]["name"],
        "why": _why(tax, chosen),
    }


@mcp.tool()
def run_with_best_model(
    prompt: str,
    only_provider: Optional[str] = None,
    brain: Optional[str] = None,
    allow_paid: bool = False,
    scope: Optional[str] = None,
    concise: bool = False,
) -> dict:
    """Route a prompt to the best model AND run it there, returning the answer.

    This delegates a sub-task to whichever model fits best — offloading trivial
    work to a free local model, or hard work to a larger one. Use it to actually
    get an answer on the right-sized model rather than answering everything on the
    (expensive) host model.

    Execution policy: free local Ollama models always run. If routing picks a
    paid/subscription model (Claude CLI or a cloud API), the decision is returned
    but NOT executed unless allow_paid=true — preventing surprise spend.

    Args:
        prompt: The task/prompt to route and run.
        only_provider: Pin ONE exact provider: claude-cli / anthropic / openai / groq. For "cloud" vs "local" use `scope` instead.
        brain: Override the classifier model.
        allow_paid: Permit executing a paid/subscription model. Default False.
        scope: "local" = only Ollama, "cloud" = only cloud/subscription, None = all.
        concise: Developer-friendly shortcut. Returns only {model, provider, answer}
            (no taxonomy/why) AND asks the model to answer briefly. Default False.
    """
    d = router.decide(prompt, only_provider=only_provider, brain_name=brain, scope=scope,
                      prefer_runnable=True)
    tax, chosen = d["taxonomy"], d["chosen"]
    is_local = chosen["provider"] == "ollama"

    if not is_local and not allow_paid:
        reason = (f"routing chose a paid/subscription model ({chosen['provider']}); "
                  f"call again with allow_paid=true to run it")
        if concise:
            return {"model": chosen["name"], "provider": chosen["provider"],
                    "executed": False, "reason": reason}
        return {"model": chosen["name"], "provider": chosen["provider"],
                "tier": chosen["tier"], "cost": chosen["cost"],
                "taxonomy": _public_taxonomy(tax), "why": _why(tax, chosen),
                "executed": False, "answer": None, "reason": reason}

    to_run = (prompt + "\n\nAnswer concisely and directly, minimal preamble.") if concise else prompt
    answer, note = providers.run_model(chosen, to_run)
    if concise:
        return {"model": chosen["name"], "provider": chosen["provider"], "answer": answer}
    return {"model": chosen["name"], "provider": chosen["provider"],
            "tier": chosen["tier"], "cost": chosen["cost"],
            "taxonomy": _public_taxonomy(tax), "why": _why(tax, chosen),
            "executed": answer is not None, "answer": answer, "note": note}


def main():
    """Run over stdio (default) or HTTP (`--http [--host H] [--port P]`).

    stdio  — for a local agent that launches this process (Claude Code/Desktop).
    http   — hosts a long-running service other agents/machines connect to at
             http://<host>:<port>/mcp (streamable-http transport).
    """
    import sys
    argv = sys.argv[1:]

    def opt(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else default

    if "--http" in argv:
        mcp.settings.host = opt("--host", "127.0.0.1")
        mcp.settings.port = int(opt("--port", "8000"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
