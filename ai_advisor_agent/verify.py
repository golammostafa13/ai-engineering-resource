"""One-command verification for the whole tool (advisor + router + MCP server).

    ../.venv/bin/python ai_advisor_agent/verify.py           # fast checks only (<5s)
    ../.venv/bin/python ai_advisor_agent/verify.py --live    # also run real LLM end-to-end

Fast checks are deterministic and need no network: unit tests, provider discovery,
MCP tool wiring, and the routing decision logic (privacy guard, complexity boost,
escalation, tier/capability routing). --live additionally drives real model calls
through Ollama and (if logged in) the Claude CLI, which are slower.
"""
import asyncio
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

_passed = 0
_failed = 0


def check(label, ok, detail=""):
    global _passed, _failed
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if ok:
        _passed += 1
    else:
        _failed += 1
    return ok


def _run_unit_file(name):
    r = subprocess.run([PY, str(HERE / name)], capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    check(f"{name} exits 0", r.returncode == 0, tail)


def unit_tests():
    print("\n== Unit tests (pure logic) ==")
    _run_unit_file("test_advisor.py")
    _run_unit_file("test_router.py")


def discovery_and_wiring():
    print("\n== Discovery + MCP wiring ==")
    import providers
    models = providers.discover_all()
    provs = {m["provider"] for m in models}
    check("Ollama models discovered (live)",
          any(m["provider"] == "ollama" and m.get("runnable") for m in models),
          f"{sum(m['provider']=='ollama' for m in models)} local model(s)")
    check("Claude subscription tiers present", "claude-cli" in provs,
          "haiku/sonnet/opus via CLI" if "claude-cli" in provs else "not registered")
    check("Cloud registry loaded", {"anthropic", "openai"} <= provs, sorted(provs))
    for m in models:  # every model must be fully tagged
        keys = {"name", "provider", "tier", "capability", "cost"}
        if not keys <= set(m):
            check("all models fully tagged", False, m.get("name"))
            break
    else:
        check("all models fully tagged", True, f"{len(models)} models")

    import mcp_server
    tools = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    check("MCP exposes recommend_model", "recommend_model" in tools)
    check("MCP exposes run_with_best_model", "run_with_best_model" in tools)


def routing_logic():
    print("\n== Routing decision logic (offline) ==")
    import router
    local = [
        {"name": "coder-1.5b", "capability": "Coding", "tier": "Small", "provider": "ollama"},
        {"name": "gen-small", "capability": "General", "tier": "Small", "provider": "ollama"},
    ]
    mixed = local + [
        {"name": "gen-large", "capability": "General", "tier": "Large", "provider": "ollama"},
        {"name": "opus", "capability": "General", "tier": "Large", "provider": "claude-cli"},
        {"name": "r1", "capability": "Reasoning", "tier": "Large", "provider": "ollama"},
    ]

    def tax(**kw):
        base = {"complexity": "Medium", "capability": "General",
                "security": "Public", "model_type": "Standard"}
        base.update(kw)
        return base

    check("simple task → cheapest small",
          router.route(tax(complexity="Small"), mixed)["tier"] == "Small")
    check("coding task → coding model",
          router.route(tax(complexity="Small", capability="Coding"), mixed)["capability"] == "Coding")
    check("reasoning flag → reasoning model",
          router.route(tax(complexity="Large", model_type="Reasoning"), mixed)["capability"] == "Reasoning")
    check("scope=local restricts to Ollama",
          all(m["provider"] == "ollama" for m in router._select_pool(mixed, scope="local")))
    check("scope=cloud excludes Ollama",
          all(m["provider"] != "ollama" for m in router._select_pool(mixed, scope="cloud")))
    escalated = router.route(tax(complexity="Large", capability="Coding"), mixed)
    check("large task escalates past tiny specialist",
          escalated["tier"] == "Large" and escalated["name"] != "coder-1.5b",
          f"chose {escalated['name']} [{escalated['tier']}]")
    refined = router._refine_taxonomy(tax(complexity="Small"), "please refactor this race condition")
    check("hard-signal boosts complexity Small→Large", refined["complexity"] == "Large",
          f"refined='{refined.get('_refined')}'")
    # strong reasoning signal bumps Medium→Large AND flags a reasoning model
    proof = router._refine_taxonomy(tax(complexity="Medium"), "prove this theorem rigorously")
    check("reasoning signal: Medium→Large + model_type=Reasoning",
          proof["complexity"] == "Large" and proof["model_type"] == "Reasoning")
    # priority-aware tiering: Quality shifts up, Cost shifts down
    check("priority Quality shifts tier up", router._target_tier("Medium", "Quality") == 2)
    check("priority Cost shifts tier down", router._target_tier("Medium", "Cost") == 0)


def live_checks():
    print("\n== Live end-to-end (real model calls; slower) ==")

    def run_cli(args, expect):
        r = subprocess.run([PY, str(HERE / "router.py"), *args],
                           capture_output=True, text=True, timeout=600)
        # Only inspect the decision line — not the whole output (the model listing
        # mentions every model, which would cause false-positive substring matches).
        line = next((l for l in r.stdout.splitlines() if "Routed to" in l), "(no route line)")
        check(f"router {' '.join(args[:2])} … → provider {expect}", expect in line, line.strip())

    # Robust expectations: a tiny classifier can't be pinned to an exact tier, but
    # the PROVIDER a prompt routes to is stable and meaningful.
    run_cli(["--dry-run", "What is 2 plus 2?"], "ollama")          # simple → free local
    run_cli(["--claude", "--dry-run", "Prove sqrt(2) is irrational, step by step"], "claude-cli")

    print("  (MCP delegation)")
    import mcp_server
    res = mcp_server.run_with_best_model("Reply with just the number: 2+2")
    check("run_with_best_model executes a local pick",
          res.get("executed") and res.get("provider") == "ollama",
          f"model={res.get('model')} answer={str(res.get('answer'))[:20]!r}")


def main():
    live = "--live" in sys.argv
    print("Verifying AI Model Router / Advisor" + (" (with live checks)" if live else ""))
    unit_tests()
    discovery_and_wiring()
    routing_logic()
    if live:
        live_checks()
    print(f"\n{'=' * 40}\n{_passed} passed, {_failed} failed")
    if not live:
        print("(run with --live to also exercise real model calls)")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
