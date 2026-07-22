"""Fast unit tests for router.py's pure routing logic — no Ollama or network.

Run:  .venv/bin/python ai_advisor_agent/test_router.py
"""
import router

# A pretend inventory spanning tiers, capabilities AND providers (local + cloud).
FLEET = [
    {"name": "qwen2.5-coder:1.5b", "capability": "Coding", "tier": "Small", "provider": "ollama"},
    {"name": "llama3.2", "capability": "General", "tier": "Small", "provider": "ollama"},
    {"name": "llama3.3:70b", "capability": "General", "tier": "Large", "provider": "ollama"},
    {"name": "deepseek-r1:32b", "capability": "Reasoning", "tier": "Large", "provider": "ollama"},
    {"name": "qwen2.5-coder:32b", "capability": "Coding", "tier": "Large", "provider": "ollama"},
    # cloud models — routable across providers, tagged from the registry
    {"name": "claude-opus-4-8", "capability": "Reasoning", "tier": "Large", "provider": "anthropic"},
    {"name": "claude-haiku-4-5", "capability": "General", "tier": "Small", "provider": "anthropic"},
]


def check(label, got, expected):
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got {got!r}")
    return ok


def test_tag_model():
    ok = True
    ok &= check("coder->Coding", router.tag_model("qwen2.5-coder:1.5b", 10**9)[0], "Coding")
    ok &= check("r1->Reasoning", router.tag_model("deepseek-r1:32b", 10**9)[0], "Reasoning")
    ok &= check("plain->General", router.tag_model("llama3.2", 2 * 10**9)[0], "General")
    ok &= check("1.5b->Small", router.tag_model("foo:1.5b", 10**9)[1], "Small")
    ok &= check("70b->Large", router.tag_model("foo:70b", 10**9)[1], "Large")
    ok &= check("no-param uses size (small)", router.tag_model("mystery", 2 * 10**9)[1], "Small")
    return ok


def test_route_simple_to_small():
    # Simple general task -> smallest general model.
    tax = {"complexity": "Small", "capability": "General", "model_type": "Standard"}
    return check("simple->small general", router.route(tax, FLEET)["name"], "llama3.2")


def test_route_complex_coding_to_large_coder():
    tax = {"complexity": "Large", "capability": "Coding", "model_type": "Standard"}
    return check("complex coding->large coder", router.route(tax, FLEET)["name"], "qwen2.5-coder:32b")


def test_route_reasoning_flag_wins():
    # Even a 'Coding' capability task routes to a reasoning model when flagged.
    tax = {"complexity": "Large", "capability": "Coding", "model_type": "Reasoning"}
    return check("reasoning flag->reasoning model", router.route(tax, FLEET)["capability"], "Reasoning")


def test_route_hard_task_can_go_cloud():
    # A large reasoning task is allowed to switch UP to a cloud flagship.
    tax = {"complexity": "Large", "capability": "Reasoning", "model_type": "Reasoning"}
    chosen = router.route(tax, FLEET)
    return check("hard task -> cloud flagship allowed", chosen["provider"] in ("ollama", "anthropic"), True)


def test_route_simple_task_stays_cheap():
    # A trivial task never escalates to a premium cloud model.
    tax = {"complexity": "Small", "capability": "General", "model_type": "Standard"}
    return check("simple task -> Small tier (cheap)", router.route(tax, FLEET)["tier"], "Small")


def test_route_private_stays_local():
    # A Private task must never route to a cloud provider, even if a cloud
    # flagship would otherwise be the best fit.
    tax = {"complexity": "Large", "capability": "Reasoning", "model_type": "Reasoning",
           "security": "Private"}
    return check("private -> local provider", router.route(tax, FLEET)["provider"], "ollama")


def test_route_falls_back_to_general():
    # Vision task with no vision model -> falls back to General.
    tax = {"complexity": "Small", "capability": "Vision", "model_type": "Standard"}
    return check("vision missing->general", router.route(tax, FLEET)["capability"], "General")


def test_pick_router_brain():
    return check("brain = smallest general", router.pick_router_brain(FLEET), "llama3.2")


def main():
    results = [
        test_tag_model(),
        test_route_simple_to_small(),
        test_route_complex_coding_to_large_coder(),
        test_route_reasoning_flag_wins(),
        test_route_hard_task_can_go_cloud(),
        test_route_simple_task_stays_cheap(),
        test_route_private_stays_local(),
        test_route_falls_back_to_general(),
        test_pick_router_brain(),
    ]
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} test groups passed.")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
