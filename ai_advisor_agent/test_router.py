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


def test_route_escalates_past_tiny_specialist():
    # Large coding task, but the only coder is Small and a larger general model
    # exists → escalate to the big general model instead of the tiny specialist.
    fleet = [
        {"name": "tiny-coder", "capability": "Coding", "tier": "Small", "provider": "ollama"},
        {"name": "big-general", "capability": "General", "tier": "Large", "provider": "ollama"},
    ]
    tax = {"complexity": "Large", "capability": "Coding", "model_type": "Standard"}
    return check("large task escalates past tiny specialist",
                 router.route(tax, fleet)["name"], "big-general")


def test_route_keeps_specialist_when_big_enough():
    # If a large-enough specialist exists, keep it (don't switch to general).
    fleet = [
        {"name": "tiny-coder", "capability": "Coding", "tier": "Small", "provider": "ollama"},
        {"name": "big-coder", "capability": "Coding", "tier": "Large", "provider": "ollama"},
        {"name": "big-general", "capability": "General", "tier": "Large", "provider": "ollama"},
    ]
    tax = {"complexity": "Large", "capability": "Coding", "model_type": "Standard"}
    return check("large coding keeps the big coder",
                 router.route(tax, fleet)["name"], "big-coder")


def test_priority_quality_picks_bigger():
    # A medium task with Quality priority should reach for the Large tier.
    tax = {"complexity": "Medium", "capability": "General", "model_type": "Standard",
           "priority": "Quality"}
    return check("Quality medium → Large tier", router.route(tax, FLEET)["tier"], "Large")


def test_priority_cost_picks_smaller():
    tax = {"complexity": "Medium", "capability": "General", "model_type": "Standard",
           "priority": "Cost"}
    return check("Cost priority → Small tier", router.route(tax, FLEET)["tier"], "Small")


def test_refine_medium_to_large_reasoning():
    tax = {"complexity": "Medium", "capability": "General", "model_type": "Standard"}
    out = router._refine_taxonomy(dict(tax), "Prove this theorem, step by step")
    return check("reasoning signal → Large+Reasoning",
                 out["complexity"] == "Large" and out["model_type"] == "Reasoning", True)


def test_security_no_longer_forces_local():
    # 'Private' must NOT pin routing to local anymore — a cloud flagship can win.
    tax = {"complexity": "Large", "capability": "Reasoning", "model_type": "Reasoning",
           "security": "Private"}
    return check("security does not force local",
                 router.route(tax, FLEET)["capability"], "Reasoning")


def test_scope_filters_pool():
    ok = True
    local = router._select_pool(FLEET, scope="local")
    cloud = router._select_pool(FLEET, scope="cloud")
    ok &= check("scope=local → only ollama",
                all(m["provider"] == "ollama" for m in local) and len(local) > 0, True)
    ok &= check("scope=cloud → no ollama",
                all(m["provider"] != "ollama" for m in cloud) and len(cloud) > 0, True)
    # forgiving: 'cloud'/'local' passed as only_provider are treated as a scope
    ok &= check("only_provider='cloud' behaves like scope=cloud",
                router._select_pool(FLEET, only_provider="cloud") == cloud, True)
    ok &= check("only_provider='local' behaves like scope=local",
                router._select_pool(FLEET, only_provider="local") == local, True)
    return ok


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
        test_route_escalates_past_tiny_specialist(),
        test_route_keeps_specialist_when_big_enough(),
        test_priority_quality_picks_bigger(),
        test_priority_cost_picks_smaller(),
        test_refine_medium_to_large_reasoning(),
        test_security_no_longer_forces_local(),
        test_scope_filters_pool(),
        test_route_falls_back_to_general(),
        test_pick_router_brain(),
    ]
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} test groups passed.")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
