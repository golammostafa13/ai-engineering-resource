"""Demo: show the router actually SWITCHING models per prompt.

Runs a set of diverse prompts through the same router and prints which model each
one lands on — so you can see it change (small local ↔ coder ↔ subscription tier)
based purely on the prompt. Decision only (no generation), so it's as fast as the
classifier allows.

    ../.venv/bin/python ai_advisor_agent/demo_switch.py           # full pool (local + runnable cloud)
    ../.venv/bin/python ai_advisor_agent/demo_switch.py --claude  # only Claude subscription tiers
"""
import sys

import router

PROMPTS = [
    "What is 2 + 2?",
    "Write a Python function to reverse a string.",
    "Summarize the benefits of caching in one sentence.",
    "Refactor this large distributed authentication module for testability.",
    "Prove there are infinitely many prime numbers, step by step.",
]


def main():
    only = "claude-cli" if "--claude" in sys.argv else None
    scope = "cloud" if "-c" in sys.argv else ("local" if "-l" in sys.argv else None)
    print(f"Routing {len(PROMPTS)} prompts "
          f"(only_provider={only}, scope={scope}); prefer runnable models.\n")
    print(f"{'PROMPT':52}  {'→ MODEL':26}  {'TIER·CAP':18}  PROVIDER")
    print("-" * 108)
    for p in PROMPTS:
        d = router.decide(p, only_provider=only, scope=scope, prefer_runnable=True)
        c, tax = d["chosen"], d["taxonomy"]
        tiercap = f"{c['tier']}·{c['capability']}"
        note = f"   [refined: {tax['_refined']}]" if tax.get("_refined") else ""
        print(f"{p[:52]:52}  {c['name']:26}  {tiercap:18}  {c['provider']}{note}")


if __name__ == "__main__":
    main()
