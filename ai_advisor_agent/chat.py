"""Auto-routing chat — the router as the FRONT DOOR.

Every message you type is automatically classified, routed to the best available
model, and answered by THAT model. No manual model picker, no per-message flags —
just type. The model that answers changes automatically with the prompt (trivial →
free local, coding → local coder, hard/reasoning → your Claude subscription).

This is how you get "switch the model automatically" for real: the router owns the
loop, instead of trying (impossibly) to flip a host agent's own model selector.

    ../.venv/bin/python ai_advisor_agent/chat.py            # full pool (local + subscription)
    ../.venv/bin/python ai_advisor_agent/chat.py --claude   # only your Claude tiers (haiku/sonnet/opus)
    ../.venv/bin/python ai_advisor_agent/chat.py -l          # local only (free)
    ../.venv/bin/python ai_advisor_agent/chat.py -c          # cloud/subscription only
    ../.venv/bin/python ai_advisor_agent/chat.py --safe      # never run a paid/subscription model

Type 'exit' or Ctrl-D to quit.
"""
import sys

import providers
import router


def main():
    only = "claude-cli" if "--claude" in sys.argv else None
    scope = "cloud" if "-c" in sys.argv else ("local" if "-l" in sys.argv else None)
    # Interactive tool → allow the subscription/cloud by default; --safe stays local/free.
    allow_paid = "--safe" not in sys.argv

    print("Auto-routing chat — type a prompt and the right model answers. "
          "('exit' to quit)")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break

        d = router.decide(prompt, only_provider=only, scope=scope, prefer_runnable=True)
        chosen, tax = d["chosen"], d["taxonomy"]
        tag = f"{chosen['name']} · {chosen['tier']}·{chosen['capability']} · {chosen['provider']}"
        refined = f" · refined: {tax['_refined']}" if tax.get("_refined") else ""

        if chosen["provider"] != "ollama" and not allow_paid:
            print(f"[→ would use {tag} — rerun without --safe to allow it]")
            continue

        answer, note = providers.run_model(chosen, prompt)
        print(f"\n[{tag}{refined}]")
        print(answer if answer else f"(skipped: {note})")


if __name__ == "__main__":
    main()
