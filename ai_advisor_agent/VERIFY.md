# Verify the tool works — use cases

Two ways to check: an **automated one-command verifier**, and a **manual use-case matrix**
for the live behaviors. Run everything from the repo root; `PY=../.venv/bin/python`.

## 1. One command (recommended first step)

```bash
# fast, deterministic, no network — <5s
../.venv/bin/python ai_advisor_agent/verify.py

# also drive real model calls (Ollama + Claude CLI) — slower
../.venv/bin/python ai_advisor_agent/verify.py --live
```
It checks: unit tests (advisor + router), provider discovery, MCP tool wiring, and every
routing rule (privacy guard, complexity boost, escalation, tier/capability routing). It prints
`N passed, M failed` and exits non-zero on any failure — so it's safe to wire into CI.

**A green run means the logic is correct.** The manual cases below let you *see* it decide on
real prompts.

---

## 2. Router CLI — live use cases

Run each with `--dry-run` first (fast: classify + decide, no generation). Drop `--dry-run` to
actually run the answer. Format: `../.venv/bin/python ai_advisor_agent/router.py [flags] "PROMPT"`.

| # | Command | What to expect | Verifies |
|---|---------|----------------|----------|
| 1 | `--dry-run "What is 2+2?"` | Routed to a **Small local** (ollama) model | simple → cheapest/free |
| 2 | `--dry-run "Fix the bug in src/pool.py"` | Routed to a **Coding** model | capability match |
| 3 | `--dry-run "Refactor this large module for testability"` | complexity **boosted** (signal `refactor`) → largest available | hard-signal boost + tier routing |
| 4 | `-l --dry-run "refactor this large auth module"` | routed to a **local** (ollama) model only | `-l`/`--local` scope |
| 5 | `-c --dry-run "refactor this large auth module"` | routed to a **cloud/subscription** model only | `-c`/`--cloud` scope |
| 6 | `--claude --dry-run "What is the capital of France?"` | Routed to **haiku** (claude-cli) | subscription, simple tier |
| 7 | `--claude --dry-run "Prove sqrt(2) is irrational, step by step"` | Routed to **opus** (claude-cli) | subscription, hard tier |
| 8 | `--claude "Summarize these 3 lines: …"` | Actually runs on haiku via your subscription | end-to-end subscription call |
| 9 | `--brain llama3.2 --dry-run "…"` | `router brain: llama3.2` line | classifier override |
| 10 | `-q "what is 2+2"` | one `→ model …` line + a short answer only | concise/quiet output |
| 11 | `-c -q "summarize AI in one line"` | cloud model + short answer, no listing/taxonomy | `-c` + `-q` combined |

**In every case check:** the `router brain:` line names a **local (ollama)** model — deciding is
free — and the `→ Routed to` line's tier/capability/provider make sense for the prompt.

> The classifier is a tiny local model, so it can mislabel a task. That's a model-quality limit,
> not a routing bug — the routing logic itself is proven by `verify.py`. Sharpen it with
> `--brain <stronger model>`.

---

## 3. MCP tools — use cases

In a Claude Code session (with `model-router` connected — check `/mcp`), ask naturally:

| Ask | Tool it calls | Expect |
|-----|---------------|--------|
| "Use recommend_model to pick a model for this refactor." | `recommend_model` | JSON with `recommended_model`, `taxonomy`, `why` — nothing executed |
| "Delegate 'reformat this JSON' with run_with_best_model." | `run_with_best_model` | routes to a **free local** model, `executed: true`, an `answer` |
| "Route this hard task with run_with_best_model." | `run_with_best_model` | picks a paid model → `executed: false`, `reason: … allow_paid=true` |
| "…again with allow_paid true." | `run_with_best_model` | now `executed: true` with the paid model's answer |
| "Recommend among only my Claude tiers." | `recommend_model(only_provider="claude-cli")` | haiku/sonnet/opus by complexity |
| "Answer this locally only." | `run_with_best_model(scope="local")` | picks an Ollama model |
| "Answer this on a cloud model." | `run_with_best_model(scope="cloud", allow_paid=True)` | picks + runs a cloud/subscription model |
| "Answer this concisely." | `run_with_best_model(concise=True)` | returns just `{model, provider, answer}`, brief answer |

Verify the same shape directly (no REPL) — this is exactly what the tool runs:
```bash
cd ai_advisor_agent && PYTHONPATH=. ../.venv/bin/python -c "
import json, mcp_server
print(json.dumps(mcp_server.recommend_model('Fix this race condition'), indent=2))
print(json.dumps(mcp_server.run_with_best_model('Reply with just the number: 2+2'), indent=2))
"
```
Expect: the first recommends a model (advisory); the second has `executed: true` on a local model.
A paid pick without `allow_paid=true` returns `executed: false` with a clear `reason` — that's the
no-surprise-spend guard working.

---

## 3b. Scope + concise — quick test scope

Deterministic (no LLM) — proves the pool filtering:
```bash
cd ai_advisor_agent && ../.venv/bin/python -c "
import router, providers
full = providers.discover_all()
loc = router._select_pool(full, scope='local')
cld = router._select_pool(full, scope='cloud')
assert loc and all(m['provider']=='ollama' for m in loc), 'local scope broken'
assert cld and all(m['provider']!='ollama' for m in cld), 'cloud scope broken'
print('local:', sorted({m[\"provider\"] for m in loc}))
print('cloud:', sorted({m[\"provider\"] for m in cld}))
print('OK')
"
```
Live (one classify each):
```bash
../.venv/bin/python ai_advisor_agent/router.py -l -q "reverse a string in python"   # → an ollama model + short answer
../.venv/bin/python ai_advisor_agent/router.py -c -q "reverse a string in python"   # → a cloud model + short answer
```
Expect the `-l` line's provider to be `ollama` and the `-c` line's provider to be non-local; both print
only the routed model and a brief answer (no listing/taxonomy).

## 4. What "working" looks like — the guarantees to confirm

- **Free to decide:** the classifier brain is always a local model; routing spends nothing.
- **Right-sizing:** trivial → smallest/cheapest; hard → largest available (local or, if allowed, cloud/subscription).
- **Capability match:** coding→coder, reasoning→reasoning, with graceful fallback to General.
- **No tiny-specialist trap:** a Large task escalates past a small specialist to a bigger model when one exists.
- **Privacy:** `Private` prompts never leave the machine.
- **Reasoning-aware:** reasoning prompts (`prove`, `theorem`, `weigh tradeoffs`, …) are flagged `model_type=Reasoning` and reach a reasoning-grade model (e.g. opus), even if the tiny classifier called them Medium.
- **Priority-aware:** the `priority` axis shifts the tier — `Quality` picks a bigger model, `Cost`/`Speed` a smaller one. Priority is inferred from prompt wording ("best quality, cost no object" → Quality; "cheapest that works" → Cost), and the shift itself is proven deterministically by `verify.py` (`_target_tier`).
- **Scope control:** `-l`/`--local` restricts routing to local models, `-c`/`--cloud` to cloud/subscription models (MCP tools take `scope="local"|"cloud"`). The `security` axis is *not* used to force local anymore — use `-l` when you want that.
- **No surprise spend:** paid/subscription models run only when explicitly allowed (`--claude`, `allow_paid=true`).

If `verify.py` is green and a couple of the live cases above behave as described, the tool is working.
