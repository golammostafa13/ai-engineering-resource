# Model Router — How to Verify

The router is a **top layer over many providers** — it switches models across local Ollama *and*
cloud providers (Anthropic / Claude, OpenAI, Groq). There are three things worth verifying:
1. **Routing logic** — given a classified task + a fleet of models, does it pick the right one?
2. **Cross-provider discovery** — does it merge local (live) + cloud (live if key, else registry)?
3. **Live behavior** — on the real machine, does discovery + classification + routing work end-to-end?

## Layer 0 — Cross-provider discovery (no keys needed, <1s)

Confirms the router sees BOTH local and cloud models, with the right source per provider:
```bash
../.venv/bin/python -c "import providers; [print(m['provider'], m['name'], m['tier'], m['capability'], m['cost'], 'runnable=' + str(m.get('runnable'))) for m in providers.discover_all()]"
```
Expect: local Ollama models `live=True, runnable=True, cost=Free (local)`; cloud models from
`model_registry.json` with `runnable=False` **until** you export that provider's key
(e.g. `ANTHROPIC_API_KEY`), after which they discover live and become runnable.

## Layer 1 — Fast unit tests (no Ollama, <1s)

These test the pure logic with a *pretend* fleet, so they run instantly and are deterministic.
```bash
.venv/bin/python ai_advisor_agent/test_router.py
```
They cover:
- **Tagging:** `qwen2.5-coder`→Coding, `deepseek-r1`→Reasoning, `llama3.2`→General; `1.5b`→Small, `70b`→Large; no param count → falls back to file size.
- **Routing:** simple→smallest model, complex-coding→largest coder, reasoning-flag→reasoning model, missing capability (Vision)→General fallback, router-brain = smallest general model.

This is the primary proof the routing is correct — because it uses a full mixed fleet (Small+Large, Coding+Reasoning+General) that you may not have installed locally.

## Layer 2 — Live dry-run (fast: classify + route, skip generation)

`--dry-run` classifies the prompt and prints which model it WOULD use, without running the
(slow) answer generation. Use this to check the classifier + routing on real prompts.
```bash
.venv/bin/python ai_advisor_agent/router.py --dry-run "Fix the bug in this Python function"
```
Verify: does the **Taxonomy** line look right, and is the **→ Routed to** model sensible for it?

## Layer 3 — Full live run (classify + route + run the prompt)
```bash
.venv/bin/python ai_advisor_agent/router.py "Write a haiku about the rain"
```

## Live test-case matrix

Run each with `--dry-run` first (fast). Expected routing is described by **tier + capability**,
because the router always picks the best *available* model — the exact name depends on what you
have pulled. With only Small models installed, tier differences collapse to "smallest available".

| # | Prompt | Expected taxonomy | Expected route (tier · capability) |
|---|--------|-------------------|-------------------------------------|
| 1 | "Write a haiku about the rain." | Small · Writing · Standard | Small · General |
| 2 | "Fix the bug in this Python function that leaks a file handle." | Coding · Standard | Coding model (any tier) |
| 3 | "Refactor and architect a large multi-module Python codebase for testability." | Large · Coding · Standard | **Large** · Coding (largest coder available) |
| 4 | "Prove that the square root of 2 is irrational, step by step." | Math · Large · **Reasoning** | Reasoning model if available, else General/largest |
| 5 | "What's the capital of France?" | Small · General · Standard | **Smallest** · General |
| 6 | "Plan a multi-service data migration weighing tradeoffs and risks." | Reasoning · Large · **Reasoning** | Reasoning model if available, else largest General |
| 7 | "Summarize these meeting notes into 3 bullets." | Writing/General · Small–Medium · Standard | Small · General |
| 8 | "Describe what's in this screenshot." | Vision | Vision model if available, else General fallback |

### What to check in each result
- **Taxonomy** line is reasonable (esp. `complexity` and `capability`).
- **Simple prompts (#1, #5, #7)** route to a **Small** model — the core "simple task → small model" behavior.
- **Coding prompts (#2, #3)** route to the **coding** model, not a general one.
- **#3 vs #2:** once you pull both a small and large coder, #3 (Large) should pick the **bigger** coder and #2 the smaller — proving tier routing. (`ollama pull qwen2.5-coder:32b` to see this.)
- **Reasoning prompts (#4, #6)** route to a reasoning model *if one is available* — this may now be a **cloud** flagship (e.g. `claude-opus-4-8` / `o3`) since cloud models are in the pool. Otherwise they fall back gracefully (never crash).
- **Vision (#8)** routes to a vision model if available, else falls back to General.
- **Cross-provider escalation:** with only small local models pulled, hard `Large`/`Reasoning` tasks should switch to a **cloud** model (`provider != ollama`). The classifier ("router brain") always stays local — verify it is never a cloud model. Trivial tasks (#1, #5) must stay on a **Small** local model, never escalate to a premium cloud one.
- **Cloud execution:** without a key, a cloud pick prints `(cloud call skipped — set <KEY> ...)` and does not crash. With the key exported, the same prompt actually calls that provider and returns an answer.
- **Privacy guard:** a prompt classified `security: Private` (e.g. "review our proprietary internal codebase…") must route to a **local** (`ollama`) model even when a cloud flagship would otherwise win — sensitive prompts never leave the machine. Proven deterministically by `test_route_private_stays_local`.

> Note: classification is done by a small local model, so it can occasionally mislabel a task
> (e.g. code review tagged "Writing"). That's a model-quality limitation, not a routing bug —
> the routing logic itself is proven by Layer 1. A bigger router-brain model sharpens this.
