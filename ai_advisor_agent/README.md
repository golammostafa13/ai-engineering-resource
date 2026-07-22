# AI Model Router & Advisor

A local-first tool that **picks the right AI model for each prompt automatically** and
switches to it — across **local Ollama**, your **Claude Pro/Max subscription** (via the
`claude` CLI), and keyed **cloud APIs** (Anthropic / OpenAI / Groq). Trivial tasks go to a
small free local model; hard/reasoning tasks go to a flagship — decided by a free local
classifier, so *choosing* a model never costs anything.

> Why this exists: Claude Code (and the API) have **no native per-prompt model routing by
> task complexity** — only manual `/model` or the fixed `opusplan` split. This tool fills
> that gap and can be driven from the CLI, a chat REPL, or as **MCP tools** an agent calls.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r ai_advisor_agent/requirements.txt
# then, for the two runnable backends you want:
ollama pull llama3.2          # local, free  (https://ollama.com)
claude    # log in once, to enable your Claude subscription tiers (haiku/sonnet/opus)
```
Cloud APIs are optional — set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GROQ_API_KEY` to
add them; otherwise they're routable-on-paper from `model_registry.json`.

## The pieces

| File | What it is |
|------|-----------|
| `router.py` | The router: classify a prompt → route to the best model → run it. CLI entry. |
| `providers.py` | Provider layer — discovers/tags models across Ollama + subscription + cloud. |
| `advisor.py` | Advisor: searches the web for current best models for a task, you pick. |
| `mcp_server.py` | Exposes the router as **MCP tools** (`recommend_model`, `run_with_best_model`). |
| `chat.py` | Auto-routing chat REPL — type a prompt, the right model answers. |
| `demo_switch.py` | Prints how several prompts route — see the model switch per prompt. |
| `verify.py` | One-command verification (fast + `--live`). |
| `model_registry.json` | Editable metadata/fallback pool for cloud + subscription models. |

## Use it

```bash
PY=../.venv/bin/python   # run from repo root, or use ./ from ai_advisor_agent/

# 1) CLI — route + run one prompt
$PY ai_advisor_agent/router.py "refactor this large auth module"
$PY ai_advisor_agent/router.py --dry-run "…"     # decide only, no generation
$PY ai_advisor_agent/router.py -q "what is 2+2"  # concise output
$PY ai_advisor_agent/router.py --claude "…"      # only your Claude tiers
$PY ai_advisor_agent/router.py -l "…" | -c "…"   # local-only | cloud-only

# 2) Auto-routing chat — just type, the right model answers automatically
$PY ai_advisor_agent/chat.py --claude

# 3) See the switching
$PY ai_advisor_agent/demo_switch.py
```

### As MCP tools (inside Claude Code / any MCP agent)
```bash
claude mcp add --scope user model-router -- \
  /ABS/PATH/.venv/bin/python /ABS/PATH/ai_advisor_agent/mcp_server.py
```
Then in a session: the tools `recommend_model` (advisory) and `run_with_best_model`
(delegate + run) are available. On-demand switching via the bundled slash commands
`/route` and `/route-claude`. Full setup + hosting (stdio/HTTP): [MCP.md](MCP.md).

> Note: an MCP tool cannot change Claude Code's *own* model selector — that's a protocol
> limit. It switches at the **answer** level by delegating to the routed model.

## How routing works (short)

1. **Classify** the prompt with a free local "brain" (smallest local model) into a taxonomy:
   complexity · capability · priority · model_type.
2. **Refine** with prompt signals (`_refine_taxonomy`) — hard/reasoning words raise complexity
   and flag reasoning models; code/image cues fix capability.
3. **Route** (`route`): capability match → tier by complexity, shifted by priority
   (`Quality`↑, `Cost`/`Speed`↓); escalate past tiny specialists; prefer runnable models.
4. **Run** on the chosen model (or, for cloud/subscription, only when allowed).

## Verify

```bash
$PY ai_advisor_agent/verify.py          # fast, deterministic (<5s)
$PY ai_advisor_agent/verify.py --live   # + real model calls
$PY ai_advisor_agent/test_router.py     # unit tests
$PY ai_advisor_agent/test_mcp_client.py # end-to-end over the real MCP protocol
```
See [VERIFY.md](VERIFY.md) for the full use-case matrix.
