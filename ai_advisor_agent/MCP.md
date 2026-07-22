# Model Router as an MCP tool

This exposes the router ([`router.py`](router.py)) to any MCP-capable agent (Claude Code,
Claude Desktop, …) as two callable tools, via [`mcp_server.py`](mcp_server.py).

## What it can (and can't) do

An MCP tool **cannot** reach into the host agent and flip its model selector — the protocol
doesn't let a tool change the host's own model mid-conversation. What it does instead:

| Tool | What it does |
|------|--------------|
| `recommend_model(prompt)` | **Advisory.** Classifies the prompt and returns the single best model + taxonomy + reasoning. Runs nothing. The agent (or you) decides what to do with the answer. |
| `run_with_best_model(prompt)` | **Delegation.** Routes *and runs* the prompt on the chosen model, returning the answer. This is real "switching" by offloading — a big session pushes a trivial sub-task down to a free local model, or a hard one up to opus, in one call. |

**Execution policy** for `run_with_best_model`: free local Ollama models always run. If routing
picks a **paid/subscription** model (Claude CLI or a cloud API), it returns the *decision* but
does **not** execute unless you pass `allow_paid=true` — so there's no surprise spend.

Both tools accept:
- `only_provider` — restrict routing to one provider, e.g. `"claude-cli"` to pick among only your
  Claude subscription tiers (haiku/sonnet/opus).
- `brain` — override the classifier model (default: smallest free local model).

## Register with Claude Code

```bash
claude mcp add model-router -- \
  /home/spectrum/Desktop/study/ai-engineering-resource/.venv/bin/python \
  /home/spectrum/Desktop/study/ai-engineering-resource/ai_advisor_agent/mcp_server.py
```
Then in a Claude Code session, `/mcp` should list `model-router` with the two tools. Ask things like
*"use recommend_model to decide which model should handle this refactor"* or
*"delegate this JSON reformat with run_with_best_model"*.

To remove: `claude mcp remove model-router`. To register for **Claude Desktop**, add the same
command/args under `mcpServers` in its `claude_desktop_config.json`.

## Two ways to host

### A) Local (stdio) — simplest, recommended for one machine
The agent launches the server itself; nothing to keep running. This is the registration shown
above. Debug it by hand with:
```bash
.venv/bin/python ai_advisor_agent/mcp_server.py     # speaks MCP over stdio; Ctrl-C to stop
```

### B) Hosted (HTTP) — a long-running service other machines/agents connect to
Run it as a service that stays up and serves `http://<host>:<port>/mcp`:
```bash
# localhost only
.venv/bin/python ai_advisor_agent/mcp_server.py --http --port 8000
# reachable from your LAN (bind all interfaces)
.venv/bin/python ai_advisor_agent/mcp_server.py --http --host 0.0.0.0 --port 8000
```
Register the hosted server (on any machine that can reach it):
```bash
claude mcp add --transport http model-router http://127.0.0.1:8000/mcp
```

Keep it running in the background with your process manager of choice — e.g. **systemd**:
```ini
# ~/.config/systemd/user/model-router.service
[Unit]
Description=Model Router MCP server
[Service]
ExecStart=/home/spectrum/Desktop/study/ai-engineering-resource/.venv/bin/python \
  /home/spectrum/Desktop/study/ai-engineering-resource/ai_advisor_agent/mcp_server.py --http --port 8000
Restart=on-failure
[Install]
WantedBy=default.target
```
```bash
systemctl --user daemon-reload && systemctl --user enable --now model-router
```
…or just `nohup … --http --port 8000 &`, or a `tmux`/`screen` session, or a Docker container.

> **Exposing beyond localhost:** the server has no built-in auth. If you bind `0.0.0.0` or put it
> on the internet, front it with a reverse proxy (nginx/Caddy) that adds TLS + an auth header, and
> pass that header via `claude mcp add --transport http … --header "Authorization: Bearer <token>"`.
> The router also shells out to your local `claude` CLI and Ollama, so it must run on a machine
> where those are installed and logged in.

## Example results

`run_with_best_model("What is 2+2? Reply with just the number.")` — trivial task → free local model, executed:
```json
{
  "model": "llama3.2:latest", "provider": "ollama", "tier": "Small", "cost": "Free (local)",
  "executed": true, "answer": "4",
  "why": "task classified Small/General → llama3.2:latest [Small·General·Free (local)]"
}
```

`recommend_model("Fix this race condition in my worker pool", only_provider="claude-cli")` — hard
signal boosts complexity → opus (advisory only):
```json
{
  "recommended_model": "opus", "provider": "claude-cli", "tier": "Large", "cost": "Subscription",
  "complexity_adjusted": "race condition",
  "why": "task classified Large/Coding; ... complexity boosted (hard-task signal: 'race condition') → opus [Large·General·Subscription]"
}
```

> Note: the classifier ("router brain") runs on a free local model, so *deciding* which model to use
> costs nothing. Private-flagged prompts are kept on local models and never sent to a cloud/CLI provider.
