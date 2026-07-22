---
description: Route this prompt among your Claude subscription tiers (haiku/sonnet/opus) via MCP and return the answer
---
Use the `model-router` MCP server to answer the request below on the Claude
subscription, letting the router auto-pick the tier (haiku → sonnet → opus).

Call the `run_with_best_model` tool with:
- `prompt`: $ARGUMENTS
- `only_provider`: "claude-cli"
- `allow_paid`: true
- `concise`: true

Then reply with ONLY:
1. A single header line: `[<model> · <provider>]` from the tool result.
2. The tool result's `answer`, verbatim.

Do NOT answer the request yourself — the routed Claude tier produces the answer.
