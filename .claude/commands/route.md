---
description: Route this prompt to the best model via the model-router MCP tool and return its answer
---
Use the `model-router` MCP server to answer the user's request below, letting the
router pick the model automatically.

Call the `run_with_best_model` tool with:
- `prompt`: $ARGUMENTS
- `allow_paid`: true
- `concise`: true

Then reply to the user with ONLY:
1. A single header line: `[<model> · <provider>]` from the tool result.
2. The tool result's `answer`, verbatim.

Do NOT answer the request yourself and do NOT add commentary — the whole point is
that the routed model produces the answer, so the model switches automatically per
prompt via MCP.
