# AI Advisor — Test Cases

Two layers of testing:

## 1. Fast unit tests (no Ollama, runs in <1s)
Tests the deterministic helpers — query anchoring, grounding, JSON parsing, dedupe.
```bash
.venv/bin/python ai_advisor_agent/test_advisor.py
```

## 2. End-to-end taxonomy matrix (needs Ollama running; each run is slow, ~1–4 min)
Feed each prompt to the advisor and check that (a) the **taxonomy** is classified as
expected, and (b) the returned models are **real, versioned LLMs** (not apps/tools/products),
annotated with cost. Run one interactively:
```bash
.venv/bin/python ai_advisor_agent/advisor.py
```
or non-interactively (task on line 1, menu choice on line 2):
```bash
printf '%s\n%s\n' "<task prompt>" "1" | .venv/bin/python ai_advisor_agent/advisor.py
```

| # | Task prompt | Expected taxonomy | What good output looks like |
|---|-------------|-------------------|-----------------------------|
| 1 | "Review my Python pull requests for bugs and code quality. I want the best possible results." | Coding · Large · Public · **Quality** · Standard | Top-tier coding models with versions (e.g. Claude Opus 4.8, GPT-4o, DeepSeek-R1) |
| 2 | "Reformat short chat messages into clean JSON, nothing fancy, keep costs low." | General · Small · Public · **Cost** · Standard | Cheap small models (e.g. GPT-4o mini, Gemini Flash, Llama 3.1 8B); cost = `$ cheap`/`Free` |
| 3 | "Review our proprietary internal codebase for security vulnerabilities; data must stay on-prem." | Coding/Reasoning · Large · **Private** · **Reasoning** | **Local / open-weight only** (e.g. Qwen2.5-Coder 32B, DeepSeek-R1); cost = `Free (local)` |
| 4 | "Solve and explain multi-step calculus and linear-algebra problems." | **Math** · Large · Public · Balanced · **Reasoning** | Reasoning/math models (e.g. o3, DeepSeek-R1, QwQ) |
| 5 | "Turn rough bullet notes into polished product documentation." | **Writing** · Medium · Public · Balanced · Standard | Good writing models (e.g. Claude Sonnet 5, GPT-4o) |
| 6 | "Brainstorm and write marketing campaign copy." | **Creative** · Medium · Public · Balanced · Standard | Creative-strong models |
| 7 | "Extract structured data from scanned invoice images." | **Vision** · Medium · Public · Balanced · Standard | Multimodal/vision models (e.g. GPT-4o, Gemini 3 Pro, Llama 3.2 Vision) |
| 8 | "Real-time autocomplete in my editor — must respond in milliseconds." | Coding · Small · Public · **Speed** · Standard | Small/fast low-latency models |
| 9 | "General customer-support chatbot for FAQs, moderate volume, budget-conscious." | General · Medium · Public · **Cost** · Standard | Mid/cheap general models with pricing shown |
| 10 | "Plan a complex multi-service data-migration strategy weighing tradeoffs." | **Reasoning** · Large · Public · Quality · **Reasoning** | Top reasoning models (e.g. o3, Claude Opus 4.8) |

### What to verify in each result
- **Taxonomy line** matches the expected classification (esp. `priority`, `model_type`, and the Math/Writing capabilities).
- **`model_type`** is right: hard math/logic/planning → `Reasoning`; formatting/chat/writing/simple coding → `Standard` (no expensive reasoning model pushed for easy tasks).
- Every recommendation is a **specific versioned model** ("Claude Opus 4.8", not "Claude").
- **No products/IDEs/tools** (no Antigravity, Cursor, Pumble, GitHub repos, compression libs).
- **Private** tasks (#3) return only local/open-weight options.
- **Cost/Small** tasks (#2, #9) surface cheaper models with a populated `cost` field.
- Any name not confirmed in search results is flagged `⚠ not confirmed`.

> Note: because candidates come from live web search, exact names vary run-to-run — that
> is intended dynamic behavior. Verify the *category* and *shape* of the answer, not exact names.
