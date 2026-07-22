"""Fast unit tests for advisor.py's pure logic — no Ollama or network needed.

Run:  .venv/bin/python ai_advisor_agent/test_advisor.py
These cover the deterministic helpers (query anchoring, grounding, JSON parsing).
The end-to-end model behavior is exercised separately via test_cases.md.
"""
import advisor


def check(label, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"[{status}] {label}: got {got!r}")
    return got == expected


def test_normalize():
    ok = True
    ok &= check("normalize GPT-4o", advisor._normalize("GPT-4o"), "gpt4o")
    ok &= check("normalize spaced", advisor._normalize("Llama 3.1 70B"), "llama3170b")
    ok &= check("normalize empty", advisor._normalize(""), "")
    return ok


def test_is_grounded():
    corpus = "Top picks include GPT-4o mini and DeepSeek-R1 for coding."
    ok = True
    ok &= check("grounded hyphen-insensitive", advisor.is_grounded("GPT 4o", corpus), True)
    ok &= check("grounded exact", advisor.is_grounded("DeepSeek-R1", corpus), True)
    ok &= check("not grounded", advisor.is_grounded("Claude Opus 4.8", corpus), False)
    ok &= check("empty name not grounded", advisor.is_grounded("", corpus), False)
    return ok


def test_anchor_query():
    ok = True
    # Missing an LLM term -> anchors appended.
    ok &= check(
        "anchors appended",
        advisor._anchor_query("reformat chat messages cheaply"),
        "reformat chat messages cheaply LLM AI language model",
    )
    # Already mentions an LLM term -> left unchanged.
    ok &= check(
        "llm term left alone",
        advisor._anchor_query("best coding LLM 2026"),
        "best coding LLM 2026",
    )
    ok &= check(
        "brand term left alone",
        advisor._anchor_query("cheapest claude model for writing"),
        "cheapest claude model for writing",
    )
    return ok


def test_parse_json_action():
    ok = True
    ok &= check(
        "plain json",
        advisor._parse_json_action('{"action": "search_web", "query": "x"}'),
        {"action": "search_web", "query": "x"},
    )
    ok &= check(
        "fenced json",
        advisor._parse_json_action('```json\n{"action": "final_answer", "models": []}\n```'),
        {"action": "final_answer", "models": []},
    )
    return ok


def test_accumulate_grounded():
    corpus = "GPT-4o mini is cheap. Claude Opus 4.8 is premium. DeepSeek-R1 is open."
    best, seen = [], set()
    models = [
        {"name": "GPT-4o mini"},              # grounded
        {"name": "Awesome-LLMs-List"},        # not grounded -> dropped
        {"name": "GPT 4o mini"},              # duplicate of #1 -> deduped
        {"name": "Claude Opus 4.8"},          # grounded
    ]
    advisor._accumulate_grounded(models, corpus, best, seen)
    names = [m["name"] for m in best]
    return check("accumulate keeps grounded+deduped", names, ["GPT-4o mini", "Claude Opus 4.8"])


def main():
    results = [
        test_normalize(),
        test_is_grounded(),
        test_anchor_query(),
        test_parse_json_action(),
        test_accumulate_grounded(),
    ]
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} test groups passed.")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
