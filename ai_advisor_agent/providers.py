"""Provider layer — lets the router switch models across BOTH local Ollama and
cloud providers (Anthropic / Claude, OpenAI, Groq, ...).

This is the "top layer" that sits above every provider: the router asks this
module "what models can I use?" and "run this prompt on that model", and this
module hides where each model actually lives.

Discovery is AUTO (per the chosen design):
  - Local Ollama is always live and free (GET /api/tags) — no key needed.
  - A cloud provider is discovered LIVE from its API when its key is set
    (e.g. ANTHROPIC_API_KEY); otherwise its models come from a small offline
    registry (model_registry.json), so the router can still route ACROSS
    providers without any keys.

Execution follows the same rule: a cloud model is actually called only when its
key is present — otherwise the router reports the switch decision and skips the
call (staying free/local by default). Every provider is reached through the
OpenAI-compatible chat format, so one client shape serves them all.
"""
import json
import os
import re
from pathlib import Path

import requests
from openai import OpenAI

OLLAMA_HOST = "http://localhost:11434"
REGISTRY_PATH = Path(__file__).with_name("model_registry.json")

# Cloud providers reachable via an OpenAI-compatible endpoint. Adding another is
# just one more entry here plus (optionally) rows in model_registry.json.
CLOUD_PROVIDERS = {
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/",
        "models_url": "https://api.anthropic.com/v1/models",
        "auth": lambda k: {"x-api-key": k, "anthropic-version": "2023-06-01"},
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/",
        "models_url": "https://api.openai.com/v1/models",
        "auth": lambda k: {"Authorization": f"Bearer {k}"},
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1/",
        "models_url": "https://api.groq.com/openai/v1/models",
        "auth": lambda k: {"Authorization": f"Bearer {k}"},
    },
}


def _params_from_name(name):
    """Pull a parameter count like '1.5b' or '70b' out of a model name."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())
    return float(m.group(1)) if m else None


def tag_model(name, size_bytes=0):
    """Infer (capability, tier, params_b) for a model from its name and size.

    Works for both local names ('qwen2.5-coder:1.5b') and cloud names
    ('claude-opus-4-8'). The registry can override these tags for cloud models
    where the name alone isn't a reliable signal.
    """
    low = name.lower()
    if "coder" in low or "code" in low:
        capability = "Coding"
    elif any(t in low for t in ("r1", "qwq", "reason", "think", "-o1", "-o3")):
        capability = "Reasoning"
    elif any(t in low for t in ("vl", "vision", "llava")):
        capability = "Vision"
    else:
        capability = "General"

    params = _params_from_name(name)
    if params is not None:
        tier = "Small" if params < 4 else "Medium" if params < 30 else "Large"
    elif size_bytes:  # local model with no param count in the name → use disk size
        gb = size_bytes / 1e9
        tier = "Small" if gb < 3 else "Medium" if gb < 15 else "Large"
    else:  # cloud model, no size → guess tier from common name hints
        if any(t in low for t in ("opus", "ultra", "-large", "70b", "large")):
            tier = "Large"
        elif any(t in low for t in ("haiku", "mini", "nano", "small", "flash", "lite")):
            tier = "Small"
        else:
            tier = "Medium"
    return capability, tier, params


def _load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _discover_ollama():
    """Local Ollama models — always live, always free."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return []  # Ollama not running — cloud may still be available
    models = []
    for m in resp.json().get("models", []):
        name = m.get("name", "")
        size = m.get("size", 0)
        capability, tier, params = tag_model(name, size)
        models.append({
            "name": name,
            "provider": "ollama",
            "capability": capability,
            "tier": tier,
            "params_b": params,
            "size_gb": round(size / 1e9, 2),
            "cost": "Free (local)",
            "live": True,        # discovered from a running server
            "runnable": True,    # can actually be executed right now
        })
    return models


def _registry_models(provider):
    """Tagged models for a cloud provider taken from the offline registry."""
    out = []
    for entry in _load_registry().get(provider, []):
        name = entry["name"]
        cap, tier, params = tag_model(name)
        out.append({
            "name": name,
            "provider": provider,
            "capability": entry.get("capability", cap),
            "tier": entry.get("tier", tier),
            "params_b": params,
            "size_gb": None,
            "cost": entry.get("cost", "unknown"),
        })
    return out


def _discover_cloud(provider, cfg):
    """A cloud provider's models: LIVE from its API if the key is set, else from
    the offline registry. Registry metadata (capability/tier/cost) is layered on
    even for live results, since a bare model id isn't enough to tag it."""
    key = os.environ.get(cfg["key_env"])
    reg = {m["name"]: m for m in _registry_models(provider)}

    if not key:
        # No key → routable on paper (from the registry) but not runnable now.
        for m in reg.values():
            m["live"] = False
            m["runnable"] = False
        return list(reg.values())

    try:
        resp = requests.get(cfg["models_url"], headers=cfg["auth"](key), timeout=15)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except requests.RequestException:
        # Key present but the call failed — fall back to registry, still runnable.
        for m in reg.values():
            m["live"] = False
            m["runnable"] = True
        return list(reg.values())

    models = []
    for row in rows:
        name = row.get("id") or row.get("name", "")
        if not name:
            continue
        meta = reg.get(name)
        if meta:
            meta["live"] = True
            meta["runnable"] = True
            models.append(meta)
        else:  # live model we have no registry metadata for → tag heuristically
            cap, tier, params = tag_model(name)
            models.append({
                "name": name, "provider": provider, "capability": cap,
                "tier": tier, "params_b": params, "size_gb": None,
                "cost": "unknown", "live": True, "runnable": True,
            })
    return models


def discover_all():
    """Every model the router can consider, across all providers, tagged and
    merged into one pool. Each dict carries 'provider', 'runnable' (can we call
    it now?) and 'live' (discovered from a live API vs the offline registry)."""
    models = _discover_ollama()
    for provider, cfg in CLOUD_PROVIDERS.items():
        models.extend(_discover_cloud(provider, cfg))
    return models


def _client_for(model):
    """Build an OpenAI-compatible client for whichever provider owns this model."""
    provider = model["provider"]
    if provider == "ollama":
        return OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")
    cfg = CLOUD_PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    return OpenAI(base_url=cfg["base_url"], api_key=key)


def run_model(model, prompt, temperature=0.7):
    """Run the prompt on the chosen model. Returns (answer, note).

    Cloud models are only called when their key is set (per the chosen design);
    otherwise the switch decision stands but execution is skipped gracefully.
    """
    if not model.get("runnable", model["provider"] == "ollama"):
        env = CLOUD_PROVIDERS.get(model["provider"], {}).get("key_env", "the API key")
        return None, f"skipped — set {env} to actually call {model['provider']}"
    client = _client_for(model)
    resp = client.chat.completions.create(
        model=model["name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content, "ok"
