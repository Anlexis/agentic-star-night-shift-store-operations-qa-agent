"""AgentCore Platform v1.0"""

# RET-C2-668 - runtime configuration loader.
#
# config/agent.yaml is the static manifest: identity, entry point, trust level
# and the compile-time `requires` gates. It holds NO runtime values. Everything
# tunable at runtime - max_retry, timeout_s, and the `retrieval` / `llm`
# blocks - lives in config/config.yaml and is read from here.
#
# One loader, so there is exactly one answer to "where does this value come
# from". A reader pointed at the manifest instead would not fail; it would find
# no such key, fall back to its own default, and every declared value in
# config/config.yaml would be dead while the tests stayed green.

from pathlib import Path
from typing import Any, Dict

# src/services/runtime_config.py -> parents[2] = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"

# Last-resort values, used only when config/config.yaml cannot be read at all
# (an exotic deployment layout). They mirror the shipped file, so an unreadable
# config degrades to the documented behaviour rather than to an empty mapping.
FALLBACK_RETRIEVAL: Dict[str, Any] = {
    "top_k": 5,
    "score_threshold": 0.5,
    "kb_path": "config/kb/cvs_night_ops_kb.json",
}
FALLBACK_LLM: Dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 2500,
}
FALLBACK_AGENT: Dict[str, Any] = {
    "max_retry": 3,
    "timeout_s": 30,
}


def load_runtime_config() -> Dict[str, Any]:
    """Read config/config.yaml. Returns {} only if it is genuinely unreadable."""
    try:
        import yaml

        loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _block(config: Dict[str, Any], key: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    value = config.get(key)
    if isinstance(value, dict) and value:
        return dict(value)
    return dict(fallback)


def retrieval_config() -> Dict[str, Any]:
    """The declared `retrieval` block - never an empty mapping."""
    return _block(load_runtime_config(), "retrieval", FALLBACK_RETRIEVAL)


def llm_config() -> Dict[str, Any]:
    """The declared `llm` block - never an empty mapping."""
    return _block(load_runtime_config(), "llm", FALLBACK_LLM)


def agent_config() -> Dict[str, Any]:
    """The graph-level runtime block (max_retry / timeout_s) plus the tuning blocks.

    This is what the entry point passes as `Graph(config=...)`, so the values
    declared in config/config.yaml actually reach the framework backbone
    instead of it falling back to its own built-in defaults.
    """
    config = load_runtime_config()
    merged: Dict[str, Any] = dict(FALLBACK_AGENT)
    for key, value in config.items():
        if key in ("retrieval", "llm"):
            continue
        merged[key] = value
    merged["retrieval"] = _block(config, "retrieval", FALLBACK_RETRIEVAL)
    merged["llm"] = _block(config, "llm", FALLBACK_LLM)
    return merged
