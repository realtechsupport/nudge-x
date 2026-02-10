"""
Prompt selection module (auto-discovery).

This package exposes a unified interface:

    from mllm.prompts import system_prompt, questions, multi_shot_examples

The concrete prompt version is selected based on the PROMPT_VERSION
environment variable (validated by mllm.config.validate_env).

To add a new version, simply create a file named ``prompts_v<N>.py`` in this
directory.  It must define at least ``system_prompt`` and ``questions``.
Then set PROMPT_VERSION=v<N> in your .env — no other code changes needed.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Any

from mllm.config.settings import PROMPT_VERSION


# --- Auto-discover all prompts_v*.py modules in this package ---
_VERSION_MODULES: Dict[str, Any] = {}

for _finder, _name, _ispkg in pkgutil.iter_modules(__path__):
    if _name.startswith("prompts_v"):
        _version_key = _name.replace("prompts_", "")   # e.g. "v4", "v7", "v12"
        _VERSION_MODULES[_version_key] = importlib.import_module(f".{_name}", __package__)


# --- Normalize and validate the requested version ---
def _normalize_version(value: str | None) -> str:
    if value is None:
        raise ValueError("PROMPT_VERSION is not set. Please define it in your .env file.")
    v = value.strip().lower()
    # Accept both 'v7' and '7' styles
    if v.startswith("v"):
        v = v[1:]
    return f"v{v}"


_normalized_version = _normalize_version(PROMPT_VERSION)

if _normalized_version not in _VERSION_MODULES:
    valid = ", ".join(sorted(_VERSION_MODULES.keys(), key=lambda k: int(k[1:])))
    raise ValueError(
        f"Unsupported PROMPT_VERSION '{PROMPT_VERSION}'. "
        f"Available versions: {valid}"
    )

_module = _VERSION_MODULES[_normalized_version]

# --- Re-export the selected symbols ---
# Every prompt file must define: system_prompt
# Optional: questions, multi_shot_examples (or multi_shot_examples_V<N>)
system_prompt = _module.system_prompt

questions = getattr(_module, "questions", [])

multi_shot_examples = getattr(
    _module,
    "multi_shot_examples",
    # Fallback for older naming convention (e.g. multi_shot_examples_V7)
    getattr(_module, f"multi_shot_examples_V{_normalized_version[1:]}", ""),
)

