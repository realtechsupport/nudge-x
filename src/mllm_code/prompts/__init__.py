"""
Prompt selection module.

This package exposes a unified interface:

    from mllm_code.prompts import system_prompt, questions, multi_shot_examples

The concrete prompt version is selected based on the PROMPT_VERSION
environment variable (validated by mllm_code.config.validate_env).
"""

from __future__ import annotations

from typing import Dict, Any

from mllm_code.config.settings import PROMPT_VERSION
from . import prompts_v4, prompts_v5, prompts_v6, prompts_v7


_VERSION_MODULES: Dict[str, Any] = {
    "v4": prompts_v4,
    "v5": prompts_v5,
    "v6": prompts_v6,
    "v7": prompts_v7,
}


def _normalize_version(value: str | None) -> str:
    if value is None:
        # validate_env() should already have enforced presence; this is a safeguard.
        raise ValueError("PROMPT_VERSION is not set. Please define it in your .env file.")
    v = value.strip().lower()
    # Accept both 'v7' and '7' styles
    if v.startswith("v"):
        v = v[1:]
    return f"v{v}"


_normalized_version = _normalize_version(PROMPT_VERSION)

if _normalized_version not in _VERSION_MODULES:
    valid = ", ".join(sorted(_VERSION_MODULES.keys()))
    raise ValueError(
        f"Unsupported PROMPT_VERSION '{PROMPT_VERSION}'. "
        f"Expected one of: {valid}"
    )

_module = _VERSION_MODULES[_normalized_version]

# Re-export the selected symbols so callers can keep using:
# from mllm_code.prompts import system_prompt, questions, multi_shot_examples
system_prompt = _module.system_prompt
questions = _module.questions

# Some prompt versions may expose either multi_shot_examples or a variant.
multi_shot_examples = getattr(
    _module,
    "multi_shot_examples",
    getattr(_module, "multi_shot_examples_V7", ""),
)

