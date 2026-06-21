import os
import time
from typing import List, Dict, Optional

import litellm

from cs329a_hw3._llm_cache import cached

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------
# The assignment hard-codes Together AI model ids throughout the notebook and the
# package. We transparently translate those ids to Anthropic / OpenAI / Gemini
# models served via litellm, so no notebook model strings need editing.
#
# Current assignment (everything in the cheap/fast tier; mirrors the spec's
# "small model + tools" intent):
#   * workhorse / generation / 8B-baseline    -> claude-haiku-4-5
#   * 70B-baseline / decomposition / refine   -> claude-haiku-4-5  (was sonnet;
#                                                user request to compare)
#   * LLM judge (evaluate_qa)                 -> gpt-5-nano        (cheapest GPT)
#   * fusion 3rd model (provider diversity)   -> gemini-3.1-flash-lite
MODEL_MAP = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "anthropic/claude-haiku-4-5",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "anthropic/claude-haiku-4-5",
    "google/gemma-3n-E4B-it": "openai/gpt-5-nano",
    "OpenAI/gpt-oss-20B": "gemini/gemini-3.1-flash-lite",
}
_KNOWN_PREFIXES = ("anthropic/", "openai/", "gemini/", "together_ai/", "groq/", "vertex_ai/")
_DEFAULT_MODEL = "anthropic/claude-haiku-4-5"

# Models that don't accept a custom `temperature` (must use the API default).
# gpt-5 family raises UnsupportedParamsError if you pass temperature != 1.
_NO_TEMPERATURE_PREFIXES = ("openai/gpt-5",)


def _supports_temperature(model: str) -> bool:
    return not any(model.startswith(p) for p in _NO_TEMPERATURE_PREFIXES)


def map_model(model: str) -> str:
    """Translate an assignment model id to a litellm-routable model id."""
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    if any(model.startswith(p) for p in _KNOWN_PREFIXES):
        return model
    return _DEFAULT_MODEL


class _Message:
    """Minimal stand-in for an OpenAI/litellm message exposing ``.content``."""

    def __init__(self, content: Optional[str]):
        self.content = content


def generate_together(
    model: str,
    messages: List[Dict],
    temperature: float = 0.7,
    response_format: Optional[Dict] = None,
    max_tokens: Optional[int] = None,
):
    """
    Generate a response via litellm (Anthropic/OpenAI/Gemini), preserving the
    original ``.content`` return contract and adding disk caching + logging.
    """
    mapped = map_model(model)

    # Only OpenAI reliably honours dict-style response_format here; for other
    # providers we drop it and rely on prompt-level JSON instructions instead.
    rf = response_format if (response_format and mapped.startswith("openai/")) else None

    # litellm requires max_tokens for Anthropic; Gemini "thinking" models need a
    # generous budget so the visible answer is not starved.
    mt = max_tokens or 4096

    # gpt-5 family does not accept a custom temperature -> drop it.
    effective_temp = temperature if _supports_temperature(mapped) else None

    payload = {
        "provider": "litellm",
        "model": mapped,
        "messages": messages,
        "temperature": effective_temp,
        "response_format": rf,
        "max_tokens": mt,
    }

    def _call():
        args = {"model": mapped, "messages": messages, "max_tokens": mt}
        if effective_temp is not None:
            args["temperature"] = effective_temp
        if rf:
            args["response_format"] = rf
        for attempt in range(3):
            try:
                response = litellm.completion(**args)
                return response["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                print(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    print("Final attempt failed. Returning None.")
                    return None

    content = cached(payload, _call, tag="hw3")
    # Defensive: always return a Message object so callers that do
    # `response.content.strip()...` don't crash on transient API failure.
    return _Message(content if content is not None else "")
