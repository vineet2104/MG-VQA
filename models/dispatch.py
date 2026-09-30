"""
models.dispatch — Unified VLM dispatcher. Routes to the correct provider.
"""

from typing import List
import numpy as np

from models.parsing import parse_vlm_response

VLM_MODEL_NAME = "azure/openai/gpt-5.2-codex"


def _get_provider(model_name: str):
    """Return the provider module's request() function for a model name.

    Routing rules (checked in order):
        "local/*"                   → models.providers.local
        "gemini*"                   → models.providers.gemini
        "azure/*|gcp/*|aws/*|nvidia/*|openai/*" → models.providers.nvidia
        "grok-*"                    → models.providers.xai
        "qwen/*|google/gemma-*|deepseek/*"     → models.providers.openrouter
        anything else               → models.providers.openai_compat
    """
    if model_name.startswith("anthropic/"):
        from models.providers.anthropic import request
        return request
    elif model_name.startswith("local/"):
        from models.providers.local import request
        return request
    elif model_name.startswith("gemini"):
        from models.providers.gemini import request
        return request
    elif any(model_name.startswith(p) for p in
             ("azure/", "gcp/", "aws/", "nvidia/", "openai/", "xai/")):
        from models.providers.nvidia import request
        return request
    elif model_name.startswith("grok-"):
        from models.providers.xai import request
        return request
    elif (model_name.startswith("qwen/")
          or model_name.startswith("google/gemma-")
          or model_name.startswith("deepseek/")):
        from models.providers.openrouter import request
        return request
    else:
        from models.providers.openai_compat import request
        return request


def call_model(
    images: List[np.ndarray],
    prompt: str,
    system_prompt: str,
    model_name: str,
    temp: float = 0.0,
    n_tokens: int = 8192,
    disable_thinking: bool = False,
    extra_messages: List[dict] = None,
) -> str:
    """Route to the appropriate backend based on model name prefix.

    Returns the raw response text string.
    """
    provider_fn = _get_provider(model_name)
    return provider_fn(
        images=images,
        prompt=prompt,
        system_prompt=system_prompt,
        model_name=model_name,
        temp=temp,
        n_tokens=n_tokens,
        disable_thinking=disable_thinking,
        extra_messages=extra_messages,
    )


def call_vlm(
    images: List[np.ndarray],
    user_prompt: str,
    system_prompt: str,
    model_name: str = VLM_MODEL_NAME,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    extra_messages: List[dict] = None,
) -> dict:
    """Call a VLM and parse its JSON response in one step.

    Returns a dict with at least {"tool": "..."}.
    """
    response_text = call_model(
        images=images, prompt=user_prompt, system_prompt=system_prompt,
        model_name=model_name, temp=temperature, n_tokens=max_tokens,
        extra_messages=extra_messages,
    )
    return parse_vlm_response(response_text)
