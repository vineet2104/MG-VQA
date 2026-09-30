"""
models.providers.nvidia — NVIDIA inference API (OpenAI-compatible gateway).

Handles model names with prefixes: azure/, gcp/, aws/, nvidia/, openai/
Requires: NV_API_KEY environment variable.
"""

import os
from typing import List
import numpy as np

from models.utils import encode_image_to_base64
from models.retry import RateLimitError, retry_api_call
from models.catalog import gateway_protocol
from models.protocols import build_payload, response_text

NVIDIA_API_BASE_URL = "https://inference-api.nvidia.com/v1"


def request(
    images: List[np.ndarray],
    prompt: str,
    system_prompt: str,
    model_name: str,
    temp: float = 0.0,
    n_tokens: int = 8192,
    disable_thinking: bool = False,
    extra_messages: List[dict] = None,
) -> str:
    """Send a request to the NVIDIA inference API."""
    import requests as _requests

    api_key = os.environ.get("NV_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set NV_API_KEY (or NVIDIA_API_KEY).")

    protocol = gateway_protocol(model_name)
    base_url = os.environ.get("NVIDIA_API_BASE_URL", NVIDIA_API_BASE_URL).rstrip("/")

    content = []
    for img in images:
        b64 = encode_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    # Few-shot exemplar messages (unified format → OpenAI format)
    fewshot_msgs: List[dict] = []
    if extra_messages:
        fewshot_msgs = _convert_extra_messages(extra_messages)

    # Temperature handling per model family
    model_lower = model_name.lower()
    temperature_unsupported = (
        # Astra rejects temperature even when it is the default zero.
        "gpt-6-astra" in model_lower
        or "opus-4" in model_lower or "opus4" in model_lower
        or "sonnet-4" in model_lower or "haiku-4" in model_lower
        or "claude-4" in model_lower
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            *fewshot_msgs,
            {"role": "user", "content": content},
        ],
        "max_tokens": n_tokens,
    }
    if not temperature_unsupported:
        actual_temp = temp
        if "gemini" in model_lower:
            actual_temp = max(temp, 0.1)
        elif "claude" in model_lower and temp == 0:
            actual_temp = 0.01
        payload["temperature"] = actual_temp

    # Disable hidden reasoning tokens on reasoning-capable models
    if disable_thinking:
        if "claude" in model_lower:
            payload["thinking"] = {"type": "disabled"}
        if "gemini" in model_lower:
            payload["thinking_config"] = {"thinking_budget": 0}
            payload["thinkingBudget"] = 0
        if any(k in model_lower for k in ("o1", "o3", "o4", "gpt-5", "gpt5")):
            payload["reasoning_effort"] = "minimal"
        if "qwen" not in model_lower:
            payload["reasoning"] = {"effort": "minimal"}

    if protocol != "chat":
        # Native protocols use different content blocks and token limits.
        payload = build_payload(protocol, model_name, images, prompt,
                                system_prompt, n_tokens, extra_messages)
        if disable_thinking and protocol == "messages":
            payload["thinking"] = {"type": "disabled"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if protocol == "messages":
        headers["anthropic-version"] = "2023-06-01"
    endpoint = {"chat": "chat/completions", "responses": "responses",
                "messages": "messages"}[protocol]

    def _do_request() -> str:
        resp = _requests.post(
            f"{base_url}/{endpoint}",
            headers=headers, json=payload, timeout=180,
        )
        if resp.status_code != 200:
            err_msg = (
                f"NVIDIA API error {resp.status_code} for model '{model_name}': "
                f"{resp.text[:500]}"
            )
            if resp.status_code == 429:
                raise RateLimitError(err_msg)
            raise RuntimeError(err_msg)
        data = resp.json()
        try:
            text = response_text(protocol, data)
            if not text:
                raise RuntimeError(f"Empty {protocol} output for {model_name}")
            return text
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected response structure: {str(data)[:500]}")

    return retry_api_call(_do_request, label=f"NVIDIA:{model_name}")


def _convert_extra_messages(extra_messages: List[dict]) -> List[dict]:
    """Convert unified-format messages to OpenAI chat format."""
    msgs = []
    for msg in extra_messages:
        content_parts = []
        for img in (msg.get("images") or []):
            b64 = encode_image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        text = msg.get("text", "")
        if text:
            content_parts.append({"type": "text", "text": text})
        msgs.append({"role": msg["role"], "content": content_parts})
    return msgs
