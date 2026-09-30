"""
models.providers.openrouter — OpenRouter API (Qwen3-VL, Gemma 4, DeepSeek, etc.)

Handles model names starting with "qwen/", "google/gemma-", "deepseek/".
Requires: OPENROUTER_API_KEY environment variable.
"""

import os
from typing import List
import numpy as np

from models.utils import encode_image_to_base64
from models.retry import RateLimitError, retry_api_call

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


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
    """Send a request to OpenRouter."""
    import requests as _requests

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set.")

    content = []
    for img in images:
        b64 = encode_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    fewshot_msgs = _convert_extra_messages(extra_messages) if extra_messages else []

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            *fewshot_msgs,
            {"role": "user", "content": content},
        ],
        "max_tokens": n_tokens,
        "temperature": temp,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vineet2104/Agentic_MQA",
        "X-Title": "Agentic-MQA",
    }

    def _do_request() -> str:
        resp = _requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=180,
        )
        if resp.status_code != 200:
            err_msg = (
                f"OpenRouter API error {resp.status_code} for model "
                f"'{model_name}': {resp.text[:500]}"
            )
            if resp.status_code == 429:
                raise RateLimitError(err_msg)
            raise RuntimeError(err_msg)
        data = resp.json()
        try:
            msg = data["choices"][0]["message"]
            text = msg.get("content") or ""
            if not text and msg.get("reasoning_content"):
                text = msg["reasoning_content"]
            return text
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected response: {str(data)[:500]}")

    return retry_api_call(_do_request, label=f"OpenRouter:{model_name}")


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
