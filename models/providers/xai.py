"""
models.providers.xai — xAI Grok API.

Handles model names starting with "grok-" (e.g. grok-4.3).
Requires: XAI_API_KEY environment variable.
"""

import os
from typing import List
import numpy as np

from models.utils import encode_image_to_base64
from models.retry import RateLimitError, retry_api_call

XAI_API_URL = "https://api.x.ai/v1/chat/completions"


def request(
    images: List[np.ndarray],
    prompt: str,
    system_prompt: str,
    model_name: str = "grok-4.3",
    temp: float = 0.0,
    n_tokens: int = 8192,
    disable_thinking: bool = False,
    extra_messages: List[dict] = None,
) -> str:
    """Send a request to xAI's Grok API."""
    import requests as _requests

    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("XAI_API_KEY environment variable not set.")

    content = []
    for img in images:
        b64 = encode_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "high",
            },
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
    }

    def _do_request() -> str:
        resp = _requests.post(
            XAI_API_URL, headers=headers, json=payload, timeout=180,
        )
        if resp.status_code != 200:
            err_msg = (
                f"xAI API error {resp.status_code} for model "
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

    return retry_api_call(_do_request, label=f"xAI:{model_name}")


def _convert_extra_messages(extra_messages: List[dict]) -> List[dict]:
    """Convert unified-format messages to OpenAI chat format."""
    msgs = []
    for msg in extra_messages:
        content_parts = []
        for img in (msg.get("images") or []):
            b64 = encode_image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            })
        text = msg.get("text", "")
        if text:
            content_parts.append({"type": "text", "text": text})
        msgs.append({"role": msg["role"], "content": content_parts})
    return msgs
