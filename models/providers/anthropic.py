"""Direct Anthropic Messages API (anthropic/<model>, not the NVIDIA gateway)."""
import os
import requests
from models.protocols import build_payload, response_text
from models.retry import RateLimitError, retry_api_call


def request(images, prompt, system_prompt, model_name, temp=0.0,
            n_tokens=8192, disable_thinking=False, extra_messages=None):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY for anthropic/ models.")
    payload = build_payload("messages", model_name.removeprefix("anthropic/"),
                            images, prompt, system_prompt, n_tokens, extra_messages)
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}

    def send():
        resp = requests.post("https://api.anthropic.com/v1/messages", json=payload,
                             headers={"x-api-key": key,
                                      "anthropic-version": "2023-06-01"}, timeout=180)
        if resp.status_code == 429:
            raise RateLimitError("Anthropic rate limit")
        resp.raise_for_status()
        text = response_text("messages", resp.json())
        if not text:
            raise RuntimeError("Empty Anthropic text output")
        return text
    return retry_api_call(send, label=f"Anthropic:{model_name}")
