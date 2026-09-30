"""
models.providers.local — Local model inference (HuggingFace checkpoints).

Handles model names starting with "local/" (e.g. local/my_checkpoint).

Two modes:
  - Server mode (parallel eval): If LOCAL_MODEL_URL is set, requests are
    sent via HTTP to model_server.py. Multiple workers share one GPU.
  - In-process mode (single worker): Model loaded directly on GPU.

Set LOCAL_MODEL_PATH to override the model path.
"""

import os
import io
import time
from typing import List
import numpy as np
from PIL import Image

from models.retry import retry_api_call

_LOCAL_MODEL = None  # singleton: (model, processor, model_path)


def request(
    images: List[np.ndarray],
    prompt: str,
    system_prompt: str,
    model_name: str = "local/mgvqa_agent",
    temp: float = 0.0,
    n_tokens: int = 8192,
    disable_thinking: bool = False,
    extra_messages: List[dict] = None,
) -> str:
    """Run inference on a locally-loaded HuggingFace model."""
    server_url = os.environ.get("LOCAL_MODEL_URL", "")
    if server_url:
        return _request_server(
            images, prompt, system_prompt, model_name, temp, n_tokens,
            server_url, extra_messages=extra_messages)
    return _request_inprocess(
        images, prompt, system_prompt, model_name, temp, n_tokens,
        extra_messages=extra_messages)


def _request_server(
    images: List[np.ndarray],
    prompt: str,
    system_prompt: str,
    model_name: str,
    temp: float,
    n_tokens: int,
    server_url: str,
    extra_messages: List[dict] = None,
) -> str:
    """Send request to the model_server.py HTTP endpoint."""
    import base64
    import requests

    messages = [{"role": "system", "content": system_prompt}]

    if extra_messages:
        for msg in extra_messages:
            content_parts = []
            for img in (msg.get("images") or []):
                pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            content_parts.append({"type": "text", "text": msg.get("text", "")})
            messages.append({"role": msg["role"], "content": content_parts})

    # Current turn
    content_parts = []
    for img in images:
        pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content_parts.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content_parts})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": n_tokens,
        "temperature": temp,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload, timeout=600,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"[local-server] Error attempt {attempt+1}: {e}, "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _request_inprocess(
    images: List[np.ndarray],
    prompt: str,
    system_prompt: str,
    model_name: str,
    temp: float,
    n_tokens: int,
    extra_messages: List[dict] = None,
) -> str:
    """Load model in-process and run inference directly."""
    import torch

    model_path = os.environ.get(
        "LOCAL_MODEL_PATH", model_name.removeprefix("local/"))
    model, processor = _get_local_model(model_path)

    # Build multi-turn messages in Qwen VL chat format
    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

    if extra_messages:
        for msg in extra_messages:
            content = []
            for img in (msg.get("images") or []):
                pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                content.append({"type": "image", "image": pil_img})
            content.append({"type": "text", "text": msg.get("text", "")})
            messages.append({"role": msg["role"], "content": content})

    # Current user turn
    user_content = []
    for img in images:
        pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        user_content.append({"type": "image", "image": pil_img})
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})

    # Gather all PIL images for the processor
    all_images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "image" and "image" in part:
                    all_images.append(part["image"])

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=all_images if all_images else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=(temp > 0),
            temperature=max(temp, 0.01) if temp > 0 else None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True)[0]

    return output_text


def _get_local_model(model_path: str):
    """Lazy-load a local Qwen3-VL model. Cached as module-level singleton."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None and _LOCAL_MODEL[2] == model_path:
        return _LOCAL_MODEL[0], _LOCAL_MODEL[1]

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[local] Loading model from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[local] Model loaded on {model.device}")
    _LOCAL_MODEL = (model, processor, model_path)
    return model, processor
