"""
models.providers.gemini — Google Gemini direct API.

Handles model names starting with "gemini" (e.g. gemini-3.1-pro-preview,
gemini-robotics-er-1.6-preview).
Requires: GEMINI_API_KEY environment variable.
"""

import io
from typing import List
import numpy as np
from PIL import Image

from models.retry import retry_api_call

try:
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


def request(
    images: List[np.ndarray],
    prompt: str,
    system_prompt: str,
    model_name: str = "gemini-3.1-pro-preview",
    temp: float = 0.0,
    n_tokens: int = 8192,
    disable_thinking: bool = False,
    extra_messages: List[dict] = None,
) -> str:
    """Send a request to the Google Gemini API.

    disable_thinking: sets thinking_budget=0 (Gemini 2.5+/3.x only).
    extra_messages: unified-format messages inserted before the new query.
    """
    if not _GEMINI_AVAILABLE:
        raise RuntimeError(
            "google-genai package not installed. Run: pip install google-genai"
        )
    client = genai.Client(http_options={"api_version": "v1alpha"})

    # Robotics-ER rejects system_instruction — fold into user text instead.
    is_robotics_er = "robotics-er" in (model_name or "").lower()
    user_text = prompt
    if is_robotics_er and system_prompt and system_prompt.strip():
        user_text = f"{system_prompt}\n\n{prompt}"

    contents = []
    if extra_messages:
        contents.extend(_convert_extra_messages(extra_messages))

    parts = []
    for img_arr in images:
        pil_img = Image.fromarray(img_arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type="image/jpeg", data=buf.getvalue()
                ),
            )
        )
    parts.append(genai_types.Part(text=user_text))
    contents.append(genai_types.Content(role="user", parts=parts))

    cfg_kwargs = dict(
        temperature=temp,
        max_output_tokens=n_tokens,
    )
    if not is_robotics_er:
        cfg_kwargs["system_instruction"] = system_prompt
    if disable_thinking:
        try:
            cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=0)
        except Exception:
            pass

    def _do_request() -> str:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=genai_types.GenerateContentConfig(**cfg_kwargs),
        )
        if response.text is not None:
            return response.text
        text_parts = []
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
        return "".join(text_parts)

    return retry_api_call(_do_request, label=f"Gemini:{model_name}")


def _convert_extra_messages(extra_messages: List[dict]) -> list:
    """Convert unified-format messages to Gemini Content objects."""
    contents = []
    for msg in extra_messages:
        parts = []
        for img in (msg.get("images") or []):
            pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            parts.append(
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        mime_type="image/jpeg", data=buf.getvalue()
                    ),
                )
            )
        text = msg.get("text", "")
        if text:
            parts.append(genai_types.Part(text=text))
        role = "user" if msg["role"] == "user" else "model"
        contents.append(genai_types.Content(role=role, parts=parts))
    return contents
