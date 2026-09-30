"""
models — VLM API routing, response parsing, and provider registry.
===================================================================

Public API:
    call_model(images, prompt, system_prompt, model_name, ...) → str
    call_vlm(images, prompt, system_prompt, model_name, ...) → dict
    parse_vlm_response(response_text) → dict
    VLM_MODEL_NAME — default model name constant
    RateLimitError — raised on HTTP 429

Adding a new model provider:
    See models.md
"""

from models.dispatch import (
    call_model,
    call_vlm,
    VLM_MODEL_NAME,
)
from models.parsing import parse_vlm_response
from models.retry import RateLimitError
from models.singlepass_augment import augment_singlepass_system_prompt

__all__ = [
    "call_model",
    "call_vlm",
    "parse_vlm_response",
    "VLM_MODEL_NAME",
    "RateLimitError",
    "augment_singlepass_system_prompt",
]
