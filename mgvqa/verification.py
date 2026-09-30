"""
core.verification — Answer verification and forced-answer logic.

Extracted from demo_conversational.py for reuse.
"""

from typing import List, Optional
import numpy as np

from models import call_model, parse_vlm_response
from mgvqa.prompts import load_prompt


# ---------------------------------------------------------------------------
# Forced answer selection (when budget/perception-loop exhausted)
# ---------------------------------------------------------------------------

def pick_forced_answer(q_desc: dict, action_history: list) -> str:
    """Pick a default letter when the agent hasn't committed to an answer.

    Heuristic priority:
      1. If the latest `belief` field in action_history already names one
         of the option letters, use that.
      2. For MCQ with "None of the above", pick NOTA (safest hedge).
      3. For yes/no questions, prefer "no" (safer default).
      4. Fallback to "A".
    """
    options = q_desc.get("answer_options", []) or []
    option_map = q_desc.get("option_map", {}) or {}
    letters = [chr(ord("A") + i) for i in range(len(options))] if options else ["A"]

    # 1. Last-belief scan
    for entry in reversed(action_history):
        belief = (entry.get("belief") or "").upper()
        if not belief:
            continue
        for L in letters:
            if f'"{L}"' in belief or f"'{L}'" in belief \
                    or f" {L} " in f" {belief} " or f" {L}." in belief:
                return L
        break  # only inspect the most recent turn with a belief

    # 2. NOTA (counting / interactive_counting / under_object)
    for i, opt in enumerate(options):
        if "none of the above" in str(opt).strip().lower():
            return letters[i]

    # 3. yes/no — prefer "no"
    for L, text in option_map.items():
        if str(text).strip().lower() == "no":
            return L

    # 4. fallback
    return letters[0] if letters else "A"


# ---------------------------------------------------------------------------
# Optional VLM-based answer verification
# ---------------------------------------------------------------------------

def verify_answer(
    rgb: np.ndarray,
    question: str,
    target_name: str,
    proposed_answer: str,
    model_name: str,
    target_display_name: str = None,
) -> dict:
    """Ask a separate VLM call to verify the proposed answer.

    Returns a parsed dict from the verification VLM response.
    """
    prompt_template = load_prompt("answer_verification_prompt.txt")
    prompt = prompt_template.format(
        question=question,
        target_object=target_display_name or (
            target_name.replace("_", " ") if target_name else "the scene"
        ),
        proposed_answer=proposed_answer,
    )

    response_text = call_model(
        images=[rgb],
        prompt=prompt,
        system_prompt="You are a careful visual verification agent.",
        model_name=model_name,
        temp=0.0,
        n_tokens=4096,
    )

    return parse_vlm_response(response_text)
