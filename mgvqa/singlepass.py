"""
core.singlepass — Single-pass (no-tools) baseline mode.

Extracted from demo_conversational.py for reuse.
"""

from typing import List, Dict, Optional
import numpy as np

from models import call_model, parse_vlm_response
from mgvqa.utils import SINGLEPASS_SYSTEM_PROMPT
from mgvqa.prompts import build_type_specific_constraint, build_user_prompt
from tools import format_ab_targets_block


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_singlepass_prompt(
    question: str,
    question_type: str,
    answer_options: List[str],
    q_desc: dict,
    ab_info: Optional[Dict[str, Dict]] = None,
    identify_report: Optional[dict] = None,
) -> str:
    """Build the user prompt for single-pass baseline.

    Mirrors the agentic prompt's question + type constraint + options,
    but with no perception/state blocks and a direct-answer instruction.
    """
    if question_type in {"count", "find", "beneath", "compare", "marked_comparison"}:
        return build_user_prompt("direct", question=question,
                                 question_type=question_type,
                                 answer_options=answer_options)

    # Historical non-release task support only.
    type_constraint = build_type_specific_constraint(
        question_type, q_desc, mode="singlepass")
    options_str = ", ".join(answer_options)

    # `count` has no options — it takes a free integer.  Every other V2/V1
    # task is multiple-choice, so it lists options + a "choose one" directive.
    _has_options = bool(answer_options) and question_type != "count"

    lines = [
        f"QUESTION: {question}",
        f"TYPE: {question_type} — {type_constraint}",
    ]
    if _has_options:
        lines.append(f"OPTIONS: {options_str}")
    lines.append("")
    lines.append("You have ONE image of the scene (no tools, no manipulation).")
    if _has_options:
        lines.append("Choose exactly one of the answer options above.")
    else:
        # count: integer answer, no option list.
        lines.append("Give your answer as stated in the TYPE line above.")
    lines.append('Respond with ONLY a JSON object: {"answer": "your_answer", '
                 '"reasoning": "what you see"}')

    # A/B target points (text-only; images are always clean).
    if ab_info and question_type in ("same_attribute", "relative_size"):
        lines.append("")
        lines.append("== A/B TARGETS (normalized [row, col] on a [0, 1000] grid) ==")
        lines.append(format_ab_targets_block(ab_info))

    # Type-specific output fields
    if question_type == "existence":
        lines.append(
            'If your answer corresponds to "yes", also include '
            '"point": [row, col] normalised to a [0, 1000] grid.'
        )
    elif question_type == "relative_size":
        lines.append(
            'Also include "ratio": <number> — the approximate volume ratio '
            "of object A to object B."
        )

    if identify_report is not None:
        from experiments.marked_size.question import identify_stage2_addendum
        lines.append("")
        lines.append(identify_stage2_addendum(identify_report))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-pass execution
# ---------------------------------------------------------------------------

def run_singlepass(
    rgb: np.ndarray,
    user_prompt: str,
    model_name: str,
    verbose: bool = True,
    question_type: str = "",
) -> dict:
    """Send a single image + question to the VLM and parse the answer.

    Parameters
    ----------
    rgb : np.ndarray
        The RGB image (H, W, 3).
    user_prompt : str
        Pre-built prompt from build_singlepass_prompt().
    model_name : str
        VLM model identifier.

    Returns
    -------
    dict with keys: answer, point, ratio, reasoning, raw_response, success
    """
    # One published system prompt across all four tasks and planner models.
    system_prompt = SINGLEPASS_SYSTEM_PROMPT

    raw_response = call_model(
        images=[rgb],
        prompt=user_prompt,
        system_prompt=system_prompt,
        model_name=model_name,
        temp=0.0,
        n_tokens=16384,
    )

    parsed = parse_vlm_response(raw_response)
    answer = parsed.get("answer", "")
    reasoning = parsed.get("reasoning", "")
    point = parsed.get("point", None)
    ratio = parsed.get("ratio", None)

    if parsed.get("tool") == "error":
        if verbose:
            print(f"  ✗ PARSE ERROR: {raw_response[:200]}")
        return {
            "answer": None,
            "point": None,
            "ratio": None,
            "reasoning": "",
            "raw_response": raw_response,
            "success": False,
        }

    if verbose:
        print(f"  VLM answer: {answer}")
        if point is not None:
            print(f"  Point: {point}")
        if ratio is not None:
            print(f"  Ratio: {ratio}")
        print(f"  Reasoning: {reasoning}")

    return {
        "badge_points": parsed.get("badge_points"),
        "answer": str(answer).strip(),
        "point": point,
        "ratio": ratio,
        "reasoning": reasoning,
        "raw_response": raw_response,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Arm 2: two-stage singlepass (identify, then answer)
# ---------------------------------------------------------------------------

def run_singlepass_identify(
    rgb: np.ndarray,
    question: str,
    question_type: str,
    answer_options: List[str],
    q_desc: dict,
    model_name: str,
    verbose: bool = True,
) -> dict:
    """Two VLM calls on the SAME single image, no tools.

    Stage 1 asks only "where are the badges?"; stage 2 asks the question with
    the model's own stage-1 report pasted back in.  This is the control for the
    agentic arm's report_badges gate: it forces identification without granting
    the ability to change the scene.
    """
    from experiments.marked_size.question import (IDENTIFY_STAGE1_PROMPT,
                                                  patch_singlepass_system_prompt)
    sys_prompt = patch_singlepass_system_prompt(SINGLEPASS_SYSTEM_PROMPT)

    raw1 = call_model(images=[rgb], prompt=IDENTIFY_STAGE1_PROMPT,
                      system_prompt=sys_prompt, model_name=model_name,
                      temp=0.0, n_tokens=2048)
    rep = parse_vlm_response(raw1) or {}
    report = {"badge_points": rep.get("badge_points"),
              "notes": (rep.get("notes") or "")[:300]}
    if verbose:
        print(f"  [identify] badge_points={report['badge_points']}")

    prompt2 = build_singlepass_prompt(
        question=question, question_type=question_type,
        answer_options=answer_options, q_desc=q_desc,
        identify_report=report)
    out = run_singlepass(rgb=rgb, user_prompt=prompt2, model_name=model_name,
                         verbose=verbose, question_type=question_type)
    out["badge_points"] = report.get("badge_points")
    out["identify_stage"] = {"prompt": IDENTIFY_STAGE1_PROMPT,
                             "raw": raw1[:1200], "report": report}
    out["stage2_prompt"] = prompt2
    return out
