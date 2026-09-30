"""
tools.py — Perception, manipulation & prompt-formatting tools
================================================================

Provides the 5 VLM tools (perceive_get_obstructors was removed):

  Perception (RoboRefer/Molmo + SAM3):
    - perceive_scene            — full scan; VLM lists objects → point → segment.
                                  Non-object labels (table, floor, …) are auto-
                                  filtered via FILTER_OUT_LABELS before pointing.
    - perceive_find_object      — multi-instance: returns ALL visible instances of
                                  a named object with IoU > 0.7 deduplication.

  Manipulation (GraspGen 6-DOF + push):
    - execute_grasp_and_remove  — 3×6-DOF attempts, angle-first ranking,
                                  90° finger offset, mask dilation, SAM3
                                  re-perception between retries.
    - execute_push              — closed-gripper sweep in an image-relative
                                  direction (left/right/up/down).
    - (answer_question is handled in the demo loop, not here.)

  Prompt formatting (structured state blocks for the per-turn prompt):
    - build_perception_meta     — structured dict from perception results
                                  (labels, centroids on [0,1000] grid, quadrants).
    - format_perception_yaml    — YAML-style text block sent to the VLM.
    - format_scene_state        — world model: goal, known objects, removed
                                  objects, attempts summary, open unknowns.
    - format_agent_notes        — VLM's own plan / belief scratchpad.
    - format_recent_actions     — last N=2 actions, verbatim with reasoning.
    - format_longterm_summary   — compressed tool-usage tally.

  Helpers:
    - Non-object filter: FILTER_OUT_LABELS, _is_non_object(), _filter_non_objects()
    - compute_centroid_rowcol(), compute_quadrant()  — mask geometry on [0,1000] grid
    - draw_ab_markers           — A/B marker overlay for same_attribute / relative_size
    - save_gt_annotated_image   — GT bounding-box debug image (not sent to VLM)
    - format_marker_mapping, format_action_history  — legacy text formatters
    - build_sam_occupancy       — combined binary SAM mask for push ray-tracing
    - re_perceive_object        — SAM3 re-prompt using GT seg centroid

  Model loading:
    - load_molmo_model, load_roborefer_model, load_sam3_model
    - set_pointing_backend      — switch between 'roborefer' (default) and 'molmo'
    - set_finger_offset         — set GraspGen finger rotation offset
"""

import os
import re
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    BitsAndBytesConfig,
)

# Disable ultralytics hub sync/telemetry before importing SAM.
os.environ.setdefault("YOLO_TELEMETRY", "false")
from ultralytics import SAM
from ultralytics.utils import SETTINGS as _ULTRALYTICS_SETTINGS
_ULTRALYTICS_SETTINGS["sync"] = False

from models import call_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOLMO_MODEL_NAME = "allenai/Molmo-7B-D-0924"
MOLMO_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
ROBOREFER_MODEL_NAME = "Zhoues/RoboRefer-8B-SFT"
ROBOREFER_REPO_DIR = os.path.join(os.path.dirname(__file__), "third_party", "roborefer")
SAM3_WEIGHTS = os.path.join(os.path.dirname(__file__), "third_party", "sam3", "sam3.pt")

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts", "vlm_demo")

# RoboRefer prompt suffix — the model is sensitive to this exact phrasing
ROBOREFER_POINT_SUFFIX = (
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], "
    " where each tuple contains the x and y coordinates of a point satisfying "
    "the conditions above. The coordinates should be between 0 and 1, indicating "
    "the normalized pixel locations of the points in the image."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_molmo_model(
    model_name: str = MOLMO_MODEL_NAME,
    cache_dir: str = MOLMO_CACHE_DIR,
    device: Optional[torch.device] = None,
):
    """Load Molmo-7B-D with 4-bit quantisation.  Returns (processor, model)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Molmo model: {model_name}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )
    model.eval()
    print("Skipping torch.compile (incompatible with Molmo's variable-length inputs).")
    model._supports_default_dynamic_cache = lambda: False
    print("Molmo model loaded successfully.")
    return processor, model


def load_sam3_model(weights_path: str = SAM3_WEIGHTS) -> SAM:
    """Load SAM3 model for point-prompted segmentation."""
    print(f"Loading SAM3 model from {weights_path}...")
    sam_model = SAM(weights_path)
    print("SAM3 model loaded successfully.")
    return sam_model


def load_roborefer_model(
    model_name: str = ROBOREFER_MODEL_NAME,
    cache_dir: str = MOLMO_CACHE_DIR,
    device: Optional[torch.device] = None,
):
    """Load RoboRefer-8B-SFT.  Returns ``(None, model)`` — processor slot is
    ``None`` so the caller signature stays compatible with Molmo.

    The RoboRefer repo must be at ``ROBOREFER_REPO_DIR`` (default:
    ``<project>/RoboRefer``).  It is added to ``sys.path`` on first call.
    """
    import sys
    if ROBOREFER_REPO_DIR not in sys.path:
        sys.path.insert(0, ROBOREFER_REPO_DIR)

    # Resolve the snapshot path from the HF cache
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=os.path.join(cache_dir, "hub") if not cache_dir.endswith("hub") else cache_dir,
    )
    print(f"Loading RoboRefer model from: {model_path}")

    import llava as _llava                      # noqa: reimport from RoboRefer
    from llava import conversation as _clib

    model = _llava.load(model_path)
    _clib.default_conversation = _clib.conv_templates["auto"].copy()
    print("RoboRefer model loaded successfully.")
    # Return (None, model) to keep the same 2-tuple interface as load_molmo_model
    return None, model


# ---------------------------------------------------------------------------
# RoboRefer helpers
# ---------------------------------------------------------------------------

def get_roborefer_output(
    image: Image.Image, _processor, model, prompt: str,
) -> str:
    """Run RoboRefer inference on a PIL image.

    ``_processor`` is ignored (kept for API parity with ``get_molmo_output``).
    The prompt is automatically appended with the coordinate-format suffix
    that RoboRefer is trained to expect.
    """
    import sys, tempfile
    if ROBOREFER_REPO_DIR not in sys.path:
        sys.path.insert(0, ROBOREFER_REPO_DIR)
    from llava.media import Image as LlavaImage

    # RoboRefer expects a file path, not a PIL object — write to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        image.save(tmp.name)
        full_prompt = [LlavaImage(tmp.name), prompt + ROBOREFER_POINT_SUFFIX]
        with torch.no_grad(), torch.inference_mode():
            answer = model.generate_content(full_prompt)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        os.unlink(tmp.name)
    return answer


def parse_roborefer_points(
    text: str, image_size: Tuple[int, int],
) -> List[dict]:
    """Parse RoboRefer output ``[(x1, y1), ...]`` into ``[{label, x, y}, ...]``.

    Coordinates are normalised to [0, 1]; we convert to pixel coords.
    """
    width, height = image_size
    detections: List[dict] = []
    # Try to find all (x, y) float pairs in the response
    matches = re.findall(
        r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text,
    )
    for x_str, y_str in matches:
        x_norm, y_norm = float(x_str), float(y_str)
        # RoboRefer outputs normalised [0, 1] coords
        if x_norm <= 1.0 and y_norm <= 1.0:
            px = int(x_norm * width)
            py = int(y_norm * height)
        else:
            # Fallback: already pixel coords
            px, py = int(x_norm), int(y_norm)
        detections.append({"label": "object", "x": px, "y": py})
    return detections


# ---------------------------------------------------------------------------
# Pointing backend dispatch
# ---------------------------------------------------------------------------

# Module-level flag: "molmo", "roborefer", or "vlm".  Set by demo.py at startup.
_pointing_backend: str = "molmo"
# When backend == "vlm", the main VLM (same model as the agent) acts as
# the pointer.  demo.py sets ``_vlm_pointer_model_name`` once at startup.
_vlm_pointer_model_name: str = ""


def set_pointing_backend(backend: str, vlm_model_name: str = "") -> None:
    """Set the active pointing backend.

    Valid values:
        'molmo'     — local Molmo-7B-D
        'roborefer' — local RoboRefer-8B-SFT
        'vlm'       — delegate pointing to a fixed VLM (``vlm_model_name``)

    When ``backend == "vlm"``, the model identified by ``vlm_model_name``
    answers pointing queries.  Historically this was the same model as
    the agent's planner ("planner-as-pointer"), but the canonical
    benchmark now PINS this to ``gcp/google/gemini-3.1-pro-preview``
    so every model under test (Claude, Qwen, GPT-5.x, Gemini-Robotics-
    ER, ...) shares the SAME pointing tool.  This isolates planning
    ability from per-model pointing accuracy and makes cross-model
    comparisons fair.

    Callers in ``demo.py`` / ``demo_conversational.py`` pass
    ``args.pointing_vlm_model`` here, which itself defaults to
    ``gcp/google/gemini-3.1-pro-preview``.  See README § "Pointing
    backend" and the conversational-eval docs.
    """
    global _pointing_backend, _vlm_pointer_model_name
    assert backend in ("molmo", "roborefer", "vlm"), \
        f"Unknown pointing backend: {backend}"
    _pointing_backend = backend
    if backend == "vlm":
        if not vlm_model_name:
            raise ValueError(
                "pointing_model='vlm' requires vlm_model_name to be passed "
                "to set_pointing_backend().")
        _vlm_pointer_model_name = vlm_model_name


def get_pointing_output(
    image: Image.Image, processor, model, prompt: str,
) -> str:
    """Dispatch to the active pointing backend."""
    if _pointing_backend == "roborefer":
        return get_roborefer_output(image, processor, model, prompt)
    if _pointing_backend == "vlm":
        return get_vlm_pointer_output(image, prompt)
    return get_molmo_output(image, processor, model, prompt)


def parse_pointing_output(
    text: str, image_size: Tuple[int, int],
) -> List[dict]:
    """Parse point output from the active pointing backend."""
    if _pointing_backend == "roborefer":
        return parse_roborefer_points(text, image_size)
    if _pointing_backend == "vlm":
        return parse_vlm_pointer_points(text, image_size)
    return parse_molmo_points(text, image_size)


# ---------------------------------------------------------------------------
# Molmo helpers
# ---------------------------------------------------------------------------

def get_molmo_output(image: Image.Image, processor, model, prompt: str) -> str:
    """Run Molmo inference on a PIL image.  Returns generated text."""
    with torch.no_grad(), torch.inference_mode():
        inputs = processor.process(images=[image], text=prompt)
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        inputs = {
            k: v.to(model_device,
                     dtype=model_dtype if v.dtype.is_floating_point else v.dtype
                     ).unsqueeze(0)
            for k, v in inputs.items()
        }
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=512, do_sample=True,
                temperature=0.7, top_p=0.9, top_k=50,
                stop_strings="<|endoftext|>",
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            ),
            tokenizer=processor.tokenizer,
        )
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return generated_text


def parse_molmo_points(text: str, image_size: Tuple[int, int]) -> List[dict]:
    """Parse Molmo point output into [{label, x, y}, ...]."""
    width, height = image_size
    detections: List[dict] = []

    # Pattern 1: XML point tags
    matches = re.findall(
        r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"[^>]*>([^<]*)</point>',
        text, re.IGNORECASE,
    )
    if matches:
        for x_pct, y_pct, label in matches:
            x = int(float(x_pct) / 100.0 * width)
            y = int(float(y_pct) / 100.0 * height)
            if label.strip():
                detections.append({"label": label.strip(), "x": x, "y": y})
        return detections

    # Pattern 2: alt-attribute form
    matches = re.findall(
        r'x="([\d.]+)"\s+y="([\d.]+)"\s+alt="([^"]*)"', text, re.IGNORECASE,
    )
    if matches:
        for x_pct, y_pct, label in matches:
            detections.append({
                "label": label.strip() or "object",
                "x": int(float(x_pct) / 100.0 * width),
                "y": int(float(y_pct) / 100.0 * height),
            })
        return detections

    # Pattern 3: "label: (x, y)"
    matches = re.findall(
        r'([A-Za-z_\s]+?)(?::\s*|at\s*)\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)', text,
    )
    if matches:
        for label, x_str, y_str in matches:
            x, y = float(x_str), float(y_str)
            if x <= 100 and y <= 100:
                x, y = int(x / 100.0 * width), int(y / 100.0 * height)
            else:
                x, y = int(x), int(y)
            if label.strip():
                detections.append({"label": label.strip(), "x": x, "y": y})
        return detections

    return detections


# ---------------------------------------------------------------------------
# VLM-as-pointer backend (FIXES_FOR_April18 P1-2)
# ---------------------------------------------------------------------------

VLM_POINTER_SYSTEM = (
    "You are a visual localisation assistant.  When given an image and the "
    "name of a target object, decide whether the object is clearly present "
    "and, if so, return its centre point as normalized coordinates on a "
    "0–1000 grid.  Row 0 is the top, row 1000 is the bottom; col 0 is the "
    "left edge, col 1000 is the right edge.\n\n"
    "Output STRICT JSON on a single line — one of:\n"
    '  {"found": true, "row": <int>, "col": <int>, "confidence": "high"|"medium"|"low"}\n'
    "  if the object is clearly present (use confidence to express certainty), OR\n"
    '  {"found": false, "reason": "<what you actually see at the closest candidate location>"}\n'
    "  if the object is NOT present, only something resembling it is visible, "
    "or you are unsure.\n\n"
    "It is better to return found=false with an honest reason than to point "
    "at the nearest plausible object that is not actually the target.  "
    "Do not wrap your answer in markdown, code fences, or any other text."
)


def get_vlm_pointer_output(image: Image.Image, prompt: str) -> str:
    """Ask the main VLM to point to a named object.

    ``prompt`` is the Molmo/RoboRefer-style phrasing, e.g.
    ``"Point to the blue phillips box."``  We re-phrase it to request the
    strict JSON format documented in ``VLM_POINTER_SYSTEM`` and forward
    the image + prompt to the main ``call_model`` router.
    """
    if not _vlm_pointer_model_name:
        raise RuntimeError(
            "VLM pointer backend is active but _vlm_pointer_model_name is "
            "unset.  Call set_pointing_backend('vlm', vlm_model_name=...) "
            "first.")

    # Strip the stock "Point to the " prefix if present so the user prompt
    # reads as a direct query.
    label_query = prompt
    for _p in ("Point to the ", "Point to "):
        if label_query.startswith(_p):
            label_query = label_query[len(_p):]
            break
    label_query = label_query.rstrip(".").strip()

    user_prompt = (
        f"Target object: {label_query}.\n"
        f"Look carefully at the image.  Is this object clearly present?\n"
        f'If YES: {{"found": true, "row": <int 0-1000>, "col": <int 0-1000>, '
        f'"confidence": "high"|"medium"|"low"}}\n'
        f'If NO or UNSURE: {{"found": false, "reason": "<what you see instead>"}}'
    )

    if isinstance(image, Image.Image):
        rgb = np.asarray(image)
    else:
        rgb = image  # already numpy array

    try:
        # Pointer calls need reasoning enabled so the model can evaluate
        # whether the target is truly present (not just the nearest
        # visual match).  With disable_thinking=True the model was
        # forced to emit coordinates without the internal sanity check,
        # leading to false positives on absent objects (see analysis:
        # ep 026, ep 029).  n_tokens=4096 gives headroom for thinking.
        response = call_model(
            images=[rgb],
            prompt=user_prompt,
            system_prompt=VLM_POINTER_SYSTEM,
            model_name=_vlm_pointer_model_name,
            temp=0.0, n_tokens=4096,
            disable_thinking=False,
        )
    except Exception as e:
        # Propagate as empty-string so callers treat it as "no point".
        print(f"[VLM-pointer] call_model failed: {e}")
        return ""
    return response or ""


def parse_vlm_pointer_points(
    text: str, image_size: Tuple[int, int],
) -> List[dict]:
    """Parse a ``{"row": R, "col": C}`` JSON blob into ``[{label, x, y}]``.

    ``row`` / ``col`` are in a 0–1000 grid; we rescale to pixel coords.
    Returns an empty list when the point is (-1, -1) or the JSON cannot
    be parsed.
    """
    if not text:
        return []
    width, height = image_size

    # Pull the first balanced JSON object out of the response — the VLM
    # occasionally wraps it in backticks or adds stray prose.
    import json as _json
    import re as _re

    def _truncation_note(raw: str) -> str:
        # Short diagnostic to print when the JSON is unterminated: the
        # VLM likely ran out of output-token budget, not a bad prompt.
        return (f"[VLM-pointer] unterminated JSON — response was "
                f"{len(raw)} chars, ends with: "
                f"{raw[-80:]!r}")

    obj = None
    start = text.find("{")
    if start >= 0:
        depth = 0
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end >= 0:
            try:
                obj = _json.loads(text[start:end])
            except Exception:
                obj = None

    # Fallback #1: unterminated JSON (e.g. ``{"row": 450, "col":`` —
    # happens when the output-token budget is too tight for reasoning
    # models).  Scan the raw text for the two integers directly.
    # Fallback #2: plain "row=450, col=300"-style output.
    if obj is None:
        # Match ``"row": 450`` / ``row = 450`` / ``row 450`` etc.
        row_match = _re.search(
            r"(?i)\brow\b\s*[=:\-]?\s*(-?\d{1,4})", text)
        col_match = _re.search(
            r"(?i)\bcol(?:umn)?\b\s*[=:\-]?\s*(-?\d{1,4})", text)
        if row_match and col_match:
            try:
                row = int(row_match.group(1))
                col = int(col_match.group(1))
                obj = {"row": row, "col": col}
                print(f"[VLM-pointer] recovered from malformed response "
                      f"via regex: row={row}, col={col}")
            except (TypeError, ValueError):
                obj = None
        if obj is None:
            print(_truncation_note(text))
            return []

    # ── Handle found: false (abstention) ──
    found_flag = obj.get("found")
    if found_flag is not None and not found_flag:
        reason = obj.get("reason", "unknown")
        print(f"[VLM-pointer] NOT FOUND — reason: {reason}")
        return []

    row = obj.get("row")
    col = obj.get("col")
    try:
        row = int(row)
        col = int(col)
    except (TypeError, ValueError):
        print(f"[VLM-pointer] JSON parsed but row/col non-integer: {obj!r}")
        return []

    # Sentinel for "not visible" (legacy format).
    if row < 0 or col < 0:
        return []

    # Extract confidence (default to "medium" for backwards compat)
    confidence = obj.get("confidence", "medium")

    # Clamp into [0, 1000] then rescale to pixels.
    row = max(0, min(1000, row))
    col = max(0, min(1000, col))
    y = int(round(row / 1000.0 * max(height - 1, 1)))
    x = int(round(col / 1000.0 * max(width - 1, 1)))
    return [{"label": "", "x": x, "y": y, "confidence": confidence}]


# ---------------------------------------------------------------------------
# Perception pipelines
# ---------------------------------------------------------------------------

VLM_IDENTIFY_SYSTEM = (
    "You are a visual perception assistant.  When given a top-down image of "
    "objects on a table, identify every separate physical object and provide "
    "its approximate centre point on a normalised 0–1000 grid (row 0 = top, "
    "row 1000 = bottom; col 0 = left, col 1000 = right).\n\n"
    "IMPORTANT RULES:\n"
    "- List every separate physical object as an INDIVIDUAL INSTANCE.\n"
    "- If you see multiple visually identical objects (e.g. 4 identical white "
    "boxes), list EACH ONE separately with its own name and centre point — "
    "do NOT collapse duplicates.\n"
    "- Use the most specific name you can read from visible labels/branding. "
    "If no label is legible, use a short generic descriptive name.\n"
    "- Do NOT include the table surface, floor, shadows, or background.\n"
    '- For each object, rate your confidence: "high" if you can clearly see '
    'and identify it, "low" if it is partially occluded or you are unsure '
    'what it is, "medium" otherwise.\n\n'
    "Return STRICT JSON — an array of objects, one per physical item:\n"
    '[{"name": "short name", "row": <int>, "col": <int>, "confidence": "high"|"medium"|"low"}, ...]\n'
    "No markdown, code fences, or any other text outside the JSON array."
)

VLM_OBSTRUCT_SYSTEM = (

    "You are a visual perception assistant helping a robot arm uncover a hidden object. "
    "When given an image and told which object is hidden, list only the objects that are "
    "most likely physically blocking or on top of the hidden object. "
    "Return only a comma-separated list of short lowercase object names. "
    "Do not include explanations or any other text."
)


def _get_object_labels_from_vlm(
    rgb: np.ndarray, model_name: str, verbose: bool = True,
) -> List[str]:
    """Ask the VLM to enumerate visible objects.  Returns a list of names.

    .. deprecated::
        Use :func:`_get_object_labels_and_points_from_vlm` for the
        combined call that returns both labels and points.  This legacy
        function is kept for backwards compat with non-VLM pointing
        backends.
    """
    response = call_model(
        images=[rgb],
        prompt=(
            "List every separate physical object you can see in this image "
            "as individual instances.  If there are multiple identical "
            "objects, list each one separately.  Return only a "
            "comma-separated list of short lowercase names."
        ),
        system_prompt=(
            "You are a visual perception assistant. "
            "When given an image, return only a comma-separated list of "
            "every separate physical object you can see, as individual "
            "instances (do not collapse duplicates). Use short, lowercase "
            "names. Do not include the table, floor, or background."
        ),
        model_name=model_name, temp=0.0, n_tokens=16384,
        disable_thinking=False,
    )
    if verbose:
        print(f"[VLM-identify] Raw response: {response.strip()}")
    return [l.strip() for l in response.split(",") if l.strip()]


def _get_object_labels_and_points_from_vlm(
    rgb: np.ndarray, model_name: str, verbose: bool = True,
) -> List[dict]:
    """Combined call: VLM returns object names AND centre points together.

    Returns a list of ``{"name": str, "row": int, "col": int}`` dicts
    (normalised 0–1000 grid).  Non-object labels are filtered out.

    This eliminates the separate per-label pointing call in perceive_scene
    when ``pointing_model=vlm``, so labels and points are naturally tied
    and instance enumeration is reliable.
    """
    import json as _json

    response = call_model(
        images=[rgb],
        prompt=(
            "List every separate physical object on the table with its "
            "centre point.  If you see multiple identical objects, list "
            "each one separately.  Use the most specific name you can "
            "read from visible labels."
        ),
        system_prompt=VLM_IDENTIFY_SYSTEM,
        model_name=model_name, temp=0.0, n_tokens=16384,
        disable_thinking=False,
    )
    if verbose:
        print(f"[VLM-identify-combined] Raw response ({len(response)} chars): "
              f"{response}")

    # Parse JSON array from response
    items = []
    # Find the first '[' and the matching ']'
    start = response.find("[")
    if start >= 0:
        depth = 0
        end = -1
        for i in range(start, len(response)):
            if response[i] == "[":
                depth += 1
            elif response[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end >= 0:
            try:
                items = _json.loads(response[start:end])
            except Exception as e:
                if verbose:
                    print(f"[VLM-identify-combined] JSON parse error: {e}")
                items = []

        # ── Truncation recovery ──
        # If the closing ']' was never found, the response was truncated
        # mid-array (thinking models spend tokens on hidden reasoning).
        # Salvage all complete {...} entries from the partial array.
        if end < 0 and start >= 0:
            fragment = response[start:]
            if verbose:
                print(f"[VLM-identify-combined] Truncated array — "
                      f"attempting recovery from {len(fragment)} chars")
            # Extract all balanced {...} objects from the fragment
            recovered = []
            i = 0
            while i < len(fragment):
                if fragment[i] == "{":
                    obj_depth = 0
                    obj_end = -1
                    for j in range(i, len(fragment)):
                        if fragment[j] == "{":
                            obj_depth += 1
                        elif fragment[j] == "}":
                            obj_depth -= 1
                            if obj_depth == 0:
                                obj_end = j + 1
                                break
                    if obj_end > 0:
                        try:
                            obj = _json.loads(fragment[i:obj_end])
                            recovered.append(obj)
                        except Exception:
                            pass
                        i = obj_end
                    else:
                        break  # incomplete object — stop
                else:
                    i += 1
            if recovered:
                items = recovered
                if verbose:
                    print(f"[VLM-identify-combined] Recovered {len(recovered)} "
                          f"complete entries from truncated response")

    if not items and verbose:
        print("[VLM-identify-combined] Could not parse JSON array from response")

    # Validate entries and filter non-objects
    valid = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name or _is_non_object(name):
            continue
        try:
            row = int(entry["row"])
            col = int(entry["col"])
        except (KeyError, TypeError, ValueError):
            continue
        if row < 0 or col < 0:
            continue
        conf = str(entry.get("confidence", "medium")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "medium"
        valid.append({"name": name, "row": row, "col": col, "confidence": conf})

    if verbose:
        print(f"[VLM-identify-combined] {len(valid)} objects after filtering "
              f"(from {len(items)} raw entries)")
    return valid


def _sam_segment(sam_model, rgb, px, py, verbose=True, label=""):
    """Run SAM3 at a single point.  Returns mask or None."""
    W, H = rgb.shape[1], rgb.shape[0]
    px, py = max(0, min(px, W - 1)), max(0, min(py, H - 1))
    try:
        results = sam_model.predict(source=rgb, points=[[px, py]], labels=[1], verbose=False)
        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            return results[0].masks.data[0].bool().cpu().numpy()
    except Exception as e:
        if verbose:
            print(f"[SAM3] Error for '{label}' at ({px}, {py}): {e}")
    return None


def perceive_scene(
    rgb: np.ndarray,
    molmo_processor, molmo_model, sam_model: SAM,
    vlm_model_name: str = "", verbose: bool = True,
) -> Tuple[np.ndarray, Dict[int, str], List[dict]]:
    """Full scene perception: VLM lists objects → point → SAM segments.

    When ``pointing_model=vlm``, uses a single combined VLM call that
    returns both object names and centre points — labels and coordinates
    are naturally tied (no separate per-label pointing).  For Molmo /
    RoboRefer backends, falls back to the legacy two-step flow.

    Non-object labels (table, floor, etc.) are filtered out automatically.
    """
    H, W = rgb.shape[:2]
    pil_image = Image.fromarray(rgb)

    detections = []

    if _pointing_backend == "vlm":
        # ── Combined call: labels + points + confidence in one VLM query ──
        items = _get_object_labels_and_points_from_vlm(
            rgb, model_name=vlm_model_name, verbose=verbose,
        )
        for item in items:
            row = max(0, min(1000, item["row"]))
            col = max(0, min(1000, item["col"]))
            y = int(round(row / 1000.0 * max(H - 1, 1)))
            x = int(round(col / 1000.0 * max(W - 1, 1)))
            detections.append({
                "label": item["name"],
                "x": x, "y": y,
                "confidence": item.get("confidence", "medium"),
            })
    else:
        # ── Legacy two-step: VLM labels → per-label pointing ──
        labels = _get_object_labels_from_vlm(rgb, model_name=vlm_model_name, verbose=verbose)
        if verbose:
            print(f"[VLM-identify] Found {len(labels)} objects: {labels}")
        if not labels:
            return rgb.copy(), {}, []

        # Filter non-object labels before pointing
        labels = [l for l in labels if not _is_non_object(l)]
        if verbose and len(labels) == 0:
            print("[VLM-identify] All labels filtered as non-objects.")

        backend_tag = _pointing_backend.capitalize()
        for label in labels:
            point_text = get_pointing_output(pil_image, molmo_processor, molmo_model, f"Point to the {label}.")
            if verbose:
                print(f"[{backend_tag}] '{label}' → {point_text[:200]}")
            pts = parse_pointing_output(point_text, image_size=(W, H))
            if pts:
                det = pts[0]
                det["label"] = label
                detections.append(det)
            elif verbose:
                print(f"[{backend_tag}] Could not find point for '{label}', skipping.")

    if not detections:
        return rgb.copy(), {}, []

    valid = []
    masks = []
    for det in detections:
        mask = _sam_segment(sam_model, rgb, det["x"], det["y"], verbose, det["label"])
        if mask is None:
            continue

        # Deduplicate: if IoU > 0.7 with an existing mask, keep the one
        # with the larger area (more reliable segment).  See §3.
        dup_idx = None
        mask_area = int(mask.sum())
        for j, existing_mask in enumerate(masks):
            inter = np.logical_and(mask, existing_mask).sum()
            union = np.logical_or(mask, existing_mask).sum()
            if union > 0 and inter / union > 0.7:
                dup_idx = j
                break
        if dup_idx is not None:
            existing_area = int(masks[dup_idx].sum())
            if mask_area > existing_area:
                # Replace with the larger mask (better segment)
                masks[dup_idx] = mask
                valid[dup_idx]["mask"] = mask
                valid[dup_idx]["label"] = det["label"]
            # Either way, skip adding a new entry
            continue

        det["mask"] = mask
        masks.append(mask)
        valid.append(det)

    if not valid:
        return rgb.copy(), {}, []

    marker_map = {i: d["label"] for i, d in enumerate(valid, 1)}
    labelled = _create_labelled_overlay(rgb, masks, list(marker_map.keys()), marker_map)
    return labelled, marker_map, valid


def perceive_find_object(
    rgb: np.ndarray, target_name: str,
    molmo_processor, molmo_model, sam_model: SAM,
    verbose: bool = True, target_display_name: str = None,
) -> Tuple[np.ndarray, Dict[int, str], List[dict], str]:
    """Locate ALL visible instances of a named object.

    Returns (labelled_img, marker_map, detections, status) where marker_map
    is {1: display, 2: display, ...} — one entry per detected instance.
    Status is "found" if >=1 instance was detected, else "not_visible".
    Masks with IoU > 0.7 are deduplicated.
    """
    if not target_name and not target_display_name:
        return rgb.copy(), {}, [], "no_target"
    H, W = rgb.shape[:2]
    pil_image = Image.fromarray(rgb)
    display = target_display_name or target_name.replace("_", " ")

    point_text = get_pointing_output(pil_image, molmo_processor, molmo_model, f"Point to the {display}.")
    if verbose:
        print(f"[FindObject] {_pointing_backend} for '{display}': {point_text[:300]}")
    pts = parse_pointing_output(point_text, image_size=(W, H))
    if not pts:
        return rgb.copy(), {}, [], "not_visible"

    # Segment ALL returned points (multi-instance)
    valid = []
    masks = []
    for pt in pts:
        pt["label"] = display
        mask = _sam_segment(sam_model, rgb, pt["x"], pt["y"], verbose, display)
        if mask is None:
            continue
        # Deduplicate: skip if IoU > 0.7 with any existing mask
        dup = False
        for existing_mask in masks:
            inter = np.logical_and(mask, existing_mask).sum()
            union = np.logical_or(mask, existing_mask).sum()
            if union > 0 and inter / union > 0.7:
                dup = True
                break
        if dup:
            continue
        pt["mask"] = mask
        masks.append(mask)
        valid.append(pt)

    if not valid:
        return rgb.copy(), {}, [], "not_visible"

    marker_map = {i: display for i in range(1, len(valid) + 1)}
    labelled = _create_labelled_overlay(rgb, masks, list(marker_map.keys()), marker_map)
    return labelled, marker_map, valid, "found"


from perception.grounding import perceive_find_object_at_point


def perceive_get_obstructors(
    rgb: np.ndarray, target_name: str, vlm_model_name: str,
    molmo_processor, molmo_model, sam_model: SAM,
    verbose: bool = True, target_display_name: str = None,
) -> Tuple[np.ndarray, Dict[int, str], List[dict]]:
    """Identify and segment objects blocking the target."""
    if not target_name and not target_display_name:
        return rgb.copy(), {}, []
    H, W = rgb.shape[:2]
    pil_image = Image.fromarray(rgb)
    display = target_display_name or target_name.replace("_", " ")

    prompt = (
        f"The target object '{display}' is NOT directly visible in this scene — "
        f"it is buried under or surrounded by other objects. "
        f"List the objects that are most likely on top of or directly blocking the "
        f"'{display}'. These are the objects the robot should remove first."
    )
    response = call_model(
        images=[rgb], prompt=prompt, system_prompt=VLM_OBSTRUCT_SYSTEM,
        model_name=vlm_model_name, temp=0.0, n_tokens=4096,
        disable_thinking=False,  # reasoning on — better obstruction analysis
    )
    if verbose:
        print(f"[GetObstructors] VLM identified blockers: {response.strip()}")
    labels = [l.strip() for l in response.split(",") if l.strip()]
    if not labels:
        return rgb.copy(), {}, []

    detections = []
    backend_tag = _pointing_backend.capitalize()
    for label in labels:
        point_text = get_pointing_output(pil_image, molmo_processor, molmo_model, f"Point to the {label}.")
        pts = parse_pointing_output(point_text, image_size=(W, H))
        if pts:
            det = pts[0]
            det["label"] = label
            detections.append(det)
        elif verbose:
            print(f"[GetObstructors] {backend_tag} could not point to '{label}', skipping.")
    if not detections:
        return rgb.copy(), {}, []

    valid = []
    masks = []
    for det in detections:
        mask = _sam_segment(sam_model, rgb, det["x"], det["y"], verbose, det["label"])
        if mask is not None:
            det["mask"] = mask
            masks.append(mask)
            valid.append(det)
    if not valid:
        return rgb.copy(), {}, []

    marker_map = {i: d["label"] for i, d in enumerate(valid, 1)}
    labelled = _create_labelled_overlay(rgb, masks, list(marker_map.keys()), marker_map)
    return labelled, marker_map, valid


# ---------------------------------------------------------------------------
# Overlay / marker drawing
# ---------------------------------------------------------------------------


def draw_pointing_debug(
    rgb: np.ndarray,
    detections: List[dict],
    marker_map: Dict[int, str] = None,
) -> np.ndarray:
    """Draw pointing-model output on a copy of the RGB image.

    Each detection's (x, y) pixel coordinate is drawn as a coloured circle
    with its marker ID and label.  Useful for debugging what RoboRefer/Molmo
    actually pointed to before SAM segmentation.

    Returns a new image (does not modify *rgb* in place).
    """
    img = np.ascontiguousarray(rgb.copy())
    H, W = img.shape[:2]
    radius = max(5, min(H, W) // 80)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(H, W) / 1200)
    thickness = 1

    # Colour palette (cycle if > 10 detections)
    palette = [
        (255, 0, 0), (0, 200, 0), (0, 80, 255), (255, 200, 0),
        (200, 0, 200), (0, 200, 200), (255, 100, 0), (100, 255, 0),
        (0, 100, 255), (200, 200, 0),
    ]

    for i, det in enumerate(detections):
        px, py = int(det["x"]), int(det["y"])
        colour = palette[i % len(palette)]
        mid = i + 1  # marker IDs are 1-based
        label = det.get("label", "")
        if marker_map:
            label = marker_map.get(mid, label)

        # Filled + outline circle
        cv2.circle(img, (px, py), radius, colour, -1)
        cv2.circle(img, (px, py), radius, (0, 0, 0), 1)

        # Label text
        text = f"{mid}: {label}" if label else str(mid)
        tx, ty = px + radius + 3, py + 4
        # Background rectangle for readability
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 3),
                      (0, 0, 0), -1)
        cv2.putText(img, text, (tx, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    return img


def _create_labelled_overlay(
    rgb: np.ndarray, masks: List[np.ndarray],
    labels: List[int], marker_map: Dict[int, str],
) -> np.ndarray:
    """Create a numbered segmentation overlay on the RGB image.

    The mask is rendered with ~60% opacity (clearly tinted) and a thick
    contour (4 px) so viewers can see exactly which pixels belong to each
    object.  Labels are drawn at the mask centroid with a dark pill
    background so they remain legible against any background colour.
    """
    labelled = rgb.copy().astype(np.float32)
    np.random.seed(42)
    # Use saturated colours (hue-spread rather than pure random) so each mask
    # is visually distinct.  We generate a palette in HSV and convert to BGR.
    hsv_palette = np.zeros((max(len(masks), 1), 1, 3), dtype=np.uint8)
    for i in range(len(masks)):
        hsv_palette[i, 0] = [int(180 * i / max(len(masks), 1)), 220, 230]
    bgr_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    # Input rgb is in RGB; swap to match
    palette = bgr_palette[:, ::-1]  # BGR → RGB for blending with `rgb`

    # Alpha-blend each mask separately so overlapping regions still show both
    # colours blended with the underlying image.
    ALPHA = 0.55
    for i, mask in enumerate(masks):
        colour = np.array(palette[i], dtype=np.float32)
        m3 = mask[..., None].astype(np.float32) * ALPHA
        labelled = labelled * (1.0 - m3) + colour * m3

    labelled = np.clip(labelled, 0, 255).astype(np.uint8)

    # Thick contours in a slightly darker shade for crisp boundaries.
    for i, mask in enumerate(masks):
        colour = tuple(int(c * 0.6) for c in palette[i])   # darker outline
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(labelled, contours, -1, colour, 4, cv2.LINE_AA)

    # Labels on top (drawn last so they are never occluded by contours).
    for i, mask in enumerate(masks):
        M = cv2.moments(mask.astype(np.uint8) * 255)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
        text = str(labels[i])
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), bl = cv2.getTextSize(text, font, 0.9, 2)
        pad = 6
        # Dark pill background
        cv2.rectangle(labelled,
                      (cx - tw // 2 - pad, cy - th // 2 - pad),
                      (cx + tw // 2 + pad, cy + th // 2 + pad + bl),
                      (0, 0, 0), -1)
        # Light border for contrast
        cv2.rectangle(labelled,
                      (cx - tw // 2 - pad, cy - th // 2 - pad),
                      (cx + tw // 2 + pad, cy + th // 2 + pad + bl),
                      (255, 255, 255), 1)
        cv2.putText(labelled, text, (cx - tw // 2, cy + th // 2),
                    font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return labelled


def compute_ab_marker_info(
    seg: np.ndarray,
    marker_obj_ids: Dict[str, int],
    marker_full_area: Dict[str, int],
    visibility_threshold: float = 0.30,
) -> Dict[str, Dict]:
    """Compute visibility + normalized [0, 1000] centroid for A/B target objects.

    The image is kept clean — we do NOT draw markers on it.  The resulting
    info is provided to the VLM as a text block (see ``format_ab_targets_block``).

    Called fresh every turn of the agentic loop, so a point that was
    blank when the object was occluded automatically reappears once
    obstructors are cleared and the visible fraction crosses the
    ``visibility_threshold`` again.

    Args:
        seg:                PyBullet segmentation buffer (same resolution as RGB).
        marker_obj_ids:     ``{"A": obj_id, "B": obj_id}``.
        marker_full_area:   ``{"A": full_pixels, "B": full_pixels}`` — the
                            unoccluded mask area for each object, captured at
                            spawn time.
        visibility_threshold: Fraction of ``full_area`` that must be visible
                            for the object to be considered identifiable.

    Returns:
        ``{"A": {"visible": bool, "point": [row, col] or None,
                "visibility": float (0–1)},
           "B": ...}``
        ``point`` is normalized to a [0, 1000] grid (row=0 top, col=0 left),
        or ``None`` if the object is below the visibility threshold.
        ``visibility`` is retained in the returned dict for programmatic
        callers (logs, tests) but is deliberately NOT exposed to the VLM
        by ``format_ab_targets_block``.
    """
    H, W = seg.shape[:2]
    out: Dict[str, Dict] = {}
    for letter in ("A", "B"):
        oid = marker_obj_ids.get(letter) if marker_obj_ids else None
        full_area = (marker_full_area or {}).get(letter, 0)
        if oid is None or full_area <= 0:
            out[letter] = {"visible": False, "point": None, "visibility": 0.0}
            continue
        vis_mask = (seg == oid)
        vis_px = int(vis_mask.sum())
        ratio = vis_px / float(full_area)
        if ratio < visibility_threshold or vis_px == 0:
            out[letter] = {"visible": False, "point": None, "visibility": float(ratio)}
            continue
        ys, xs = np.where(vis_mask)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        # Snap centroid to the nearest visible pixel — the mean of a
        # non-convex visible region (e.g. L-shaped) can fall in the
        # occluded gap, which would give the VLM a point on the
        # occluder rather than on the target object.
        cy_px, cx_px = int(round(cy)), int(round(cx))
        if (cy_px < 0 or cy_px >= H or cx_px < 0 or cx_px >= W
                or not vis_mask[cy_px, cx_px]):
            # Mean centroid is NOT on the visible mask — find the
            # closest visible pixel via squared-distance.
            dists = (ys - cy) ** 2 + (xs - cx) ** 2
            nearest_idx = np.argmin(dists)
            cy, cx = float(ys[nearest_idx]), float(xs[nearest_idx])
        row = int(round(cy / max(H - 1, 1) * 1000))
        col = int(round(cx / max(W - 1, 1) * 1000))
        row = max(0, min(1000, row))
        col = max(0, min(1000, col))
        out[letter] = {"visible": True, "point": [row, col],
                        "visibility": float(ratio)}
    return out


def format_ab_targets_block(ab_info: Dict[str, Dict]) -> str:
    """Render A/B target info as a text block for the user prompt.

    Each object gets a single line with its normalized ``[row, col]``
    centroid on the [0, 1000] grid.  If the object is below the
    visibility threshold, its point is left blank — the block is
    recomputed every turn, so the point reappears once the object
    becomes visible again (e.g. after removing an obstructor).

    Includes the section header — callers should only invoke this for
    same_attribute / relative_size questions.  Non-A/B questions pass
    an empty string so the entire section is omitted.
    """
    if not ab_info:
        return ""
    lines = [
        "== A/B TARGETS ==",
        "(Each entry gives a normalized [row, col] centroid on a [0, 1000] grid;",
        "row=0 top, col=0 left. A blank point means the object is severely",
        "cluttered/occluded — clear obstructors and the point will reappear.)",
    ]
    for letter in ("A", "B"):
        info = ab_info.get(letter)
        if info is None:
            lines.append(f"  {letter}: (undefined)")
            continue
        if info.get("visible") and info.get("point") is not None:
            r, c = info["point"]
            lines.append(f"  {letter}: point=[{r}, {c}]")
        else:
            lines.append(f"  {letter}: point=[ ,  ]")
    return "\n".join(lines)


def draw_ab_markers(
    rgb: np.ndarray, seg: np.ndarray,
    marker_obj_ids: Dict[str, int],
    marker_full_area: Dict[str, int],
    visibility_threshold: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, bool]]:
    """Draw A/B letter markers on objects if they are ≥ ``visibility_threshold`` visible.

    NOTE: as of Issue #1 (FIXES_FOR_April16.MD), markers are NO LONGER drawn
    on images shown to the VLM.  A/B identity is communicated in text via
    ``format_ab_targets_block()``.  This function is retained only for
    debug / GT-annotation images that humans inspect (e.g. ``*_gt_scene.png``).

    Returns (annotated_rgb, {"A": bool, "B": bool}).
    """
    annotated = np.ascontiguousarray(rgb.copy())
    h, w = seg.shape[:2]
    markers_visible: Dict[str, bool] = {}
    colors = {"A": (255, 50, 50), "B": (50, 50, 255)}

    for letter in ("A", "B"):
        oid = marker_obj_ids.get(letter)
        full_area = marker_full_area.get(letter, 0)
        if oid is None or full_area <= 0:
            markers_visible[letter] = False
            continue
        vis_mask = (seg == oid)
        ratio = int(vis_mask.sum()) / float(full_area)
        if ratio < visibility_threshold:
            markers_visible[letter] = False
            continue
        markers_visible[letter] = True
        ys, xs = np.where(vis_mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        color = colors[letter]
        radius = max(14, min(h, w) // 30)
        cv2.circle(annotated, (cx, cy), radius, (0,0,0), -1)
        cv2.circle(annotated, (cx, cy), radius, color, 2)
        fs = radius / 20.0
        tsz = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
        cv2.putText(annotated, letter,
                    (cx - tsz[0]//2, cy + tsz[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 2, cv2.LINE_AA)
    return annotated, markers_visible


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def load_prompt(filename: str) -> str:
    """Load a prompt template from ``prompts/vlm_demo/``."""
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_marker_mapping(marker_map: Dict[int, str]) -> str:
    """Format marker_map as a numbered list for the prompt."""
    if not marker_map:
        return "(no objects detected)"
    return "\n".join(f"  [{n}] {marker_map[n]}" for n in sorted(marker_map.keys()))


def format_action_history(history: List[dict]) -> str:
    """Build a detailed action-history string for the prompt."""
    if not history:
        return "(no actions taken yet)"
    lines = []
    for i, entry in enumerate(history, 1):
        tool = entry.get("tool", "unknown")
        reasoning = entry.get("reasoning", "")
        if tool == "perceive_scene":
            n = entry.get("n_objects_detected", 0)
            mm = entry.get("marker_map", {})
            obj_list = ", ".join(f"[{k}] {v}" for k, v in sorted(mm.items(), key=lambda x: x[0]))
            lines.append(f"  Step {i}: perceive_scene → {n} objects: {obj_list}")
        elif tool == "perceive_find_object":
            oname = entry.get("object_name", "?")
            status = entry.get("status", "?")
            if status == "found":
                lines.append(f"  Step {i}: perceive_find_object('{oname}') → found")
            elif status == "scene_scan":
                n = entry.get("n_objects_detected", 0)
                note = entry.get("note", "")
                lines.append(f"  Step {i}: perceive_find_object → scene scan ({n} objects). {note}")
            else:
                lines.append(f"  Step {i}: perceive_find_object('{oname}') → NOT VISIBLE")
        elif tool == "perceive_get_obstructors":
            obs = entry.get("obstructors", [])
            mm = entry.get("marker_map", {})
            obj_list = ", ".join(f"[{k}] {v}" for k, v in sorted(mm.items(), key=lambda x: x[0]))
            lines.append(f"  Step {i}: perceive_get_obstructors → {len(obs)} obstructors: {obj_list}")
        elif tool == "grasp_and_remove":
            oid = entry.get("object_id", "?")
            oname = entry.get("object_name", "?")
            ok = entry.get("success", False)
            note = entry.get("note", "")
            if ok:
                lines.append(f"  Step {i}: grasp_and_remove(#{oid} '{oname}') → REMOVED ✓")
            else:
                lines.append(f"  Step {i}: grasp_and_remove(#{oid} '{oname}') → FAILED ✗ {note}")
        elif tool == "push":
            oid = entry.get("object_id", "?")
            oname = entry.get("object_name", "?")
            direction = entry.get("direction", "?")
            ok = entry.get("success", False)
            note = entry.get("note", "")
            if ok:
                lines.append(f"  Step {i}: push(#{oid} '{oname}', {direction}) → PUSHED ✓")
            else:
                lines.append(f"  Step {i}: push(#{oid} '{oname}', {direction}) → FAILED ✗ {note}")
        elif tool == "answer_question":
            ans = entry.get("answer", "?")
            lines.append(f"  Step {i}: answer_question → \"{ans}\"")
        elif tool == "error":
            note = entry.get("note", "parse error")
            lines.append(f"  Step {i}: ERROR — {note[:150]}")
        else:
            lines.append(f"  Step {i}: {tool}")
        if reasoning:
            lines.append(f"           Reasoning: {reasoning}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Non-object label filter
# ---------------------------------------------------------------------------

FILTER_OUT_LABELS = {
    "table", "floor", "wooden floor", "wood floor", "ground", "background",
    "wall", "tray", "basket", "shadow", "surface", "patterned floor",
    "brown table", "wooden table",
}


def _is_non_object(label: str) -> bool:
    """Return True if the label is a non-object (table, floor, etc.).

    Uses substring matching so compound variants like "wooden surface",
    "brown table", "table surface" are caught without enumerating every
    possible combination.  See FIXES_FOR_April16.MD §9.
    """
    lbl = label.strip().lower()
    if lbl in FILTER_OUT_LABELS:
        return True
    return any(filt in lbl for filt in FILTER_OUT_LABELS)


def _filter_non_objects(detections: List[dict], marker_map: Dict[int, str]):
    """Remove non-object detections and re-index markers 1..N.

    Returns (filtered_detections, new_marker_map).
    """
    kept = [(d, label) for d, (_, label) in zip(detections, sorted(marker_map.items()))
            if not _is_non_object(label)]
    if not kept:
        return [], {}
    new_dets = [d for d, _ in kept]
    new_map = {i: label for i, (_, label) in enumerate(kept, 1)}
    return new_dets, new_map


# ---------------------------------------------------------------------------
# Centroid / quadrant helpers
# ---------------------------------------------------------------------------

def compute_centroid_rowcol(mask: np.ndarray, image_hw: Tuple[int, int] = None) -> Tuple[int, int]:
    """Compute centroid of a binary mask, normalized to [0, 1000] row/col grid."""
    ys, xs = np.where(mask.astype(bool))
    if len(ys) == 0:
        return (500, 500)
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    H, W = mask.shape[:2] if image_hw is None else image_hw
    row = int(round(cy / max(H - 1, 1) * 1000))
    col = int(round(cx / max(W - 1, 1) * 1000))
    return (max(0, min(1000, row)), max(0, min(1000, col)))


def compute_quadrant(row: int, col: int) -> str:
    """Return quadrant name on [0, 1000] grid."""
    # Center band: 333-666 for both row and col
    if 333 <= row <= 666 and 333 <= col <= 666:
        return "center"
    v = "top" if row < 500 else "bottom"
    h = "left" if col < 500 else "right"
    return f"{v}-{h}"


def _compute_visibility(mask: np.ndarray) -> float:
    """Estimate visibility from mask area (1.0 = fully visible placeholder).

    True visibility would need GT full-area data which is unavailable at runtime.
    We return 1.0 as a placeholder — the VLM should judge from the image.
    """
    return 1.0


# ---------------------------------------------------------------------------
# Perception meta builder
# ---------------------------------------------------------------------------

def build_perception_meta(
    perception_type: str,
    marker_map: Dict[int, str],
    detections: List[dict],
    image_hw: Tuple[int, int],
    query: str = None,
    status: str = "found",
) -> dict:
    """Build a perception_meta dict from perception results.

    Contains marker_map, centroids, visibilities, image_hw and a labelled_img
    slot (set by caller).
    """
    centroids = {}
    visibilities = {}
    mask_areas = {}
    confidences = {}
    for i, det in enumerate(detections, 1):
        mask = det.get("mask")
        if mask is not None:
            centroids[i] = compute_centroid_rowcol(mask, image_hw)
            visibilities[i] = _compute_visibility(mask)
            mask_areas[i] = int(mask.astype(bool).sum())
        else:
            print("Masks were not detected. PLEASE CHECK")
            centroids[i] = (500, 500)
            visibilities[i] = 1.0
            mask_areas[i] = 0
        confidences[i] = det.get("confidence", "medium")
    return {
        "type": perception_type,
        "marker_map": dict(marker_map),
        "centroids": centroids,
        "visibilities": visibilities,
        "mask_areas": mask_areas,
        "confidences": confidences,
        "image_hw": image_hw,
        "query": query,
        "status": status,
        "fresh": True,
        "labelled_img": None,  # set by caller
    }


# ---------------------------------------------------------------------------
# New prompt formatters (§5.3 – §5.7)
# ---------------------------------------------------------------------------

def format_perception_yaml(
    perception_meta: dict,
) -> str:
    """Return the YAML-style text block for the latest perception result."""
    if perception_meta is None:
        return ("(no perception run yet — call perceive_scene or "
                "perceive_find_object before grasping/pushing)")

    ptype = perception_meta["type"]
    mm = perception_meta["marker_map"]
    centroids = perception_meta["centroids"]
    vis = perception_meta["visibilities"]
    mask_areas = perception_meta.get("mask_areas", {})
    hw = perception_meta["image_hw"]
    total_pixels = max(hw[0] * hw[1], 1)
    query = perception_meta.get("query")
    status = perception_meta.get("status", "found")

    lines = []
    lines.append(f"perception_type: {ptype}")

    if ptype == "find_object":
        lines.append(f'query: "{query}"')
        lines.append(f"status: {status}")
        if status == "not_found":
            lines.append("note: Requested object not detected in current view. "
                         "It may be hidden, occluded, or not present. "
                         "Consider perceive_scene to get a full inventory.")
            return "\n".join(lines)
        lines.append(f"n_instances: {len(mm)}")

    lines.append(f"image_hw: [{hw[0]}, {hw[1]}]")
    if ptype == "scene":
        lines.append(f"n_objects: {len(mm)}")
    lines.append("objects:")

    confidences = perception_meta.get("confidences", {})

    for mid in sorted(mm.keys()):
        name = mm[mid]
        rc = centroids.get(mid, (500, 500))
        v = vis.get(mid, 1.0)
        area_px = mask_areas.get(mid, 0)
        area_pct = area_px / total_pixels * 100.0 if area_px else 0.0
        q = compute_quadrant(rc[0], rc[1])
        conf = confidences.get(mid, "medium")
        area_str = f"  area: {area_pct:.1f}%"
        conf_str = f"  confidence: {conf}" if conf != "medium" else ""
        vis_str = f"  visibility: {v:.1f}" if v < 0.9 else ""
        occ_note = "  (partially occluded)" if v < 0.7 else ""
        lines.append(
            f'  - id: {mid}   name: "{name}"   '
            f'centroid_rowcol: [{rc[0]}, {rc[1]}]  '
            f'quadrant: "{q}"{area_str}{conf_str}{vis_str}{occ_note}'
        )

    lines.append("note: IDs are valid until the next grasp_and_remove or push "
                  "— after any manipulation, re-run perception before using an ID.")
    lines.append("note: 'area' is the % of total image pixels occupied by the object's visible mask — "
                  "use it as a rough proxy for apparent size (larger area ≈ larger or closer object).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mental map — persistent perception aggregate (FIXES_FOR_April18 P2-3)
# ---------------------------------------------------------------------------
#
# The mental map aggregates all perception calls since the last
# manipulation.  A ``perceive_find_object`` no longer wipes a prior
# ``perceive_scene`` — the two results merge.  Entries are deduplicated
# by the *PyBullet object ID* under the centroid pixel (looked up in the
# atomic-seg frame captured alongside the Blender RGB), so two detections
# pointing to the same physical body are coalesced.  This is used
# internally for dedup only — the VLM sees perception labels, never
# PyBullet IDs.
#
# The map is cleared on every manipulation (grasp_and_remove / push).
#
# Each entry value is:
#   {
#     "name":          str,               # display label from perception
#     "centroid_rowcol": (row, col),     # latest seen
#     "quadrant":      str,
#     "area_pct":      float,             # most-recent visible area %
#     "visibility":    float,             # 0..1
#     "source_tool":   str,               # which tool first detected it
#     "first_step":    int,               # step when first seen
#     "last_step":     int,               # step when last refreshed
#   }

def mental_map_new() -> dict:
    """Return a fresh (empty) mental map."""
    return {}


def _lookup_pyb_id(seg: np.ndarray, row: int, col: int, radius: int = 5) -> int:
    """Return the PyBullet object ID at (row, col) in the seg buffer, or -1.

    Uses a small neighbourhood (``radius`` px) and takes the majority-vote
    non-zero ID — robust to the centroid landing on a mask boundary.
    """
    if seg is None:
        return -1
    H, W = seg.shape[:2]
    r0, r1 = max(0, row - radius), min(H, row + radius + 1)
    c0, c1 = max(0, col - radius), min(W, col + radius + 1)
    patch = seg[r0:r1, c0:c1]
    if patch.size == 0:
        return -1
    vals, counts = np.unique(patch[patch > 0], return_counts=True)
    if len(vals) == 0:
        return -1
    return int(vals[int(np.argmax(counts))])


def mental_map_add(
    mental_map: dict,
    perception_meta: dict,
    seg_pb: np.ndarray,
    source_tool: str,
    step: int,
) -> dict:
    """Merge a fresh perception_meta into the mental map.

    Dedup key is the PyBullet object ID under each detection's centroid.
    When the key already exists we refresh centroid/area/visibility and
    bump ``last_step``; ``first_step`` / ``source_tool`` are preserved
    from the earliest observation.  Returns ``mental_map`` (mutated).
    """
    if perception_meta is None:
        return mental_map
    mm_labels = perception_meta.get("marker_map", {})
    centroids = perception_meta.get("centroids", {})
    vis = perception_meta.get("visibilities", {})
    mask_areas = perception_meta.get("mask_areas", {})
    hw = perception_meta.get("image_hw", (1, 1))
    total_pixels = max(hw[0] * hw[1], 1)

    for mid, name in mm_labels.items():
        rc = centroids.get(mid, (500, 500))
        area_px = mask_areas.get(mid, 0)
        v = vis.get(mid, 1.0)
        area_pct = area_px / total_pixels * 100.0 if area_px else 0.0
        q = compute_quadrant(rc[0], rc[1])

        # Map normalized centroid (0-1000) back to pixel row/col for seg
        # lookup.  ``centroids`` is in the 0-1000 grid (see
        # ``compute_centroid_rowcol``).
        H, W = hw
        px_row = int(round(rc[0] / 1000.0 * max(H - 1, 1)))
        px_col = int(round(rc[1] / 1000.0 * max(W - 1, 1)))
        pyb_id = _lookup_pyb_id(seg_pb, px_row, px_col)

        # If the seg lookup fails (point on table / air), fall back to a
        # pixel-distance key so at least within-turn duplicates collapse.
        key = pyb_id if pyb_id > 0 else f"px_{px_row}_{px_col}"

        if key in mental_map:
            entry = mental_map[key]
            entry["centroid_rowcol"] = rc
            entry["quadrant"] = q
            entry["area_pct"] = area_pct
            entry["visibility"] = v
            entry["last_step"] = step
            # Keep the name that looks *less* generic (prefer the one with
            # more attribute words — crude heuristic).
            if len(str(name).split()) > len(str(entry["name"]).split()):
                entry["name"] = name
        else:
            mental_map[key] = {
                "name": name,
                "centroid_rowcol": rc,
                "quadrant": q,
                "area_pct": area_pct,
                "visibility": v,
                "source_tool": source_tool,
                "first_step": step,
                "last_step": step,
            }

    return mental_map


def format_mental_map(
    mental_map: dict,
    current_step: int,
    latest_perception_meta: dict = None,
) -> str:
    """Format the mental map as a prompt block.

    If ``latest_perception_meta`` is supplied and is fresh, it is rendered
    as the "most recent perception" block (same text as
    ``format_perception_yaml``) and the mental map is rendered under a
    separate "mental_map" section showing all *other* still-known
    entries with their age (turns since last seen).
    """
    out_lines: list = []

    latest_fresh = (latest_perception_meta is not None
                    and latest_perception_meta.get("fresh", False))
    if latest_fresh:
        out_lines.append("# Most-recent perception result")
        out_lines.append(format_perception_yaml(latest_perception_meta))

    if not mental_map:
        if not latest_fresh:
            return ("(no perception run yet — call perceive_scene or "
                    "perceive_find_object before grasping/pushing)")
        return "\n".join(out_lines)

    # Only show mental_map entries NOT already in the latest perception —
    # the latest perception result (above) already has IDs, centroids,
    # area%, confidence for everything it detected.  The mental map is
    # only useful for objects seen in earlier scans but not the latest.
    latest_names = set()
    if latest_fresh and latest_perception_meta:
        latest_names = set(
            str(v).lower()
            for v in latest_perception_meta.get("marker_map", {}).values()
        )

    entries = sorted(mental_map.items(),
                      key=lambda kv: (kv[1]["first_step"], kv[1]["name"]))
    extra_entries = [
        (_key, e) for _key, e in entries
        if e["name"].lower() not in latest_names
    ]

    if extra_entries:
        out_lines.append("")
        out_lines.append("# additional objects from earlier perceptions "
                         "(not in the latest scan)")
        out_lines.append(f"n_entries: {len(extra_entries)}")
        out_lines.append("entries:")
        for _key, e in extra_entries:
            area_str = f"  area: {e['area_pct']:.1f}%"
            vis_str = f"  visibility: {e['visibility']:.1f}" \
                if e["visibility"] < 0.9 else ""
            out_lines.append(
                f'  - name: "{e["name"]}"   '
                f'centroid_rowcol: [{e["centroid_rowcol"][0]}, {e["centroid_rowcol"][1]}]  '
                f'quadrant: "{e["quadrant"]}"{area_str}{vis_str}  '
                f'source: {e["source_tool"]}  first_seen: step {e["first_step"]}'
            )
        out_lines.append("note: These objects were seen in earlier perceptions "
                         "but not detected in the latest scan. They may have "
                         "moved or been removed.")
    return "\n".join(out_lines)


def format_scene_state(
    q_desc: dict,
    objects_removed: list,
    latest_perception_meta: dict = None,
    history: list = None,
) -> str:
    """Return the SCENE STATE block (§5.3).

    Note: A/B marker visibility is intentionally NOT included here.
    The VLM must infer marker visibility from the RGB image itself.
    """
    qtype = q_desc.get("question_type", "")
    question = q_desc.get("question", "")

    # Derive goal from question type
    if qtype == "under_object":
        cover = q_desc.get("cover_object", "the cover object")
        goal = f"identify the object hidden under the {cover}."
    elif qtype == "counting":
        goal = f"count the target objects in the scene."
    elif qtype == "interactive_counting":
        goal = f"remove objects then count how many remain."
    elif qtype == "existence":
        goal = f"determine whether the queried object is present."
    elif qtype == "same_attribute":
        goal = f"compare the color of objects marked A and B."
    elif qtype == "relative_size":
        goal = f"compare volumes of objects marked A and B."
    else:
        goal = f"answer: {question[:80]}"

    lines = [f"Goal: {goal}", ""]

    # Object inventory is provided by the PERCEPTION RESULT section
    # (with IDs, centroids, area%, confidence).  We don't duplicate it
    # here to avoid cognitive dilution.
    n_known = 0
    if latest_perception_meta and latest_perception_meta.get("marker_map"):
        n_known = len(latest_perception_meta["marker_map"])
    if n_known:
        lines.append(f"Objects perceived: {n_known} (see PERCEPTION RESULT for details)")
    else:
        lines.append("Objects perceived: none yet — run perceive_scene or perceive_find_object")

    lines.append("")

    # Removed objects
    if objects_removed:
        lines.append("Removed:")
        for name, step_removed in objects_removed:
            lines.append(f"  - {name}  (step {step_removed})")
    else:
        lines.append("Removed: (none)")

    lines.append("")

    # Attempts summary
    if history:
        from collections import Counter
        tool_counts = Counter()
        grasp_results = {}  # name → {ok, fail}
        for entry in history:
            t = entry.get("tool", "?")
            if t == "perceive_find_object":
                oname = entry.get("object_name", "?")
                tool_counts[f"perceive_find_object('{oname}')"] += 1
            elif t == "perceive_scene":
                tool_counts["perceive_scene"] += 1
            elif t == "grasp_and_remove":
                oname = entry.get("object_name", "?")
                if oname not in grasp_results:
                    grasp_results[oname] = {"ok": 0, "fail": 0}
                if entry.get("success"):
                    grasp_results[oname]["ok"] += 1
                else:
                    grasp_results[oname]["fail"] += 1
            elif t == "push":
                oname = entry.get("object_name", "?")
                direction = entry.get("direction", "?")
                ok = "✓" if entry.get("success") else "✗"
                tool_counts[f"push({oname}, {direction}) {ok}"] += 1

        lines.append("Attempts summary:")
        for k, v in tool_counts.items():
            lines.append(f"  - {k}: {v} call(s)")
        for name, r in grasp_results.items():
            lines.append(f"  - grasp {name}: {r['fail']} failed, {r['ok']} succeeded")
    else:
        lines.append("Attempts summary: (none)")

    lines.append("")

    return "\n".join(lines)


def format_agent_notes(last_plan: str, last_belief: str) -> str:
    """Return the YOUR NOTES FROM LAST TURN block (§5.5)."""
    if not last_plan and not last_belief:
        return "(no prior notes — this is your first turn; output a plan)"
    parts = []
    if last_plan:
        parts.append(f"Plan:   {last_plan}")
    if last_belief:
        parts.append(f"Belief: {last_belief}")
    return "\n".join(parts)


def format_recent_actions(history: list, n: int = 2) -> str:
    """Return last N actions verbatim with reasoning (§5.6)."""
    if not history:
        return "(no actions taken yet)"
    recent = history[-n:]
    lines = []
    for entry in recent:
        step = entry.get("step_number", "?")
        tool = entry.get("tool", "?")
        reasoning = entry.get("reasoning", "")

        if tool == "perceive_scene":
            n_obj = entry.get("n_objects_detected", 0)
            lines.append(f"  Step {step}: perceive_scene → {n_obj} objects")
        elif tool == "perceive_find_object":
            oname = entry.get("object_name", "?")
            status = entry.get("status", "?")
            n_inst = entry.get("n_instances", 0)
            if status == "found":
                lines.append(f"  Step {step}: perceive_find_object('{oname}') → {n_inst} instance(s)")
            else:
                lines.append(f"  Step {step}: perceive_find_object('{oname}') → NOT FOUND")
        elif tool == "grasp_and_remove":
            oid = entry.get("object_id", "?")
            oname = entry.get("object_name", "?")
            ok = entry.get("success", False)
            if ok:
                lines.append(f"  Step {step}: grasp_and_remove(#{oid} '{oname}') → REMOVED ✓")
            else:
                lines.append(f"  Step {step}: grasp_and_remove(#{oid} '{oname}') → FAILED ✗")
        elif tool == "push":
            oid = entry.get("object_id", "?")
            oname = entry.get("object_name", "?")
            d = entry.get("direction", "?")
            ok = entry.get("success", False)
            tag = "PUSHED ✓" if ok else "FAILED ✗"
            lines.append(f"  Step {step}: push(#{oid} '{oname}', {d}) → {tag}")
        else:
            lines.append(f"  Step {step}: {tool}")

        if reasoning:
            lines.append(f"           Reasoning: {reasoning}")
    return "\n".join(lines)


def format_longterm_summary(history: list) -> str:
    """Return compressed tool-usage summary with perception results (§5.7).

    Perception entries include the object lists they returned so the VLM
    can reference prior observations.
    See FIXES_FOR_April16.MD §13.
    """
    if not history:
        return "(no actions taken yet)"

    lines = []
    total_manips = 0

    for step_idx, entry in enumerate(history):
        tool = entry.get("tool", "?")

        if tool == "perceive_scene":
            mm = entry.get("marker_map", {})
            n = entry.get("n_objects_detected", len(mm))
            obj_list = ", ".join(f"[{k}] {v}" for k, v in sorted(mm.items(), key=lambda x: x[0]))
            line = f"  step {step_idx+1}: perceive_scene → {n} objects"
            if obj_list:
                line += f": {obj_list}"
            lines.append(line)

        elif tool == "perceive_find_object":
            oname = entry.get("object_name", "?")
            status = entry.get("status", "?")
            n = entry.get("n_objects_detected", entry.get("n_instances", 0))
            if status in ("found", "scene_scan"):
                mm = entry.get("marker_map", {})
                obj_list = ", ".join(f"[{k}] {v}" for k, v in sorted(mm.items(), key=lambda x: x[0]))
                line = f"  step {step_idx+1}: perceive_find_object('{oname}') → {status}"
                if obj_list:
                    line += f" ({n}): {obj_list}"
            else:
                line = f"  step {step_idx+1}: perceive_find_object('{oname}') → NOT VISIBLE"
            lines.append(line)

        elif tool == "grasp_and_remove":
            oname = entry.get("object_name", "?")
            ok = "succeeded" if entry.get("success") else "failed"
            lines.append(f"  step {step_idx+1}: grasp_and_remove({oname}): {ok}")
            total_manips += 1

        elif tool == "push":
            oname = entry.get("object_name", "?")
            d = entry.get("direction", "?")
            ok = "succeeded" if entry.get("success") else "failed"
            dist = entry.get("distance_cm")
            dist_str = f" {dist:.0f}cm" if dist else ""
            lines.append(f"  step {step_idx+1}: push({oname}, {d}{dist_str}): {ok}")
            total_manips += 1

        elif tool == "error":
            lines.append(f"  step {step_idx+1}: ERROR — {entry.get('note', 'parse error')[:80]}")

    lines.append(f"Total steps: {len(history)} | Manipulations consumed: {total_manips}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GT annotated image
# ---------------------------------------------------------------------------

def save_gt_annotated_image(demo, q_desc: dict, occ_stats: dict, save_path: str):
    """Save an RGB image annotated with GT object roles and bounding boxes."""
    import pybullet as p

    rgb, _, seg = demo.env.camera.get_cam_img_pybullet()
    annotated = (rgb.astype(np.float32) * 0.7).clip(0, 255).astype(np.uint8)
    H, W = annotated.shape[:2]
    target_name = q_desc.get("target_name")
    comparison = q_desc.get("comparison_object")
    cover = q_desc.get("cover_object")
    ROLE_COLORS = {
        "TARGET": (0, 0, 255), "COMPARE": (255, 165, 0),
        "COVER": (0, 165, 255), "OTHER": (160, 160, 160),
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    for oid, oname in zip(demo.env.obj_ids, demo.env.obj_names):
        mask = (seg == oid)
        if mask.sum() < 10:
            continue
        if oname == target_name:
            role = "TARGET"
        elif oname == comparison:
            role = "COMPARE"
        elif oname == cover:
            role = "COVER"
        else:
            role = "OTHER"
        color = ROLE_COLORS[role]
        thickness = 3 if role != "OTHER" else 1
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        if role != "OTHER":
            label, fs = f"{role}: {oname}", 0.5
        else:
            label, fs = oname, 0.35
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, y1-4), font, fs, (255,255,255), 1, cv2.LINE_AA)

    question = q_desc.get("question", "")
    gt_answer = q_desc.get("gt_answer", "")
    qtype = q_desc.get("question_type", "")
    y_off = 20
    for line in [f"[{qtype}] Q: {question}", f"GT: {gt_answer}"]:
        (tw, th), _ = cv2.getTextSize(line, font, 0.5, 1)
        cv2.rectangle(annotated, (0, y_off-th-4), (tw+8, y_off+4), (0,0,0), -1)
        cv2.putText(annotated, line, (4, y_off), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y_off += th + 12
    cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))


# Compatibility exports: physical tools live in manipulation/primitives.py.
from manipulation.primitives import (
    _GRIPPER_DEPTH,
    _TABLE_Z,
    _MAX_APPROACH_DEG,
    _MAX_CONTACT_DIST,
    _MIN_GRASP_CONF,
    _MASK_BORDER_FRAC,
    _N_4DOF_ATTEMPTS,
    _N_6DOF_ATTEMPTS,
    _SETTLE_STEPS,
    _POST_RELEASE_SETTLE,
    _R_GRIP_TO_EE,
    _FINGER_LAT_FRIC,
    _FINGER_SPIN_FRIC,
    _FINGER_ROLL_FRIC,
    _OBJ_LAT_FRIC,
    _OBJ_SPIN_FRIC,
    _OBJ_ROLL_FRIC,
    _PUSH_DISTANCE,
    _MIN_CLEARANCE_PX,
    _MAX_PUSH_ATTEMPTS,
    _DIRECTION_VECTORS,
    _ANTI_PUSH_IMAGE,
    _PUSH_FINGER_YAW,
    set_finger_offset,
    build_sam_occupancy,
    re_perceive_object,
    _MAX_PC_POINTS,
    set_num_pc_points,
    _build_pc_from_mask,
    _dilate_mask,
    _robot_to_pixel,
    _pixel_to_world,
    _run_graspgen,
    _make_topdown_quat,
    _filter_6dof,
    _filter_4dof,
    _rank_angle_first,
    _boost_friction,
    _fsm_grasp,
    execute_grasp_and_remove,
    _find_push_params,
    _fsm_push,
    execute_push
)

# Compatibility exports: scene graph implementation lives in perception/.
from perception.scene_graph import (
    VLM_SCENE_GRAPH_SYSTEM,
    VLM_SCENE_GRAPH_SYSTEM_V2,
    perceive_scene_v2,
    draw_scene_graph_overlay,
    format_scene_graph_block,
    _draw_labelled_overlay
)
