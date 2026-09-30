"""Scene graph perception and debug overlays (no simulator GT input)."""
import numpy as np
import cv2
from typing import Dict, List, Tuple
from models import call_model


def _is_non_object(name):
    from tools import _is_non_object as check
    return check(name)


def _sam_segment(*args, **kwargs):
    from tools import _sam_segment as segment
    return segment(*args, **kwargs)

# =====================================================================
# V2 scene-graph perception tools (for demo_new.py)
# =====================================================================
# These tools implement the `describe_scene` + `find_object`
# decomposition discussed in tool-discussion.md.  They are intentionally
# separate functions so demo.py is unaffected.
# =====================================================================

VLM_SCENE_GRAPH_SYSTEM = (
    "You are an expert visual scene analyst.  Given a top-down image of "
    "objects on a table, produce a detailed scene graph in YAML format.\n\n"
    "For EACH separate physical object, provide:\n"
    "  - obj_id: sequential integer starting at 1\n"
    "  - name: most specific name you can read from visible labels; "
    "if no label, use a short descriptive name\n"
    "  - desc: 1-2 sentence rich description (color, shape, material, "
    "any text/branding visible, approximate size relative to other objects)\n"
    "  - point_norm: [row, col] approximate centre on a normalised "
    "[0.0, 1.0] grid (row 0=top, col 0=left).  Estimate only — "
    "these are for reference, not for precise grounding.\n"
    "  - position_description: short phrase (e.g. 'top-left', 'center', "
    "'far right')\n"
    "  - relations: list of spatial relations to OTHER objects, each as "
    "[relation_type, other_obj_id]. Use ONLY these relation types:\n"
    "      on_top_of        — this object is resting on top of other_obj_id\n"
    "      partially_occluded_by — part of this object is hidden by other_obj_id\n"
    "      next_to           — this object is adjacent to other_obj_id\n"
    "    Only include relations you are confident about.  "
    "An empty list is fine.\n\n"
    "If you see multiple identical objects, list each one separately.\n"
    "Do NOT include the table surface, floor, shadows, or background.\n\n"
    "Also provide a top-level 'scene_summary' field: a 2-3 sentence "
    "free-text description of the overall scene layout, notable "
    "occlusions, and anything that might be hidden.\n\n"
    "Return STRICT YAML.  No markdown code fences.  Example:\n"
    "scene_summary: \"7 objects visible. Cluttered pile in center...\" \n"
    "objects:\n"
    "  - obj_id: 1\n"
    "    name: \"soup can\"\n"
    "    desc: \"small red can with 'ALPHABET' text\"\n"
    "    point_norm: [0.32, 0.49]\n"
    "    position_description: \"center\"\n"
    "    relations:\n"
    "      - [next_to, 2]\n"
    "  - obj_id: 2\n"
    "    name: \"yellow bottle\"\n"
    "    desc: \"cylindrical yellow squeeze bottle with white cap\"\n"
    "    point_norm: [0.16, 0.13]\n"
    "    position_description: \"top-left\"\n"
    "    relations:\n"
    "      - [partially_occluded_by, 3]\n"
)


# V2 variant — the scene graph's 2D points are no longer "approximate
# reference" only; they are consumed directly for grounding (the planner
# passes a chosen point to perceive_find_object, which segments the object at
# exactly that pixel).  So this prompt asks for the MOST PRECISE achievable
# centre point on a [0, 1000] integer grid, plus a clear label per object.
VLM_SCENE_GRAPH_SYSTEM_V2 = (
    "You are an expert visual scene analyst.  Given a top-down image of "
    "objects on a table, produce a detailed scene graph in YAML format.\n\n"
    "For EACH separate physical object, provide:\n"
    "  - obj_id: sequential integer starting at 1\n"
    "  - name: a short, specific label for the object (read any visible "
    "text/branding; otherwise a concise descriptive name)\n"
    "  - desc: 1-2 sentence rich description (color, shape, material, any "
    "text/branding visible, approximate size relative to other objects)\n"
    "  - point: [row, col] the object's PRECISE 2D centre on a [0, 1000] "
    "integer grid (row 0=top, row 1000=bottom, col 0=left, col 1000=right). "
    "Place the point squarely on the most visible, graspable part of the "
    "object — this point WILL be used directly to segment and grasp/push "
    "the object, so make it as accurate as you can.\n"
    "  - position_description: short phrase (e.g. 'top-left', 'center', "
    "'far right')\n"
    "  - relations: list of spatial relations to OTHER objects, each as "
    "[relation_type, other_obj_id]. Use ONLY these relation types:\n"
    "      on_top_of        — this object is resting on top of other_obj_id\n"
    "      partially_occluded_by — part of this object is hidden by other_obj_id\n"
    "      next_to           — this object is adjacent to other_obj_id\n"
    "    Only include relations you are confident about.  "
    "An empty list is fine.\n\n"
    "List EVERY visible object, including ones only partially visible "
    "under/behind others.  If you see multiple identical objects, list each "
    "one separately with its own point.\n"
    "Do NOT include the table surface, floor, shadows, or background.\n\n"
    "Also provide a top-level 'scene_summary' field: a 2-3 sentence "
    "free-text description of the overall scene layout, notable "
    "occlusions, and anything that might be hidden.\n\n"
    "Return STRICT YAML.  No markdown code fences.  Example:\n"
    "scene_summary: \"7 objects visible. Cluttered pile in center...\" \n"
    "objects:\n"
    "  - obj_id: 1\n"
    "    name: \"soup can\"\n"
    "    desc: \"small red can with 'ALPHABET' text\"\n"
    "    point: [320, 490]\n"
    "    position_description: \"center\"\n"
    "    relations:\n"
    "      - [next_to, 2]\n"
    "  - obj_id: 2\n"
    "    name: \"yellow bottle\"\n"
    "    desc: \"cylindrical yellow squeeze bottle with white cap\"\n"
    "    point: [160, 130]\n"
    "    position_description: \"top-left\"\n"
    "    relations:\n"
    "      - [partially_occluded_by, 3]\n"
)


def perceive_scene_v2(
    rgb: np.ndarray,
    vlm_model_name: str,
    sam_model=None,
    verbose: bool = True,
    precise_points: bool = False,
) -> Tuple[dict, np.ndarray, Dict[int, str], List[dict]]:
    """V2 scene perception — rich scene graph + optional SAM grounding.

    Parameters
    ----------
    precise_points : bool
        If True (the new V2 grounding-by-point workflow), use the
        ``VLM_SCENE_GRAPH_SYSTEM_V2`` prompt which asks for PRECISE per-object
        2D centre points (``point`` on a [0, 1000] grid) that are consumed
        directly by ``perceive_find_object`` for segmentation.  If False
        (legacy), use the original prompt whose points are "approximate
        reference only".  Either way the returned scene graph stores the
        object centre in ``point_norm`` (normalised [0, 1]).

    Returns
    -------
    scene_graph : dict
        Parsed YAML scene graph with ``scene_summary`` and ``objects`` list.
        Each object has ``obj_id``, ``name``, ``desc``, ``point_norm``,
        ``position_description``, ``relations``.
    labelled_img : np.ndarray
        Debug image with numbered labels (NOT sent to VLM).
    marker_map : dict
        ``{marker_id: name}`` — maps SAM grounding IDs to names.
        Empty if ``sam_model`` is None.
    detections : list
        Detection dicts with ``mask``, ``x``, ``y``, ``label``.
        Empty if ``sam_model`` is None.
    """
    import yaml as _yaml
    H, W = rgb.shape[:2]

    # 1. Ask the main VLM for a rich scene graph
    _sys = VLM_SCENE_GRAPH_SYSTEM_V2 if precise_points else VLM_SCENE_GRAPH_SYSTEM
    _usr = (
        "Analyse this top-down image and produce the scene graph as "
        "specified.  List every separate physical object and give each a "
        "PRECISE [row, col] centre point on the [0, 1000] grid."
        if precise_points else
        "Analyse this top-down image and produce the scene graph "
        "as specified.  List every separate physical object."
    )
    response = call_model(
        images=[rgb],
        prompt=_usr,
        system_prompt=_sys,
        model_name=vlm_model_name,
        temp=0.0,
        n_tokens=16384,
        disable_thinking=False,
    )

    if verbose:
        print(f"[SceneGraph] Raw response ({len(response)} chars):")
        for line in response[:1000].splitlines():
            print(f"  | {line}")
        if len(response) > 1000:
            print(f"  | ... ({len(response)} total chars)")

    # 2. Parse YAML
    scene_graph = {"scene_summary": "", "objects": []}
    try:
        # Strip any markdown code fences the VLM might have included
        clean = response.strip()
        if clean.startswith("```"):
            # Remove opening fence + optional language tag
            clean = clean.split("\n", 1)[1] if "\n" in clean else ""
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        # Fix common Gemini YAML formatting issue: "obj_id:3" → "obj_id: 3"
        import re as _re
        clean = _re.sub(r'(\w):(\d)', r'\1: \2', clean)
        parsed = _yaml.safe_load(clean)
        if isinstance(parsed, dict):
            scene_graph["scene_summary"] = str(
                parsed.get("scene_summary", ""))
            raw_objects = parsed.get("objects", [])
            if isinstance(raw_objects, list):
                for obj in raw_objects:
                    if not isinstance(obj, dict):
                        continue
                    name = str(obj.get("name", "")).strip()
                    if not name or _is_non_object(name):
                        continue
                    # Normalise the object centre.  Two accepted keys:
                    #   • point       — [row, col] on the [0, 1000] grid
                    #                    (V2 precise-points prompt)
                    #   • point_norm  — [row, col] on the [0, 1] grid (legacy)
                    # Both are stored internally as point_norm ([0, 1]).
                    if "point" in obj:
                        pt = obj.get("point", [500, 500])
                        _scale = 1000.0
                    else:
                        pt = obj.get("point_norm", [0.5, 0.5])
                        _scale = 1.0
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        row_n = float(pt[0]) / _scale
                        col_n = float(pt[1]) / _scale
                    else:
                        row_n, col_n = 0.5, 0.5
                    # Clamp to [0, 1]
                    row_n = max(0.0, min(1.0, row_n))
                    col_n = max(0.0, min(1.0, col_n))

                    # Parse relations
                    relations = []
                    raw_rels = obj.get("relations", [])
                    if isinstance(raw_rels, list):
                        VALID_RELS = {"on_top_of", "partially_occluded_by",
                                      "next_to"}
                        for rel in raw_rels:
                            if (isinstance(rel, (list, tuple))
                                    and len(rel) >= 2
                                    and str(rel[0]) in VALID_RELS):
                                relations.append(
                                    [str(rel[0]), int(rel[1])])

                    scene_graph["objects"].append({
                        "obj_id": int(obj.get("obj_id", 0)),
                        "name": name,
                        "desc": str(obj.get("desc", "")),
                        "point_norm": [row_n, col_n],
                        "position_description": str(
                            obj.get("position_description", "")),
                        "relations": relations,
                    })
    except Exception as e:
        if verbose:
            print(f"[SceneGraph] YAML parse error: {e}")

    if verbose:
        n = len(scene_graph["objects"])
        print(f"[SceneGraph] Parsed {n} objects, "
              f"summary: {scene_graph['scene_summary'][:120]}")

    # 3. Optionally ground each object via SAM to produce marker IDs
    labelled_img = rgb.copy()
    marker_map: Dict[int, str] = {}
    detections: List[dict] = []

    if sam_model is not None and scene_graph["objects"]:
        masks: List[np.ndarray] = []
        for sg_obj in scene_graph["objects"]:
            row_n, col_n = sg_obj["point_norm"]
            y = int(round(row_n * max(H - 1, 1)))
            x = int(round(col_n * max(W - 1, 1)))
            mask = _sam_segment(sam_model, rgb, x, y, verbose,
                                sg_obj["name"])
            if mask is None:
                continue

            # IoU dedup — keep larger mask on collision
            dup = False
            for idx, existing in enumerate(masks):
                inter = np.logical_and(mask, existing).sum()
                union = np.logical_or(mask, existing).sum()
                if union > 0 and inter / union > 0.7:
                    if mask.sum() > existing.sum():
                        # Replace smaller with larger
                        masks[idx] = mask
                        detections[idx] = {
                            "label": sg_obj["name"],
                            "x": x, "y": y,
                            "mask": mask,
                            "sg_obj": sg_obj,
                        }
                    dup = True
                    break
            if not dup:
                masks.append(mask)
                detections.append({
                    "label": sg_obj["name"],
                    "x": x, "y": y,
                    "mask": mask,
                    "sg_obj": sg_obj,
                })

        # Build marker map and labelled debug image
        for i, det in enumerate(detections):
            mid = i + 1
            marker_map[mid] = det["label"]

        if detections:
            labelled_img = _draw_labelled_overlay(
                rgb, detections, marker_map)

    return scene_graph, labelled_img, marker_map, detections


def draw_scene_graph_overlay(
    rgb: np.ndarray,
    scene_graph: dict,
) -> np.ndarray:
    """Draw the scene graph's approximate points on an image for debug.

    Each object gets a small circle at its ``point_norm`` with a
    label "<id>: <name>".  This is saved to disk for analysis only —
    it is NEVER sent to the VLM (the VLM sees only the clean RGB).
    """
    overlay = np.ascontiguousarray(rgb.copy())
    H, W = rgb.shape[:2]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 0, 255), (255, 0, 128), (0, 255, 128),
        (255, 128, 128), (128, 255, 128), (128, 128, 255),
    ]
    for i, obj in enumerate(scene_graph.get("objects", [])):
        oid = obj.get("obj_id", i + 1)
        name = obj.get("name", "?")
        pt = obj.get("point_norm", [0.5, 0.5])
        row_n = float(pt[0]) if len(pt) > 0 else 0.5
        col_n = float(pt[1]) if len(pt) > 1 else 0.5
        y = int(round(row_n * max(H - 1, 1)))
        x = int(round(col_n * max(W - 1, 1)))
        color = colors[i % len(colors)]
        # Filled circle + white outline for visibility on any background
        cv2.circle(overlay, (x, y), 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 6, color, -1, cv2.LINE_AA)
        label = f"{oid}: {name}"
        # Text with black outline for readability
        cv2.putText(overlay, label, (x + 10, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(overlay, label, (x + 10, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)
    return overlay


def format_scene_graph_block(scene_graph: dict, exact_points: bool = False) -> str:
    """Format the V2 scene graph as a prompt block for the VLM.

    Produces a clean text representation that serves as the mental map.

    Parameters
    ----------
    exact_points : bool
        If True (the V2 grounding-by-point workflow), present each object's
        centre as ``point: [row, col]`` on the **[0, 1000]** grid — the same
        value the planner passes to ``perceive_find_object`` to segment that
        object.  If False (legacy), present ``approx_point`` on the [0, 1]
        grid, tagged as approximate.
    """
    lines = []
    lines.append(f"scene_summary: \"{scene_graph.get('scene_summary', '')}\"")
    lines.append(f"n_objects: {len(scene_graph.get('objects', []))}")
    lines.append("objects:")
    for obj in scene_graph.get("objects", []):
        oid = obj.get("obj_id", "?")
        name = obj.get("name", "?")
        desc = obj.get("desc", "")
        pt = obj.get("point_norm", [0.5, 0.5])
        pos = obj.get("position_description", "")
        rels = obj.get("relations", [])

        lines.append(f"  - obj_id: {oid}")
        lines.append(f"    name: \"{name}\"")
        if desc:
            lines.append(f"    desc: \"{desc}\"")
        if exact_points:
            # [row, col] on [0, 1000] — directly usable for grounding.
            row_k = int(round(pt[0] * 1000))
            col_k = int(round(pt[1] * 1000))
            lines.append(f"    point: [{row_k}, {col_k}]")
        else:
            lines.append(f"    approx_point: [{pt[0]:.3f}, {pt[1]:.3f}]  "
                         f"(approximate — use find_object for precise grounding)")
        if pos:
            lines.append(f"    position: \"{pos}\"")
        if rels:
            rel_strs = [f"[{r[0]}, obj_{r[1]}]" for r in rels]
            lines.append(f"    relations: {', '.join(rel_strs)}")
    return "\n".join(lines)


def _draw_labelled_overlay(
    rgb: np.ndarray,
    detections: List[dict],
    marker_map: Dict[int, str],
) -> np.ndarray:
    """Draw numbered labels + translucent masks on an image (debug only)."""
    overlay = np.ascontiguousarray(rgb.copy())
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 0, 255), (255, 0, 128), (0, 255, 128),
    ]
    for i, det in enumerate(detections):
        mid = i + 1
        name = marker_map.get(mid, "?")
        color = colors[i % len(colors)]
        mask = det.get("mask")
        if mask is not None:
            tint = np.zeros_like(overlay)
            tint[mask] = color
            overlay = cv2.addWeighted(overlay, 1.0, tint, 0.3, 0)
        x, y = det.get("x", 0), det.get("y", 0)
        label = f"{mid}: {name}"
        cv2.putText(overlay, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    cv2.LINE_AA)
    return overlay
