"""Point-based find_object tool: planner point → SAM3 → one action marker."""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


def _sam_segment(*args, **kwargs):
    from tools import _sam_segment as segment
    return segment(*args, **kwargs)


def _create_labelled_overlay(*args, **kwargs):
    from tools import _create_labelled_overlay as overlay
    return overlay(*args, **kwargs)


def perceive_find_object_at_point(
    rgb: np.ndarray,
    point,
    sam_model: SAM,
    scene_graph: dict = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[int, str], List[dict], str]:
    """V2 grounding-by-point: segment the object at a VLM-chosen 2D point.

    The V2 planner reads the scene graph produced by ``perceive_scene_v2``
    (which lists every object with a precise ``point`` on the [0, 1000] grid)
    and passes the point of the object it wants to grasp/push.  No pointing
    model / no object-name text is involved — SAM3 is prompted directly at
    that pixel.

    Parameters
    ----------
    point : list/tuple
        ``[row, col]`` on the **[0, 1000]** grid (row 0=top, col 0=left) —
        the SAME convention used by the ``find`` answer point and the scene
        graph ``point`` field.
    scene_graph : dict, optional
        The most recent scene graph (from ``perceive_scene_v2``).  Used ONLY
        to attach a human-readable label to the returned marker (nearest
        object centre); it does not affect segmentation.

    Returns
    -------
    (labelled_img, marker_map, detections, status)
        Exactly one marker (id ``1``) on success — since a single point
        segments a single object.  ``marker_map`` = ``{1: label}``.
        ``detections`` = ``[{"x", "y", "mask", "label"}]``.
        ``status`` is ``"found"`` or ``"not_visible"`` (SAM returned nothing)
        or ``"bad_point"`` (missing/malformed point).
    """
    H, W = rgb.shape[:2]
    if not (isinstance(point, (list, tuple)) and len(point) >= 2):
        return rgb.copy(), {}, [], "bad_point"
    try:
        row_n = float(point[0]) / 1000.0
        col_n = float(point[1]) / 1000.0
    except (TypeError, ValueError):
        return rgb.copy(), {}, [], "bad_point"
    # Clamp to grid then map to pixels.
    row_n = max(0.0, min(1.0, row_n))
    col_n = max(0.0, min(1.0, col_n))
    x = int(round(col_n * max(W - 1, 1)))
    y = int(round(row_n * max(H - 1, 1)))

    # Label from the nearest scene-graph object centre (for readability only).
    label = "object"
    if scene_graph and scene_graph.get("objects"):
        best_d, best_name = None, None
        for o in scene_graph["objects"]:
            pn = o.get("point_norm", [0.5, 0.5])
            oy = pn[0] * max(H - 1, 1)
            ox = pn[1] * max(W - 1, 1)
            d = (ox - x) ** 2 + (oy - y) ** 2
            if best_d is None or d < best_d:
                best_d, best_name = d, o.get("name", "object")
        if best_name:
            label = best_name

    if verbose:
        print(f"[FindObjectAtPoint] point=[{point[0]}, {point[1]}] "
              f"→ px=({x}, {y}) label='{label}'")

    mask = _sam_segment(sam_model, rgb, x, y, verbose, label)
    if mask is None or np.shape(mask) != (H, W) or not np.any(mask):
        return rgb.copy(), {}, [], "not_visible"

    det = {"x": x, "y": y, "mask": mask, "label": label}
    marker_map = {1: label}
    labelled = _create_labelled_overlay(rgb, [mask], [1], marker_map)
    return labelled, marker_map, [det], "found"


# Older name-based grounding remains available only for compatibility.
def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    import tools
    return getattr(tools, name)
