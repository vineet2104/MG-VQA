"""
gt_visibility_filter.py
=======================

Helper used by ``demo_conversational.py`` (and other conversational-mode
scripts) to ensure ground-truth question generation only counts objects
that are *actually* on the table AND visible to the camera at GT-time.

Why this is needed
------------------
After spawning N objects and letting physics settle, some objects can
end up:

  (a) Off the table (rolled off, fell through a gap)  →  z far below
      ``Z_TABLE_TOP``.  These are already pruned by
      ``RobotEnvUI.spawn() → env.remove_out_of_bounds_objs()``.

  (b) Still on the table but **not visible** in the rendered RGB
      (rolled to a far edge, occluded outside the camera frustum).
      ``remove_out_of_bounds_objs`` does NOT remove these — they remain
      in ``env.obj_names`` / ``env.obj_ids``.  This is a problem
      because the question generators (``_generate_counting_question``,
      ``_generate_existence_question``, etc.) operate on
      ``list(demo.env.obj_names)`` and produce a GT that the VLM
      cannot match — the VLM never sees the object.

This module exposes a single public helper, :func:`filter_visible_obj_names`,
which returns the subset of object names that are both:

    z >= ``Z_TABLE_TOP - z_tolerance``   AND
    visible_pixels >= ``min_visible_pixels``  (in the post-settle seg)

The original ``demo.py`` script is **not** modified by this file —
this is a separate helper imported only by the conversational-mode
scripts.  See ``docs/conversational-migration.md`` for full context.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import numpy as np


# Default tolerances — tuned for the existing 1024 × 1024 camera setup.
# Object is on-table if z >= Z_TABLE_TOP - 1 cm.
DEFAULT_Z_TOLERANCE_M: float = 0.01

# Object is visible if it occupies at least this many pixels in the seg
# mask.  At 1024² that's 100 / (1024 × 1024) ≈ 0.01 % of the image —
# small enough not to filter out tiny but visible objects, large enough
# to filter "1-pixel sliver visible at far edge of frame".
DEFAULT_MIN_VISIBLE_PIXELS: int = 100


def filter_visible_obj_names(
    demo,
    settled_seg: np.ndarray,
    *,
    z_tolerance_m: float = DEFAULT_Z_TOLERANCE_M,
    min_visible_pixels: int = DEFAULT_MIN_VISIBLE_PIXELS,
    verbose: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return the subset of ``demo.env.obj_names`` that pass GT-visibility.

    Args
    ----
    demo
        ``RobotEnvUI`` instance.  We read ``demo.env.obj_ids``,
        ``demo.env.obj_names``, ``demo.env.Z_TABLE_TOP`` and call
        ``demo.env.get_obj_pos(oid)`` for each.

    settled_seg
        2-D PyBullet segmentation mask captured AFTER all physics has
        settled (the ``_settled_seg`` already computed in
        ``demo_conversational.py``).  Pixel values are object IDs.

    z_tolerance_m
        Object is "on table" if its base-pos z is at least
        ``Z_TABLE_TOP - z_tolerance_m``.  Default = 1 cm.

    min_visible_pixels
        Minimum pixel count in ``settled_seg`` to consider the object
        visible.  Default = 100 pixels (≈ 0.01 % at 1024²).

    verbose
        Print one line per dropped object with the reason.

    Returns
    -------
    names_kept : list[str]
        Subset of ``demo.env.obj_names``, IN THE SAME ORDER as the
        original ``env.obj_names`` list.  Duplicates are preserved
        (counting GT depends on duplicate counts).

    debug_info : dict
        ``{"kept": [...], "dropped_off_table": [...],
        "dropped_invisible": [...], "z_table_top": float, ...}``
        Useful to log to ``selected_objects.json``.
    """
    obj_ids = list(demo.env.obj_ids)
    obj_names = list(demo.env.obj_names)
    z_table_top = float(getattr(demo.env, "Z_TABLE_TOP", 0.785))
    z_min = z_table_top - float(z_tolerance_m)

    names_kept: List[str] = []
    dropped_off_table: List[Dict[str, Any]] = []
    dropped_invisible: List[Dict[str, Any]] = []

    for oid, name in zip(obj_ids, obj_names):
        # ── 1. On-table check (z) ──
        try:
            pos = demo.env.get_obj_pos(oid)
            z = float(pos[2])
        except Exception:
            # If we can't read the pose, treat it as gone.
            z = -1.0

        if z < z_min:
            dropped_off_table.append({
                "obj_id": int(oid), "name": name, "z": z,
            })
            if verbose:
                print(f"  [gt_filter] DROP off-table: {name} "
                      f"(z={z:.3f} < {z_min:.3f})")
            continue

        # ── 2. In-camera-view check (segmentation pixel count) ──
        n_vis = int(np.count_nonzero(settled_seg == oid))
        if n_vis < min_visible_pixels:
            dropped_invisible.append({
                "obj_id": int(oid), "name": name,
                "visible_pixels": n_vis,
            })
            if verbose:
                print(f"  [gt_filter] DROP invisible: {name} "
                      f"(only {n_vis} px visible, threshold "
                      f"{min_visible_pixels})")
            continue

        names_kept.append(name)

    debug_info = {
        "n_spawned": len(obj_names),
        "n_kept": len(names_kept),
        "n_dropped_off_table": len(dropped_off_table),
        "n_dropped_invisible": len(dropped_invisible),
        "kept": list(names_kept),
        "dropped_off_table": dropped_off_table,
        "dropped_invisible": dropped_invisible,
        "z_table_top": z_table_top,
        "z_threshold": z_min,
        "min_visible_pixels": int(min_visible_pixels),
    }
    return names_kept, debug_info


def recompute_counting_target_count(
    counting_target_name: Optional[str],
    visible_obj_names: List[str],
) -> Optional[int]:
    """Recompute counting GT after the visibility filter.

    ``_prepare_counting_scene`` (in ``questions.py``) decides the
    target name and target count BEFORE physics — based on intended
    spawns.  After settle, the actual count of that name in
    ``visible_obj_names`` may be lower (e.g. one of the target
    instances rolled off the camera).

    This helper returns the post-settle visible count so the GT
    answer matches what the VLM can actually see.

    Returns ``None`` if ``counting_target_name`` is ``None``
    (means we are not in counting-mode for this episode).

    .. warning::
       This *currently-visible* count is WRONG for counting: an instance
       buried under another object (0 visible pixels but still on the
       table, inside the camera frustum) is exactly what the agent is
       supposed to dig out and count.  Use
       :func:`recompute_counting_target_count_present` instead, which
       counts every instance that is on-table AND inside the FoV
       (occluded or not) and only drops instances that rolled off-frame /
       off-table.  Kept for backward compatibility / non-counting callers.
    """
    if counting_target_name is None:
        return None
    return sum(1 for n in visible_obj_names if n == counting_target_name)


# Minimum in-FoV unoccluded silhouette (pixels) for a buried-but-present
# object to still count.  An instance pushed beyond the camera frustum (or
# off the table) renders ~0 px even with the alpha-hide-all-others trick, so
# this cleanly separates "buried under a plate" (large silhouette → counts)
# from "gone from the scene" (~0 → does not count).
DEFAULT_MIN_INFOV_FOOTPRINT_PX: int = 100


def recompute_counting_target_count_present(
    counting_target_name: Optional[str],
    demo,
    settled_seg: np.ndarray,
    *,
    z_tolerance_m: float = DEFAULT_Z_TOLERANCE_M,
    min_infov_footprint_px: int = DEFAULT_MIN_INFOV_FOOTPRINT_PX,
    verbose: bool = False,
) -> Optional[int]:
    """Counting GT = every target instance that is PRESENT in the scene.

    "Present" = on the table (z ≥ ``Z_TABLE_TOP - z_tolerance_m``) AND
    inside the camera FoV (unoccluded silhouette ≥ ``min_infov_footprint_px``
    via ``RobotEnvUI.compute_infov_footprint`` — the alpha-hide-all-others
    trick).  This INCLUDES instances that are fully occluded in the current
    view (buried under another object), because the whole point of the
    counting task is that the agent must manipulate the scene to reveal and
    count them.  It EXCLUDES instances that rolled off-frame / off-table
    (silhouette ~0 even with everything else hidden).

    Returns ``None`` if ``counting_target_name`` is ``None`` (not counting).
    """
    if counting_target_name is None:
        return None

    obj_ids = list(demo.env.obj_ids)
    obj_names = list(demo.env.obj_names)
    z_table_top = float(getattr(demo.env, "Z_TABLE_TOP", 0.785))
    z_min = z_table_top - float(z_tolerance_m)

    # Only need footprints for the target-name instances.
    target_ids = [oid for oid, name in zip(obj_ids, obj_names)
                  if name == counting_target_name]
    if not target_ids:
        return 0

    try:
        footprints = demo.compute_infov_footprint(target_ids)
    except Exception as _e:
        if verbose:
            print(f"  [gt_filter] compute_infov_footprint failed ({_e}); "
                  f"falling back to current-visible count")
        footprints = {oid: int(np.count_nonzero(settled_seg == oid))
                      for oid in target_ids}

    count = 0
    for oid in target_ids:
        try:
            z = float(demo.env.get_obj_pos(oid)[2])
        except Exception:
            z = -1.0
        fp = int(footprints.get(oid, footprints.get(str(oid), 0)))
        on_table = z >= z_min
        in_fov = fp >= min_infov_footprint_px
        if on_table and in_fov:
            count += 1
        elif verbose:
            _cur = int(np.count_nonzero(settled_seg == oid))
            print(f"  [gt_filter] counting: DROP instance id={oid} "
                  f"(z={z:.3f} on_table={on_table}, footprint={fp}px "
                  f"in_fov={in_fov}, cur_visible={_cur}px)")
    return count
