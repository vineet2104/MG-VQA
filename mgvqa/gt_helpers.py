"""
core.gt_helpers — Ground-truth annotation and simulator state capture.

These helpers produce debug artefacts (gt_scene.png, sim-gt-stats/*.json)
that are NOT shown to the VLM — they exist for human debugging and
downstream reward-signal computation only.
"""

import os
import json
import numpy as np
import cv2


def _project_aabb_to_image(obj_id, camera):
    """Project a PyBullet body's world AABB into image pixel bbox.

    Returns (x_min, y_min, x_max, y_max) in image coords, or None on error.
    Useful for drawing a bounding box on objects that are completely
    buried (not visible in the seg buffer).
    """
    try:
        import pybullet as p
        aabb_min, aabb_max = p.getAABB(obj_id)
        # 8 corners of the AABB in world coords
        corners = []
        for x in (aabb_min[0], aabb_max[0]):
            for y in (aabb_min[1], aabb_max[1]):
                for z in (aabb_min[2], aabb_max[2]):
                    corners.append((x, y, z))
        # Project: apply view × projection
        view = np.array(camera.view_matrix).reshape(4, 4, order="F")
        proj = np.array(camera.projection_matrix).reshape(4, 4, order="F")
        W, H = camera.width, camera.height
        pts = []
        for cx, cy, cz in corners:
            p_h = np.array([cx, cy, cz, 1.0])
            clip = proj @ (view @ p_h)
            if clip[3] == 0:
                continue
            ndc = clip[:3] / clip[3]
            px = (ndc[0] * 0.5 + 0.5) * W
            py = (1.0 - (ndc[1] * 0.5 + 0.5)) * H
            pts.append((px, py))
        if not pts:
            return None
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x_min = int(max(0, min(xs))); x_max = int(min(W - 1, max(xs)))
        y_min = int(max(0, min(ys))); y_max = int(min(H - 1, max(ys)))
        if x_max <= x_min or y_max <= y_min:
            return None
        return (x_min, y_min, x_max, y_max)
    except Exception:
        return None


def _classify_target_ids(q_desc, env, occ_stats):
    """For a question descriptor, classify every currently-spawned object id
    as either a 'target' (GT-relevant) or a 'distractor' (filler).

    Per-question-type rules:
        counting          → all instances whose name == q_desc['target_name']
        existence (yes)   → q_desc['target_obj_id'] (single)
        existence (no)    → none — the answer-relevant object is absent
        same_attribute    → q_desc['marker_obj_ids']{'A','B'}
        relative_size     → q_desc['marker_obj_ids']{'A','B'}
        under_object      → cover_id (looked up by name) + hidden_id
                             (= q_desc['protected_obj_ids']) + also_valid IDs
        interactive_counting → none (whole-scene count, no focal object)

    Returns a dict of {obj_id: role_label} where role_label is one of
    'target', 'cover', 'hidden', 'also_valid', 'A', 'B', 'distractor'.
    """
    qt = q_desc.get("question_type", "")
    target_name = q_desc.get("target_name")
    marker_ids = q_desc.get("marker_obj_ids") or {}
    target_obj_id = q_desc.get("target_obj_id")
    cover_name = q_desc.get("cover_object")
    protected = q_desc.get("protected_obj_ids") or []
    also_valid_names = q_desc.get("also_valid") or []
    name_to_id = (occ_stats or {}).get("name_to_id", {}) if occ_stats else {}

    role_map: dict = {oid: "distractor" for oid in env.obj_ids}

    if qt == "counting" and target_name:
        for oid, oname in zip(env.obj_ids, env.obj_names):
            if oname == target_name:
                role_map[oid] = "target"
    elif qt == "existence":
        if target_obj_id is not None and target_obj_id in role_map:
            role_map[target_obj_id] = "target"
    elif qt in ("same_attribute", "relative_size"):
        for label in ("A", "B"):
            mid = marker_ids.get(label)
            if mid is not None and mid in role_map:
                role_map[mid] = label
    elif qt == "marked_comparison":
        # 3 texture-baked badges A/B/C → the three comparison targets.
        for label in ("A", "B", "C"):
            mid = marker_ids.get(label)
            if mid is not None and mid in role_map:
                role_map[mid] = label
    elif qt in ("under_object", "beneath"):
        # cover by name (cover_name is a display string)
        cov_id = name_to_id.get(cover_name) if cover_name else None
        if cov_id is not None and cov_id in role_map:
            role_map[cov_id] = "cover"
        for hid in protected:
            if hid in role_map:
                role_map[hid] = "hidden"
        for av_name in also_valid_names:
            av_id = name_to_id.get(av_name)
            if av_id is not None and av_id in role_map and \
                    role_map[av_id] == "distractor":
                role_map[av_id] = "also_valid"
    elif qt == "count":
        # Same as counting — all instances of the target name.
        for oid, oname in zip(env.obj_ids, env.obj_names):
            if oname == target_name:
                role_map[oid] = "target"
    elif qt == "find":
        if target_obj_id is not None and target_obj_id in role_map:
            role_map[target_obj_id] = "target"
    elif qt == "rearrangement":
        # Reference must survive; GT-removal objects are the targets.
        ref_id = q_desc.get("reference_obj_id")
        if ref_id is not None and ref_id in role_map:
            role_map[ref_id] = "reference"
        for rid in (q_desc.get("gt_removal_ids") or []):
            if rid in role_map:
                role_map[rid] = "target"
    # interactive_counting: leave everything as 'distractor' (no focal target)

    return role_map


def _capture_sim_gt_state(
    demo, q_desc, occ_stats, total_steps, original_obj_ids, ep_dir, camera,
):
    """Snapshot the simulator ground-truth state for per-step reward signals.

    Saves a JSON file in ``<ep_dir>/sim-gt-stats/step_<N>.json`` describing:
      - currently spawned objects (id, name, role_label, pose, AABB world+pixel)
      - which IDs were originally targets / distractors / etc. for this question
      - which IDs were originally spawned but are now removed
      - the question type (echoed for downstream consumers)

    NB: object IDs here are PyBullet body IDs that PERSIST across steps.
        They are NOT the marker IDs that ``perceive_find_object`` returns
        to the VLM — those are SAM3 mask indices that change every call.
        Linking sim IDs ↔ VLM marker IDs is intentionally left for a
        downstream reward script (see project plan).
    """
    try:
        import pybullet as p
    except ImportError:
        return None

    out_dir = os.path.join(ep_dir, "sim-gt-stats")
    os.makedirs(out_dir, exist_ok=True)

    role_map = _classify_target_ids(q_desc, demo.env, occ_stats)

    objects = []
    for oid, oname in zip(demo.env.obj_ids, demo.env.obj_names):
        try:
            aabb_min, aabb_max = p.getAABB(oid)
            pos, orn = p.getBasePositionAndOrientation(oid)
        except Exception:
            aabb_min = aabb_max = None
            pos = orn = None
        pixel_bbox = _project_aabb_to_image(oid, camera) if camera is not None else None
        objects.append({
            "obj_id": int(oid),
            "name": str(oname),
            "role": role_map.get(oid, "distractor"),
            "aabb_world": {
                "min": list(aabb_min) if aabb_min is not None else None,
                "max": list(aabb_max) if aabb_max is not None else None,
            },
            "aabb_pixel": (
                {"x_min": pixel_bbox[0], "y_min": pixel_bbox[1],
                 "x_max": pixel_bbox[2], "y_max": pixel_bbox[3]}
                if pixel_bbox is not None else None
            ),
            "position": list(pos) if pos is not None else None,
            "orientation_xyzw": list(orn) if orn is not None else None,
        })

    current_ids = set(demo.env.obj_ids)
    removed_ids = [int(oid) for oid in original_obj_ids if oid not in current_ids]

    # Concise summaries by role for fast reward-script lookups.
    target_role_labels = {"target", "A", "B", "cover", "hidden", "also_valid"}
    target_obj_ids = sorted(int(oid) for oid, r in role_map.items()
                            if r in target_role_labels)
    distractor_obj_ids = sorted(int(oid) for oid, r in role_map.items()
                                if r == "distractor")

    state = {
        "step": int(total_steps),
        "question_type": q_desc.get("question_type", ""),
        "image_size": {"width": int(camera.width), "height": int(camera.height)} \
            if camera is not None else None,
        "n_spawned_total": int(len(original_obj_ids)),
        "n_spawned_now": int(len(current_ids)),
        "original_obj_ids": [int(o) for o in original_obj_ids],
        "removed_obj_ids": removed_ids,
        "target_obj_ids": target_obj_ids,
        "distractor_obj_ids": distractor_obj_ids,
        "role_map": {str(int(oid)): r for oid, r in role_map.items()},
        "objects": objects,
    }

    out_path = os.path.join(out_dir, f"step_{int(total_steps)}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass
    return out_path


def write_review_manifest(
    demo, q_desc, occ_stats, ep_dir, camera,
    *, episode, episode_seed, question_type, benchmark_version,
    image_files, seg=None,
):
    """Write a single self-contained ``review.json`` for the annotation webpage.

    This is the ONE file the human-review UI needs per episode.  It bundles:
      - the editable question meta (question text, target name, options, GT)
      - image paths (clean first frame + annotated gt_scene + seg overlay)
      - the snapshot path (for the eventual replay run)
      - every object with {obj_id, name, role, aabb_pixel} so the webpage can
        draw / re-draw the red/green boxes itself (e.g. after a reviewer edits
        the target name).  The subset the benchmark boxes for THIS question is
        flagged ``is_target``.

    Deliberately NOT stepwise (unlike sim-gt-stats): one snapshot of the
    settled initial scene, matching the clean first frame the reviewer sees.

    Args:
        demo:      RobotEnvUI (settled scene, walls already removed).
        q_desc:    question descriptor.
        occ_stats: occlusion stats dict (has occlusion / name_to_id).
        ep_dir:    episode directory.
        camera:    owg_robot Camera (for AABB→pixel projection of buried objs).
        image_files: dict of {key: filename} for the images saved in ep_dir
                     (e.g. {"clean": "after_walls_topdown.png",
                            "gt_scene": "..._gt_scene.png",
                            "seg_overlay": "..._seg_overlay.png"}).
        seg:       (H, W) settled seg buffer — used to prefer a tight visible
                   bbox over the projected AABB when the object is visible.

    Returns:
        Path to the written review.json.
    """
    try:
        import pybullet as p
    except ImportError:
        p = None

    role_map = _classify_target_ids(q_desc, demo.env, occ_stats)
    target_roles = {"target", "A", "B", "cover", "hidden", "also_valid"}
    occ = occ_stats.get("occlusion", {})

    def _occ_for(oid):
        v = occ.get(oid, occ.get(str(oid)))
        return round(float(v), 4) if v is not None else None

    def _visible_bbox(oid):
        if seg is None:
            return None
        mask = (seg == oid)
        if not np.any(mask):
            return None
        ys, xs = np.where(mask)
        return {"x_min": int(xs.min()), "y_min": int(ys.min()),
                "x_max": int(xs.max()), "y_max": int(ys.max())}

    objects = []
    for oid, oname in zip(demo.env.obj_ids, demo.env.obj_names):
        role = role_map.get(oid, "distractor")
        vis_bbox = _visible_bbox(oid)
        # Prefer the tight visible bbox; fall back to projected AABB (buried).
        proj_bbox = _project_aabb_to_image(oid, camera) if camera is not None else None
        aabb_pixel = vis_bbox
        buried = False
        if aabb_pixel is None and proj_bbox is not None:
            aabb_pixel = {"x_min": proj_bbox[0], "y_min": proj_bbox[1],
                          "x_max": proj_bbox[2], "y_max": proj_bbox[3]}
            buried = True
        objects.append({
            "obj_id": int(oid),
            "name": str(oname),
            "role": role,
            "is_target": role in target_roles,
            "occlusion": _occ_for(oid),
            "buried": buried,
            "aabb_pixel": aabb_pixel,
        })

    # Compact, webpage-friendly question block (editable fields first).
    qmeta = {
        "question_type": q_desc.get("question_type", question_type),
        "question": q_desc.get("question"),
        "target_name": q_desc.get("target_name"),
        # find-"no" has no target_name — surface the queried name so the
        # reviewer can still see / edit what was asked about.
        "queried_name": q_desc.get("queried_name"),
        "cover_object": q_desc.get("cover_object"),
        "answer_options": q_desc.get("answer_options") or [],
        "gt_answer": q_desc.get("gt_answer"),
        "gt_answer_text": q_desc.get("gt_answer_text"),
        "also_valid": q_desc.get("also_valid") or [],
        "option_map": q_desc.get("option_map") or {},
        # Asset folders of the referred object(s) — persisted so the review
        # "Referred object guide" can render 8-view previews even when the
        # object is ABSENT from the scene (find=no / count=0) or was renamed
        # by the LLM phraser (which breaks name→folder lookup).
        "queried_ref_folder": q_desc.get("queried_ref_folder"),
        "target_ref_folder": q_desc.get("target_ref_folder"),
    }

    # ── compare / marked_comparison: A/B/C badge metadata ──────────────
    # Persist per-label object name, volume, and asset folder so the review
    # page can display all three badged objects and the reference-view guide
    # can render each one.  GT is a badge letter (A/B/C = largest volume).
    if q_desc.get("question_type") == "marked_comparison":
        _mids = q_desc.get("marker_obj_ids") or {}
        _mnames = (q_desc.get("gt_object_names")
                   or q_desc.get("marker_obj_names") or {})
        _mvols = q_desc.get("gt_volumes_cm3") or {}
        # Per-label asset folder comes from the marked_sites records
        # (each site has 'label' + 'folder').
        _mfolders = {s.get("label"): s.get("folder")
                     for s in (q_desc.get("marked_sites") or [])
                     if s.get("label")}
        _moccl = {s.get("label"): s.get("occlusion")
                  for s in (q_desc.get("marked_sites") or [])
                  if s.get("label")}
        marked_labels = {}
        for _lab in ("A", "B", "C"):
            marked_labels[_lab] = {
                "name": _mnames.get(_lab),
                "volume_cm3": _mvols.get(_lab),
                "obj_folder": _mfolders.get(_lab),
                "obj_id": _mids.get(_lab),
                "occlusion": _moccl.get(_lab),
            }
        qmeta["marked_labels"] = marked_labels
        qmeta["gt_object_names"] = _mnames
        # Question polarity (largest vs smallest) — surfaced so the review UI
        # can label the task clearly and reviewers aren't misled by the tier
        # name (which always tracks the LARGEST object's occlusion rank).
        qmeta["comparison_extremum"] = q_desc.get("comparison_extremum",
                                                   "largest")
        qmeta["largest_letter"] = q_desc.get("largest_letter")
        qmeta["smallest_letter"] = q_desc.get("smallest_letter")

    manifest = {
        "schema_version": 1,
        "episode": int(episode),
        "episode_seed": int(episode_seed),
        "benchmark_version": benchmark_version,
        "image_size": (
            {"width": int(camera.width), "height": int(camera.height)}
            if camera is not None else None),
        "question": qmeta,
        "images": dict(image_files or {}),
        "snapshot": "scene_snapshot/scene_snapshot.json",
        # reviewer workflow fields (filled by the webpage; defaults here)
        "review": {
            "status": "pending",   # pending | accepted | rejected
            "reject_reason": "",
            "edited_question": None,
            "edited_target_name": None,
        },
        "objects": objects,
    }

    out_path = os.path.join(ep_dir, "review.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        pass
    return out_path


def _annotate_gt_scene(rgb, seg, q_desc, occ_stats, camera=None):
    """Annotate a scene image with target/A/B markers for debugging.

    Draws labeled bounding boxes + crosshairs on the gt_scene.png so
    humans can quickly identify which objects the question is about.

    For each question type:
      - counting / count: box+label on every instance of the target
      - existence / find: box+label on target (green if present; red if absent)
      - under_object / beneath: boxes on cover (blue), primary hidden (green),
                      and also_valid alternates (lime)
      - rearrangement: reference object (red — must NOT be removed) +
                       every GT-removal object (green boxes)
      - same_attribute / relative_size: A (orange) and B (cyan) boxes
      - interactive_counting: no annotation (scene-wide question)

    Args:
        rgb: (H, W, 3) uint8 RGB image (will be copied, not modified)
        seg: (H, W) int32 PyBullet segmentation buffer
        q_desc: question descriptor dict
        occ_stats: occlusion stats dict (has name_to_id mapping)
        camera: owg_robot Camera instance (optional, used for AABB
                projection when an object is fully occluded in `seg`)

    Returns:
        Annotated (H, W, 3) uint8 BGR image ready for cv2.imwrite
    """
    img = np.ascontiguousarray(cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR))
    H, W = img.shape[:2]

    # Colors are BGR
    COLOR_TARGET = (0, 200, 0)       # green (primary target / existence "yes")
    COLOR_ABSENT = (50, 50, 220)     # red (existence "no" — absent)
    COLOR_COVER  = (230, 120, 0)     # blue (under_object cover)
    COLOR_ALT    = (0, 255, 140)     # lime (under_object also_valid)
    COLOR_A      = (0, 200, 255)     # orange (A)
    COLOR_B      = (220, 120, 0)     # cyan-ish (B)

    def _bbox_from_seg(seg, obj_id):
        mask = (seg == obj_id)
        if not np.any(mask):
            return None
        ys, xs = np.where(mask)
        return (int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()))

    def _centroid_from_bbox(bbox):
        x0, y0, x1, y1 = bbox
        return ((y0 + y1) // 2, (x0 + x1) // 2)

    def _draw_labeled_box(img, obj_id, label, color, dashed=False,
                          show_crosshair=True):
        """Draw a labeled box around the object. Uses seg first; falls
        back to projected AABB (rendered with a dashed outline) when the
        object is completely buried / invisible.
        """
        bbox = _bbox_from_seg(seg, obj_id) if seg is not None else None
        is_buried = False
        if bbox is None:
            if camera is not None:
                bbox = _project_aabb_to_image(obj_id, camera)
                is_buried = True
            if bbox is None:
                return False

        x0, y0, x1, y1 = bbox
        thickness = 3

        if is_buried or dashed:
            # Dashed rectangle for buried objects
            dash = 10; gap = 6
            # top + bottom
            for x in range(x0, x1, dash + gap):
                cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, thickness)
                cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, thickness)
            # left + right
            for y in range(y0, y1, dash + gap):
                cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, thickness)
                cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, thickness)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

        if show_crosshair:
            cy, cx = _centroid_from_bbox(bbox)
            cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 22, 2)

        # Label with dark background above the box (or below if near top)
        tag = label + (" [buried]" if is_buried else "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.55
        th_w = 2
        (tw, th), bl = cv2.getTextSize(tag, font, fs, th_w)
        tx = x0
        ty = y0 - 6
        if ty - th < 5:
            ty = y1 + th + 8
        tx = max(2, min(tx, W - tw - 4))
        cv2.rectangle(img, (tx - 2, ty - th - 3),
                      (tx + tw + 4, ty + bl + 1), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (tx - 2, ty - th - 3),
                      (tx + tw + 4, ty + bl + 1), color, 1)
        cv2.putText(img, tag, (tx + 1, ty), font, fs, color, th_w, cv2.LINE_AA)
        return True

    resolved_qtype = q_desc.get("question_type", "")
    name_to_id = occ_stats.get("name_to_id", {})
    id_to_name = occ_stats.get("id_to_name", {})

    def _id_for(name):
        oid = name_to_id.get(name)
        if oid is None:
            oid = name_to_id.get(str(name))
        return oid

    occ = occ_stats.get("occlusion", {})

    def _fmt_occ(oid):
        v = occ.get(oid, occ.get(str(oid)))
        return f" ({v:.0%} occ)" if v is not None else ""

    if resolved_qtype in ("same_attribute", "relative_size"):
        marker_obj_ids = q_desc.get("marker_obj_ids", {})
        marker_names = q_desc.get("marker_obj_names", {})
        colors = {"A": COLOR_A, "B": COLOR_B}
        for label in ("A", "B"):
            oid = marker_obj_ids.get(label)
            if oid is None:
                continue
            name = marker_names.get(label, "?")
            _draw_labeled_box(img, oid,
                              f"{label}: {name}{_fmt_occ(oid)}",
                              colors[label])

    elif resolved_qtype in ("counting", "count"):
        # Every PRESENT instance of the target (matches the counted set).
        # Instances that rolled off-table / out-of-FoV have full_area ~0 and
        # are NOT counted, so skip them here too (else the overlay shows more
        # boxes than the GT count).
        target_name = q_desc.get("target_name")
        full_area = occ_stats.get("full_area", {})
        if target_name:
            match_ids = [oid for oid, n in id_to_name.items()
                         if n == target_name
                         and full_area.get(oid, full_area.get(str(oid), 0)) >= 100]
            for idx, oid in enumerate(match_ids):
                _draw_labeled_box(img, oid,
                                  f"TGT #{idx+1}: {target_name}{_fmt_occ(oid)}",
                                  COLOR_TARGET)

    elif resolved_qtype in ("existence", "find"):
        target_name = q_desc.get("target_name")
        gt_text = (q_desc.get("gt_answer_text") or "").lower()
        if target_name:
            oid = _id_for(target_name)
            if oid is not None:
                # Object present in scene — green box
                color = COLOR_TARGET if gt_text != "no" else COLOR_ABSENT
                tag = ("TARGET" if gt_text != "no" else "PRESENT (gt=no?)")
                _draw_labeled_box(img, oid,
                                  f"{tag}: {target_name}{_fmt_occ(oid)}",
                                  color)
            else:
                # Absent from the scene — "no" answer case
                cv2.putText(img,
                            f"TARGET (absent from scene): {target_name}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, COLOR_ABSENT, 2, cv2.LINE_AA)

    elif resolved_qtype in ("under_object", "beneath"):
        cover_name = q_desc.get("cover_object")
        hidden_name = q_desc.get("target_name")
        also_valid = q_desc.get("also_valid") or []
        if cover_name:
            oid = _id_for(cover_name)
            if oid is not None:
                _draw_labeled_box(img, oid,
                                  f"COVER: {cover_name}{_fmt_occ(oid)}",
                                  COLOR_COVER)
        if hidden_name:
            oid = _id_for(hidden_name)
            if oid is not None:
                _draw_labeled_box(img, oid,
                                  f"HIDDEN: {hidden_name}{_fmt_occ(oid)}",
                                  COLOR_TARGET)
        for alt in also_valid:
            oid = _id_for(alt)
            if oid is not None:
                _draw_labeled_box(img, oid,
                                  f"ALSO: {alt}{_fmt_occ(oid)}",
                                  COLOR_ALT)

    elif resolved_qtype == "rearrangement":
        # Reference object in RED (must NOT be removed); every GT-removal
        # object in GREEN.  IDs come straight from the descriptor (image
        # centroids were computed on the settled seg), so use them directly.
        ref_id = q_desc.get("reference_obj_id")
        ref_name = q_desc.get("target_name", "?")
        direction = q_desc.get("direction", "?")
        removal_ids = q_desc.get("gt_removal_ids", [])
        removal_names = q_desc.get("gt_removal_names", [])
        if ref_id is not None:
            _draw_labeled_box(img, int(ref_id),
                              f"REF (KEEP): {ref_name}",
                              COLOR_ABSENT)   # red
        for idx, oid in enumerate(removal_ids):
            rname = removal_names[idx] if idx < len(removal_names) else "?"
            _draw_labeled_box(img, int(oid),
                              f"REMOVE #{idx+1} ({direction}): {rname}",
                              COLOR_TARGET)   # green

    else:
        # interactive_counting / unknown — draw nothing
        pass

    return img


# Main VLM reasoning loop for a single scene
# ---------------------------------------------------------------------------

