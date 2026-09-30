"""
core.worker — Single-worker episode loop for MG-VQA-Bench.

This is the execution engine that runs the episode loop:
  1. Parses CLI arguments (via core.cli.parse_args)
  2. Loads perception/manipulation models
  3. Iterates episodes: spawn scene → generate question → run agentic/singlepass
  4. Writes crash-safe results.jsonl

Called by:
  - run.py (--workers 0, single process mode)
  - orchestrator/parallel.py (spawns as subprocess)

This file is large because it handles the full episode lifecycle including
object selection, scene spawning, question generation, Blender setup, GT
annotation, and results tracking.
"""

import os
import sys
import json
import time
import random
import contextlib
import traceback
import numpy as np
import cv2
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "third_party/graspgen"))

from owg_robot.ui import RobotEnvUI
from owg.utils.config import load_config

from models import (
    call_model, call_vlm, parse_vlm_response, VLM_MODEL_NAME,
    RateLimitError,
)
from tools import (
    load_molmo_model, load_sam3_model, SAM3_WEIGHTS,
    load_roborefer_model, set_pointing_backend,
    set_finger_offset, set_num_pc_points,
    compute_ab_marker_info, format_ab_targets_block,
)
from questions import (
    QUESTION_TYPES, QUESTION_TYPES_V2,
    AB_TYPES, AB_COORD_TYPES,
    parse_annotations_gt_lookup, parse_annotations_full_jsonl,
    parse_object_annotations_json,
    parse_obj_gt_lookup,
    _prepare_counting_scene, generate_question_descriptor,
)
from gt_visibility_filter import (
    filter_visible_obj_names, recompute_counting_target_count,
    recompute_counting_target_count_present,
)

from mgvqa.utils import (
    Tee, free_gpu_memory, safe_close_demo,
    reencode_video, MAX_ACTIONS_PER_SCENE, VIDEO_OUTPUT_FPS,
    V2_OCCLUDED_MAX_VISIBILITY,
)
from mgvqa.cli import parse_args
from mgvqa.gt_helpers import (
    _capture_sim_gt_state, _annotate_gt_scene, write_review_manifest,
)
from mgvqa.scene_snapshot import (
    save_scene_snapshot, load_snapshot_json, restore_scene_into_env,
)
from mgvqa.reasoning_loop import run_vlm_reasoning_loop
from mgvqa.singlepass import SINGLEPASS_SYSTEM_PROMPT, build_singlepass_prompt, run_singlepass


def _render_target_seg_overlay(
    rgb, seg, q_desc, occ_stats, env, out_path, camera=None,
):
    """Render a segmentation overlay showing ONLY the target objects.

    For same_attribute/relative_size: highlights A and B.
    For counting: highlights all instances of the target name.
    For existence: highlights the target object.
    For under_object: highlights the cover and hidden objects.

    Heavily occluded targets get a dashed projected-AABB box so you
    can see where they are even when invisible in the seg buffer.
    """
    import pybullet as p
    from mgvqa.gt_helpers import _project_aabb_to_image

    img = np.ascontiguousarray(cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR))
    H, W = img.shape[:2]

    qt = q_desc.get("question_type", "")
    name_to_id = (occ_stats or {}).get("name_to_id", {})

    # Determine which objects to highlight
    targets = []  # list of (obj_id, label, color_bgr)
    COLOR_A = (0, 200, 255)      # orange
    COLOR_B = (255, 180, 0)      # cyan
    COLOR_TGT = (0, 220, 0)     # green
    COLOR_COVER = (230, 120, 0)  # blue
    COLOR_HIDDEN = (0, 255, 100) # lime

    if qt in AB_TYPES:
        marker_ids = q_desc.get("marker_obj_ids", {})
        for label, color in [("A", COLOR_A), ("B", COLOR_B)]:
            oid = marker_ids.get(label)
            if oid is not None:
                targets.append((oid, label, color))
    elif qt == "counting":
        target_name = q_desc.get("target_name", "")
        count = 0
        for oid, oname in zip(env.obj_ids, env.obj_names):
            if oname == target_name:
                count += 1
                targets.append((oid, f"T{count}", COLOR_TGT))
    elif qt == "existence":
        target_name = q_desc.get("target_name", "")
        oid = name_to_id.get(target_name)
        if oid is not None:
            targets.append((oid, "TGT", COLOR_TGT))
    elif qt == "under_object":
        cover_name = q_desc.get("cover_object", "")
        hidden_name = q_desc.get("target_name", "")
        cov_id = name_to_id.get(cover_name)
        if cov_id is not None:
            targets.append((cov_id, "COVER", COLOR_COVER))
        hid_id = name_to_id.get(hidden_name)
        if hid_id is not None:
            targets.append((hid_id, "HIDDEN", COLOR_HIDDEN))

    if not targets:
        # Nothing to draw (e.g. interactive_counting)
        cv2.imwrite(out_path, img)
        return

    # Draw each target — apply color ONLY to that object's pixels
    for oid, label, color in targets:
        mask = (seg == oid)
        n_vis = int(mask.sum())

        if n_vis > 0:
            # Blend color only on this object's mask pixels
            color_arr = np.array(color, dtype=np.float32)
            img[mask] = (
                0.55 * img[mask].astype(np.float32) +
                0.45 * color_arr
            ).astype(np.uint8)

            # Draw contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 3)

            # Label at centroid
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            # Fully occluded — draw projected AABB as dashed box
            bbox = _project_aabb_to_image(oid, camera) if camera else None
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                dash, gap = 12, 8
                for x in range(x0, x1, dash + gap):
                    cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, 3)
                    cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, 3)
                for y in range(y0, y1, dash + gap):
                    cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, 3)
                    cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, 3)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            else:
                cx, cy = W // 2, H - 40

        # Draw label — short: just "A", "B", "TGT", etc.
        tag = label

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th_w = 0.6, 2
        (tw, th), bl = cv2.getTextSize(tag, font, fs, th_w)
        tx = max(2, min(cx - tw // 2, W - tw - 4))
        ty = max(th + 4, cy - 15)
        cv2.rectangle(img, (tx - 3, ty - th - 4),
                      (tx + tw + 5, ty + bl + 3), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (tx - 3, ty - th - 4),
                      (tx + tw + 5, ty + bl + 3), color, 2)
        cv2.putText(img, tag, (tx, ty), font, fs, color, th_w, cv2.LINE_AA)

    cv2.imwrite(out_path, img)


def check_vlm_text_bias(
    question: str,
    answer_options: List[str],
    gt_answer: str,
    model_name: str = VLM_MODEL_NAME,
) -> Tuple[bool, str]:
    """
    Ask the VLM the question with NO image and the constrained option list.

    If the VLM already answers correctly from text alone it is using prior knowledge
    (bias), so the episode should be skipped.

    Returns:
        (is_biased, vlm_text_answer)
        is_biased=True  → VLM got it right without seeing anything; skip episode.
        is_biased=False → VLM could not answer from text alone; run full episode.
    """
    options_str = ", ".join(answer_options)
    prompt = (
        f"Answer the following question using ONLY one of the allowed options below.\n\n"
        f"QUESTION: {question}\n\n"
        f"ALLOWED OPTIONS: {options_str}\n\n"
        f"Respond with a JSON object: {{\"answer\": \"<one of the allowed options>\"}}"
    )
    response_text = call_model(
        images=[],
        prompt=prompt,
        system_prompt="You are a question answering assistant.",
        model_name=model_name,
        temp=0.0,
        n_tokens=4096,  # reasoning models consume thinking tokens before output
    )
    parsed = parse_vlm_response(response_text)
    vlm_text_answer = parsed.get("answer", "").strip().lower()
    is_biased = (vlm_text_answer == gt_answer.strip().lower())
    return is_biased, vlm_text_answer


def run_preexperiment_bias_check(
    all_obj_names: List[str],
    gt_lookup: Dict[str, Dict[str, str]],
    color_options: List[str],
    shape_options: List[str],
    qtypes_to_check: List[str],
    model_name: str,
    cache_path: str,
) -> Dict[str, List[str]]:
    """
    Run text-only bias check for every (object, question_type) combination before
    the experiment starts, then return only the non-biased pairs.

    Results are cached to ``cache_path`` (JSON) so subsequent runs with the same
    model / question-type configuration are instant.

    Returns:
        eligible_qtypes: {obj_name: [qtype, ...]} containing only (obj, qtype)
        pairs where the VLM could NOT answer correctly from text alone.
        Objects with no eligible question type are excluded entirely.
    """
    # Load existing cache (may be partial from a previous interrupted run).
    # Cache stores every checked object: non-empty list = eligible, [] = fully biased.
    # Objects absent from the cache have never been checked.
    cache: Dict[str, List[str]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    unchecked = [obj for obj in all_obj_names if obj not in cache]

    if not unchecked:
        print(f"[bias-precheck] All {len(all_obj_names)} objects already in cache "
              f"({cache_path}). Skipping API calls.", flush=True)
        return {k: v for k, v in cache.items() if v}

    total = len(unchecked) * len(qtypes_to_check)
    already = len(all_obj_names) - len(unchecked)
    print(f"[bias-precheck] {already} objects already cached; "
          f"running {total} checks for {len(unchecked)} new objects "
          f"({qtypes_to_check}) ...", flush=True)

    done = 0
    for obj_name in unchecked:
        gt_info = gt_lookup.get(obj_name, {})
        obj_eligible = []

        for qt in qtypes_to_check:
            gt_answer = gt_info.get(qt)
            if gt_answer is None:
                done += 1
                continue

            answer_options = color_options if qt == "color" else shape_options
            obj_display = gt_info.get("display_name", obj_name.replace("_", " "))
            question, _ = generate_question(obj_name, question_type=qt,
                                            display_name=obj_display)
            is_biased, bias_ans = check_vlm_text_bias(
                question, answer_options, gt_answer, model_name=model_name
            )
            done += 1
            status = "BIASED" if is_biased else "ok"
            print(f"  [{done:>{len(str(total))}}/{total}] {obj_name:30s} ({qt:6s}): "
                  f"{status}  (vlm='{bias_ans}', gt='{gt_answer}')", flush=True)

            if not is_biased:
                obj_eligible.append(qt)

        # Store even fully-biased objects (empty list) so they are not re-checked
        cache[obj_name] = obj_eligible

        # Persist after each object so a crash does not lose progress
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    eligible_qtypes = {k: v for k, v in cache.items() if v}
    n_eligible = len(eligible_qtypes)
    n_biased_only = len(all_obj_names) - n_eligible
    print(f"[bias-precheck] Done. {n_eligible}/{len(all_obj_names)} objects eligible "
          f"({n_biased_only} removed as fully biased). Cache: {cache_path}", flush=True)

    return eligible_qtypes


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GT scene annotation helper
# ---------------------------------------------------------------------------

def parse_fail_reason(reason: str) -> str:
    """Categorise a failure reason string into a general bucket."""
    if reason is None:
        return "unknown_failure"
    r = str(reason).lower()
    if "budget exhausted" in r:
        return "action_budget_exhausted"
    if "target object" in r and "not found" in r:
        return "target_not_found"
    if "accidentally removed" in r:
        return "target_accidentally_removed"
    if "maximum vlm steps" in r:
        return "max_vlm_steps_exceeded"
    if "no answer" in r or "without a final answer" in r:
        return "no_answer_provided"
    if "parse" in r or "error" in r:
        return "vlm_parse_error"
    return "unknown_failure"

def main():
    args = parse_args()

    # Load environment config (same as demo.py)
    cfg = load_config("./config/pyb/env.yaml")
    if args.seed is not None:
        base_seed = int(args.seed)
    else:
        base_seed = random.randint(0, 2**31 - 1)
    cfg.seed = base_seed

    if args.vis is not None:
        cfg.policy.vis = bool(args.vis)
    if args.verbose is not None:
        cfg.policy.verbose = int(args.verbose)

    cfg.n_objects = int(args.n_objects)

    print(f"\n{'='*60}")
    print(f"  Agentic MQA Benchmark")
    print(f"  Mode: {args.mode} | Model: {args.model_name}")
    print(f"  Questions: {args.question_type} | Episodes: {args.n_episodes} | "
          f"Objects: {args.n_objects} | Renderer: {args.renderer}")
    print(f"  Seed: {base_seed}"
          f"{'  (user-provided)' if args.seed is not None else '  (random — save this to reproduce)'}")
    want_blend_export = bool(args.save_blend_anim)
    if want_blend_export:
        print(f"  Blend export: save_blend=True "
              f"(render offline with render_blend_to_mp4.py)")
    print(f"{'='*60}")

    # Validate: blend export requires Blender renderer
    if want_blend_export and args.renderer != "blender":
        raise ValueError(
            "--save_blend_anim requires --renderer blender")

    # Load object list + ground-truth attributes.
    # Priority: --annotations_full (new JSONL with role/final_name/final_color)
    #           → --annotations (legacy unified annotations.json)
    #           → auto-detect <asset_path>/annotations.json
    #           → obj_list.txt fallback
    target_pool = None      # set only for annotations_full path
    distractor_pool = None

    if args.annotations_full and os.path.isfile(args.annotations_full):
        # ── annotations_full pathway (target/distractor role-based) ──
        # obj_folder is stored as "subdir/obj_name" (e.g. "unified_objects/hope_obj_002")
        # so asset_path must be the PARENT of both unified_objects/ and dtc_objects/.
        asset_dirs = [
            "./owg_robot/assets/unified_objects",
            "./owg_robot/assets/dtc_objects",
        ]
        args.asset_path = "./owg_robot/assets"
        # Dispatch by extension:
        #   .json  → new format (single list, inline size_cm) e.g. object_annotations*.json
        #   .jsonl → legacy format (one obj per line, sizes from external annotations.json)
        if args.annotations_full.endswith(".json"):
            gt_lookup, target_pool, distractor_pool = parse_object_annotations_json(
                args.annotations_full,
                asset_dirs=asset_dirs,
                min_dim_cm=args.min_dim_cm,
            )
        else:
            gt_lookup, target_pool, distractor_pool = parse_annotations_full_jsonl(
                args.annotations_full,
                asset_dirs=asset_dirs,
                min_dim_cm=args.min_dim_cm,
            )
        obj_pool = target_pool + distractor_pool
        print(f"[objects] {len(gt_lookup)} unique display names, "
              f"{len(target_pool)} targets + {len(distractor_pool)} distractors "
              f"= {len(obj_pool)} total.")
    else:
        annotations_path = args.annotations
        if annotations_path is None:
            candidate = os.path.join(args.asset_path, "annotations.json")
            if os.path.isfile(candidate):
                annotations_path = candidate

        if annotations_path is not None and os.path.isfile(annotations_path):
            print(f"[objects] Loading from annotations.json: {annotations_path}")
            gt_lookup, obj_pool = parse_annotations_gt_lookup(
                annotations_path, args.asset_path,
                max_graspable_dim_cm=args.max_graspable_dim_cm,
            )
            print(f"[objects] {len(gt_lookup)} unique object names, "
                  f"{len(obj_pool)} total entries (incl. duplicates).")
        else:
            obj_list_path = os.path.join(args.asset_path, "obj_list.txt")
            print(f"[objects] Loading from obj_list.txt: {obj_list_path}")
            gt_lookup = parse_obj_gt_lookup(obj_list_path)
            obj_pool = [
                {"name": name, "obj_folder": name, "color": info["color"], "shape": info["shape"]}
                for name, info in gt_lookup.items()
            ]

    all_obj_names = list(gt_lookup.keys())

    # A marked scene bank is a frozen dataset, not a retry hint. Load every
    # package once before VLM/perception models are initialized so malformed or
    # asset-incompatible banks fail loudly and consistently across all arms.
    scene_bank_rows = None
    if getattr(args, "scene_bank", None):
        if args.question_type != "marked_comparison":
            raise ValueError("--scene_bank is valid only for --question_type "
                             "marked_comparison")
        from experiments.marked_size.scene_bank import load_scene_bank_manifest
        scene_bank_rows = load_scene_bank_manifest(args.scene_bank)
        if int(args.n_episodes) != len(scene_bank_rows):
            print(f"[scene_bank] Overriding --n_episodes={args.n_episodes} with "
                  f"the frozen bank's {len(scene_bank_rows)} scenes")
        args.n_episodes = len(scene_bank_rows)
        # A bank already defines accepted, ordered episodes. Do not apply the
        # unrelated generic benchmark subset.
        args.episode_subset = ""
        print(f"[scene_bank] Loaded {len(scene_bank_rows)} frozen marked scenes "
              f"from {os.path.abspath(args.scene_bank)}")

    # ── Frozen snapshot bank (--from_snapshot) ─────────────────────────
    # General teleport-restore replay for ALL V2 tasks (count/find/beneath/
    # compare).  Points at a rollout-shaped dir (raw snapshots_only rollout OR
    # a finalized bank).  Each restored episode rehydrates its saved GT
    # descriptor, so the eval loop scores without re-generating the question.
    # Distinct from Mika's --scene_bank (marked-only package format); the two
    # are mutually exclusive.
    snapshot_eps = None
    if getattr(args, "from_snapshot", ""):
        if scene_bank_rows is not None:
            raise ValueError("--from_snapshot and --scene_bank are mutually "
                             "exclusive (both restore frozen scenes).")
        from mgvqa.scene_snapshot import discover_snapshot_episodes
        snapshot_eps = discover_snapshot_episodes(args.from_snapshot)
        if not snapshot_eps:
            raise ValueError(f"--from_snapshot: no restorable episodes found "
                             f"under {os.path.abspath(args.from_snapshot)} "
                             f"(expected ep_*/scene_snapshot/scene_snapshot.json)")
        args.n_episodes = len(snapshot_eps)
        args.episode_subset = ""   # a bank defines its own episode set
        from collections import Counter as _Cnt
        _by = _Cnt(e["question_type"] for e in snapshot_eps)
        print(f"[from_snapshot] Loaded {len(snapshot_eps)} frozen scenes from "
              f"{os.path.abspath(args.from_snapshot)}  "
              f"(by type: {dict(_by)})")

    if len(obj_pool) < int(args.n_objects):
        raise ValueError(
            f"Need at least {args.n_objects} objects in pool, "
            f"but got {len(obj_pool)}."
        )

    # Skip bias check — assume all objects are unbiased and use the full pool.
    eligible_qtypes = None
    eligible_pool = list(obj_pool)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
    else:
        from mgvqa.output import create_run_directory
        save_dir = create_run_directory(args.model_name, args.mode, timestamp=timestamp)
        print(f"MGVQA_RUN_DIR={os.path.abspath(save_dir)}", flush=True)
    results_path = os.path.join(save_dir, "results.jsonl")

    # Save run configuration
    run_config = vars(args)
    run_config["timestamp"] = timestamp
    run_config["base_seed"] = base_seed
    if scene_bank_rows is not None:
        run_config["scene_bank"] = os.path.abspath(args.scene_bank)
        run_config["scene_bank_size"] = len(scene_bank_rows)

    # ── Few-shot exemplars (only when --few_shot is set) ──
    # Loaded once at startup; reused by every episode of the matching
    # question type. The 6 reserved exemplar episode IDs are auto-added
    # to the skip list so the model never sees an example from its own
    # eval set.
    _few_shot_bundle = None
    _few_shot_exemplar_eps: set = set()
    if getattr(args, "few_shot", False):
        if args.mode != "agentic":
            print("  ⚠ --few_shot is currently only supported for "
                  "--mode agentic; ignoring for singlepass.")
        else:
            from few_shot import (
                build_few_shot_bundle, exemplar_episode_ids,
                DEFAULT_EXEMPLARS,
            )
            print(f"\n  Loading few-shot exemplars from "
                  f"{args.few_shot_rollout} ...")
            _few_shot_bundle = build_few_shot_bundle(
                rollout_dir=args.few_shot_rollout)
            _few_shot_exemplar_eps = set(exemplar_episode_ids())
            run_config["few_shot_exemplars"] = {
                qt: int(ep) for qt, ep in DEFAULT_EXEMPLARS.items()}
            run_config["few_shot_rollout"] = args.few_shot_rollout
            print(f"  ✓ Loaded {len(_few_shot_bundle)} exemplars: "
                  f"{run_config['few_shot_exemplars']}")
            print(f"  → Episodes {sorted(_few_shot_exemplar_eps)} excluded "
                  f"from evaluation (used as exemplars).\n")

    # ── Episode subset filter (--episode_subset) ──
    # When provided, only episode indices listed in the JSON file's
    # "episodes" array are evaluated. All other slots are recorded as
    # skipped with skip_reason="not_in_subset". Use to point at
    # gold-trajectories/final_benchmark_episodes.json (the 150-task
    # filtered benchmark).
    _allowed_eps_subset: set = set()
    _have_subset = False
    if getattr(args, "episode_subset", None):
        with open(args.episode_subset, "r", encoding="utf-8") as f:
            _subset_doc = json.load(f)
        _allowed_eps_subset = {int(e) for e in _subset_doc.get("episodes", [])}
        _have_subset = True
        run_config["episode_subset_path"] = args.episode_subset
        run_config["episode_subset_size"] = len(_allowed_eps_subset)
        # Subtract reserved exemplar episodes from the subset.
        _allowed_eps_subset -= _few_shot_exemplar_eps
        run_config["episode_subset_after_exemplar_exclusion"] = len(
            _allowed_eps_subset)
        print(f"  Episode subset: {len(_allowed_eps_subset)} episodes "
              f"will be evaluated "
              f"(loaded {len(_subset_doc.get('episodes', []))} from "
              f"{args.episode_subset}, "
              f"-{len(_few_shot_exemplar_eps)} exemplars).\n")

    with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # ---- Load perception models (agentic + perception modes) ----
    # Three tool-access regimes (see docs/THREE_MODE_EVAL_PLAN.md):
    #   direct      → one image, no tools           (== old 'singlepass')
    #   perception  → perceive_scene + find_object  (SAM3 + pointer, NO GraspGen)
    #   agentic     → full loop                     (SAM3 + pointer + GraspGen)
    from mgvqa.cli import normalize_mode
    _mode = normalize_mode(args.mode)
    is_direct = (_mode == "direct")
    is_perception = (_mode == "perception")
    is_agentic = (_mode == "agentic")
    # Keep the historical name alive for the single-image dispatch branch below.
    is_singlepass = is_direct
    # Which model families each mode needs:
    need_perception_models = is_perception or is_agentic   # SAM3 + pointer VLM
    need_grasp_models = is_agentic                          # GraspGen only
    # Snapshot-only mode never calls the VLM / pointer / grasp models, so
    # skip ALL heavy model loading (SAM3, GraspGen, pointer VLM).  Each
    # episode is just spawn → settle → save artifacts.
    snapshots_only = bool(getattr(args, "snapshots_only", 0))
    if snapshots_only:
        molmo_processor = molmo_model = sam_model = None
        print("\nSnapshot-only mode — skipping ALL model loading "
              "(SAM3 / GraspGen / pointer VLM).\n")
    elif need_perception_models:
        if args.pointing_model == "roborefer":
            print("\nLoading RoboRefer-8B-SFT for object pointing...")
            molmo_processor, molmo_model = load_roborefer_model()
        elif args.pointing_model == "molmo":
            print("\nLoading Molmo-7B-D for object pointing...")
            molmo_processor, molmo_model = load_molmo_model()
        else:  # vlm
            # The pointer is PINNED to args.pointing_vlm_model
            # (default: gcp/google/gemini-3.1-pro-preview) regardless of
            # what the planner --model_name is. Keeps the pointing tool
            # consistent across every model under test.
            _pointer_mn = (
                getattr(args, "pointing_vlm_model", None)
                or "gcp/google/gemini-3.1-pro-preview")
            if _pointer_mn == args.model_name:
                print(f"\nUsing main VLM ({args.model_name}) for object "
                      f"pointing — no local pointer model loaded.")
            else:
                print(f"\nPlanner = {args.model_name}")
                print(f"Pointer = {_pointer_mn}  (fixed, --pointing_vlm_model)")
                print(f"  → pointing calls always routed to {_pointer_mn} "
                      f"so all models share the same pointing tool.")
            molmo_processor = molmo_model = None
        set_pointing_backend(
            args.pointing_model,
            vlm_model_name=(
                getattr(args, "pointing_vlm_model", None)
                or "gcp/google/gemini-3.1-pro-preview")
            if args.pointing_model == "vlm" else "",
        )
        print("Loading SAM3 model for segmentation...")
        sam_model = load_sam3_model()
        _mode_label = "Perception-only" if is_perception else "Agentic"
        print(f"Perception models loaded ({_mode_label} mode). "
              "Starting experiments.\n")
    else:
        molmo_processor = molmo_model = sam_model = None
        print("\nDirect mode (no tools) — skipping perception model loading.\n")

    # ---- Load GraspGen 6-DOF grasp model (agentic mode ONLY) ----
    # Perception mode cannot manipulate, so it skips GraspGen entirely
    # (~2-3 GB VRAM saved per worker — matters when scaling to many workers).
    graspgen_sampler = None
    if need_grasp_models and not snapshots_only:
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        print(f"Loading GraspGen from {args.graspgen_config}...")
        _gg_cfg = load_grasp_cfg(args.graspgen_config)
        graspgen_sampler = GraspGenSampler(_gg_cfg)
        # --finger_offset_deg is deprecated (R_GRIP_TO_EE handles it now).
        # Call kept only to emit a warning if the user supplied a non-zero
        # value on the command line.
        set_finger_offset(args.finger_offset_deg)
        # Set GraspGen point-cloud subsample ceiling (prevents CUDA cdist
        # blow-up on dense masks at img_size=1024 — see FIXES_FOR_April18 P0-1).
        set_num_pc_points(args.num_pc_points)
        print(f"GraspGen loaded (PC cap = {args.num_pc_points}).\n")

    results = []

    # Build episode schedule: assign a question type to each episode.
    # When --question_type is a specific type, all episodes use that type.
    # When "random", episodes are distributed equally across all 6 types.
    #
    # The schedule is ALWAYS built from the full QUESTION_TYPES list
    # (including interactive_counting), even in singlepass mode.  This
    # ensures that for the same --seed and --n_episodes the schedule is
    # identical between agentic and singlepass runs.
    # Interactive_counting episodes are then **skipped** at runtime in
    # singlepass mode — they keep their slot number so subsequent
    # episode seeds and question-type assignments are preserved.
    #
    # interactive_counting is now allowed in singlepass — occluded objects
    # make visual counting non-trivial even without manipulation.

    # Benchmark version selects the type pool (V1 6-task vs V2 4-task).
    _bench_v2 = (getattr(args, "benchmark_version", "v1") == "v2")
    _sched_types = QUESTION_TYPES_V2 if _bench_v2 else QUESTION_TYPES

    if snapshot_eps is not None:
        # Frozen snapshot replay: the schedule is exactly the discovered
        # episodes' question types (V2 vocabulary), in discovery order.  No
        # shuffle — the bank IS the fixed episode set.  ``compare`` maps to
        # ``marked_comparison`` for scene-prep just like fresh generation.
        _V1_TO_V2_SCHED = {"marked_comparison": "compare", "counting": "count",
                           "existence": "find", "under_object": "beneath"}
        episode_schedule = [
            _V1_TO_V2_SCHED.get(e["question_type"], e["question_type"])
            for e in snapshot_eps
        ]
    elif args.question_type == "random":
        # Always use the full type list so the schedule is seed-identical
        # between agentic and singlepass modes.
        all_types = list(_sched_types)  # deterministic order
        n_per_type = -(-args.n_episodes // len(all_types))  # ceil division
        episode_schedule = []
        for qt in all_types:
            episode_schedule.extend([qt] * n_per_type)
        episode_schedule = episode_schedule[:args.n_episodes]
        # Deterministic shuffle so runs are reproducible
        random.Random(base_seed).shuffle(episode_schedule)
    else:
        episode_schedule = [args.question_type] * args.n_episodes

    n_per_qt = Counter(episode_schedule)
    print(f"\n  Episode schedule ({len(episode_schedule)} total):")
    # Print the active pool first, then any extra scheduled types not in it
    # (e.g. an explicitly requested `compare` / `marked_comparison`), so the
    # summary never silently drops a type.
    _print_types = list(_sched_types) + [
        qt for qt in n_per_qt if qt not in _sched_types]
    for qt in _print_types:
        if n_per_qt[qt]:
            print(f"    {qt:<20s} {n_per_qt[qt]:>4d} episodes")
    print()

    if args.episodes:
        # Support both comma-separated (0,3,6) and ranges (180-379)
        ep_range = []
        for part in args.episodes.split(","):
            part = part.strip()
            if "-" in part and not part.startswith("-"):
                lo, hi = part.split("-", 1)
                ep_range.extend(range(int(lo), int(hi) + 1))
            else:
                ep_range.append(int(part))
        ep_range = sorted(set(ep_range))
    elif args.only_ep is not None:
        ep_range = [args.only_ep]
    else:
        ep_range = range(args.n_episodes)

    # ── Episode loop with retry support ──────────────────────────────
    # When a scene is skipped (insufficient occlusion, both targets
    # visible, etc.), retry with a different seed up to MAX_SCENE_RETRIES
    # times before giving up on that episode slot. Three-way marked triples
    # have a narrower valid-set intersection (three UV sites, a size gap, an
    # occlusion-rank tier, and a reveal oracle), so give them a fairer retry
    # budget without changing frozen benchmark types.
    MAX_SCENE_RETRIES = 3
    MARKED_SCENE_RETRIES = 12
    _ep_list = list(ep_range)
    _ep_idx = 0
    _retry_count = 0
    _episodes_completed_this_subprocess = 0
    _max_eps_per_sub = getattr(args, "max_eps_per_subprocess", 0) or 0

    # ── Excluded question types ──
    # When --exclude_qtype is set (only meaningful with --question_type=random),
    # episodes scheduled for those types are recorded as skipped without ever
    # building the scene. The schedule itself is unchanged so the seeds and
    # scene selection for non-excluded types are bit-for-bit identical to a
    # non-excluded run.
    _excluded_qtypes = set()
    _exclude_arg = (getattr(args, "exclude_qtype", "") or "").strip()
    if _exclude_arg:
        _excluded_qtypes = {
            q.strip() for q in _exclude_arg.split(",") if q.strip()
        }
        # Validate against known types (either benchmark version)
        _bad = _excluded_qtypes - set(QUESTION_TYPES) - set(QUESTION_TYPES_V2)
        if _bad:
            print(f"  ⚠ --exclude_qtype contains unknown types: "
                  f"{sorted(_bad)}; ignoring them.")
            _excluded_qtypes -= _bad
        if _excluded_qtypes:
            print(f"  Excluding question types: "
                  f"{sorted(_excluded_qtypes)} — will skip "
                  f"{sum(1 for qt in episode_schedule if qt in _excluded_qtypes)}"
                  f"/{len(episode_schedule)} episodes")

    while _ep_idx < len(_ep_list):
        ep = _ep_list[_ep_idx]
        import time
        start_time = time.time()
        demo = None  # bound below; declared here for finally-block cleanup
        _loaded_from_snapshot = False  # set True if scene restored from a snapshot
        try:
            ep_question_type = episode_schedule[ep]

            # ── Skip excluded question types ──
            # Record a 'skipped' entry to keep results.jsonl per-slot count
            # stable; do not touch any RNG so non-excluded episode seeds are
            # unaffected.
            if ep_question_type in _excluded_qtypes:
                skip_record = {
                    "episode": ep,
                    "question_type": ep_question_type,
                    "status": "skipped",
                    "skip_reason": "qtype_excluded",
                }
                with open(results_path, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(skip_record) + "\n")
                print(f"  ⏭ ep {ep:03d} ({ep_question_type}): skipped "
                      f"(--exclude_qtype)")
                _ep_idx += 1
                _retry_count = 0
                continue

            # ── Skip episodes outside the configured subset ──
            # When --episode_subset is set, only listed episodes evaluate.
            # When --few_shot is set, the 6 reserved exemplar episode IDs
            # are also unconditionally skipped (regardless of subset).
            if _have_subset and ep not in _allowed_eps_subset:
                skip_reason = (
                    "few_shot_exemplar"
                    if ep in _few_shot_exemplar_eps else "not_in_subset")
                skip_record = {
                    "episode": ep,
                    "question_type": ep_question_type,
                    "status": "skipped",
                    "skip_reason": skip_reason,
                }
                with open(results_path, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(skip_record) + "\n")
                print(f"  ⏭ ep {ep:03d} ({ep_question_type}): skipped "
                      f"({skip_reason})")
                _ep_idx += 1
                _retry_count = 0
                continue
            elif (not _have_subset
                  and ep in _few_shot_exemplar_eps):
                # No subset filter, but few-shot is on → still skip the
                # 6 exemplar episodes so they're never evaluated.
                skip_record = {
                    "episode": ep,
                    "question_type": ep_question_type,
                    "status": "skipped",
                    "skip_reason": "few_shot_exemplar",
                }
                with open(results_path, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(skip_record) + "\n")
                print(f"  ⏭ ep {ep:03d} ({ep_question_type}): skipped "
                      f"(few_shot_exemplar)")
                _ep_idx += 1
                _retry_count = 0
                continue

            # interactive_counting now runs in singlepass too.

            # ── V2 → V1 scene-prep/gate mapping ────────────────────────
            # V2 tasks reuse V1 scene-construction + occlusion-gate logic.
            # ``_effective_qtype`` drives spawning + gating; the real V2
            # name (``ep_question_type``) still drives question generation
            # and scoring.
            #   count         → counting  (targets spawned first, buried)
            #   find          → existence (uniform sample; no A/B gate)
            #   beneath       → under_object (cover-hidden pairs + gate)
            #   compare       → marked_comparison (3 badged targets + gate)
            #   rearrangement → (uniform sample; no occlusion gate)
            _V2_TO_V1_SCENE = {
                "count": "counting",
                "find": "existence",
                "beneath": "under_object",
                "compare": "marked_comparison",  # 3-way badge task
                "rearrangement": "rearrangement",  # handled explicitly
            }
            _effective_qtype = _V2_TO_V1_SCENE.get(
                ep_question_type, ep_question_type)

            # A frozen bank (Mika's standalone marked replay) pins every
            # physical scene and its visual-nuisance seed.  Normal generation
            # (incl. our unified `compare` snapshot path) uses retry-shifted
            # seeds; bank replay never retries into a different scene.
            bank_row = None
            snap_ep = None
            if snapshot_eps is not None:
                if _retry_count:
                    raise RuntimeError("Frozen snapshot replay must not retry "
                                       "into a different physical scene")
                snap_ep = snapshot_eps[ep]
                _snap_json = load_snapshot_json(snap_ep["snapshot_dir"])
                episode_seed = int(_snap_json.get(
                    "replay_seed", _snap_json.get("episode_seed",
                                                  base_seed + 1000 * ep)))
            elif scene_bank_rows is not None:
                if _retry_count:
                    raise RuntimeError("Frozen scene-bank replay must not retry "
                                       "into a different physical scene")
                bank_row = scene_bank_rows[ep]
                episode_seed = int(bank_row["package"]["scene"].get(
                    "replay_seed", base_seed + 1000 * ep))
            else:
                # Deterministic episode seed (retry offset → different scene)
                episode_seed = base_seed + 1000 * ep + _retry_count * 137

            # Option B fallback: after MAX_SCENE_RETRIES normal attempts for
            # under_object, the NEXT retry uses targeted spawn (hidden target
            # placed first at bottom, distractors dropped on top).
            # Retry flow: 0,1,2 = normal (gate strict); 3 = targeted (gate relaxed)
            _force_under_object_spawn = (
                _effective_qtype == "under_object"
                and _retry_count >= MAX_SCENE_RETRIES
            )

            # ── V2 find: yes/no polarity + buried-target spawn ─────────
            # find asks "Is there an X?"; the answer is yes (X present) or
            # no (X absent).  Polarity is deterministic per episode index
            # (even→yes, odd→no) so a full schedule is ~50/50 and stable
            # across retries.  For a "yes" episode we bury the queried
            # target (spawn FIRST/bottom, fillers on top) and later gate on
            # its occlusion — same bar as beneath.  "no" needs no burial.
            # Polarity: 'mixed' (default) = even→yes / odd→no; 'yes'/'no'
            # force every find episode one way (used to top up the scarcer
            # yes half, which loses ~half to the occlusion gate).
            _find_pol = getattr(args, "find_polarity", "mixed")
            if _find_pol == "yes":
                _find_yes = (_bench_v2 and ep_question_type == "find")
            elif _find_pol == "no":
                _find_yes = False
            else:
                _find_yes = (_bench_v2 and ep_question_type == "find"
                             and ep % 2 == 0)
            _find_target_name = None  # set when we pre-bury a find-yes target
            _spawn_drop_heights = None  # per-object drop plan (counting only)
            _cover_over_target_count = 0  # counting: #target instances (big cover targets one)

            # Select objects for this episode
            ep_rng = random.Random(episode_seed)
            n_objects = int(args.n_objects)
            counting_target_name = None
            counting_target_count = None
            _blender_scene_rgb = None  # will hold post-walls Blender render

            if snap_ep is not None:
                # Snapshot order is the settled scene order (audit only); the
                # actual bodies are teleport-restored below from the snapshot.
                _snap_objects = _snap_json.get("objects", [])
                scene_obj_names = [o["name"] for o in _snap_objects]
                scene_obj_folders = [o["folder"] for o in _snap_objects]
                selected_obj_names = list(scene_obj_names)
                selected_obj_folders = list(scene_obj_folders)
            elif bank_row is not None:
                # Package order is the settled scene order, retained solely for
                # reproducibility/audit; actual bodies are instantiated below.
                _bank_objects = bank_row["package"]["scene"]["objects"]
                scene_obj_names = [o["name"] for o in _bank_objects]
                scene_obj_folders = [o["folder"] for o in _bank_objects]
                selected_obj_names = list(scene_obj_names)
                selected_obj_folders = list(scene_obj_folders)
            elif _force_under_object_spawn:
                # ── Option B fallback: targeted under_object spawn ──
                # Place the hidden target FIRST (spawned at bottom of pile),
                # then drop N-1 objects on top so it gets buried.
                # Do NOT shuffle — spawn order is critical.
                print(f"  ⚠ under_object: using targeted spawn fallback "
                      f"(retry {_retry_count})")
                _src = target_pool if target_pool is not None else eligible_pool
                _hidden_entry = ep_rng.choice(_src)
                _filler_src = distractor_pool if distractor_pool is not None else eligible_pool
                _fillers = ep_rng.sample(
                    [e for e in _filler_src if e["obj_folder"] != _hidden_entry["obj_folder"]],
                    k=min(n_objects - 1, len(_filler_src) - 1))
                # Hidden target FIRST (bottom), then fillers on top
                _all = [_hidden_entry] + _fillers
                scene_obj_names = [e["name"] for e in _all]
                scene_obj_folders = [e["obj_folder"] for e in _all]
                selected_obj_names = list(scene_obj_names)
                selected_obj_folders = list(scene_obj_folders)
                # NOTE: do NOT shuffle — hidden target must be spawned first
            elif _find_yes:
                # ── find-yes: bury the queried target ──
                # Same targeted spawn as under_object's Option B: the target
                # goes FIRST (bottom) so fillers dropped on top occlude it.
                # The occlusion gate below enforces visibility < the shared
                # V2 bar; we remember its name to query it directly.
                _src = target_pool if target_pool is not None else eligible_pool
                _tgt_entry = ep_rng.choice(_src)
                _filler_src = distractor_pool if distractor_pool is not None else eligible_pool
                _fillers = ep_rng.sample(
                    [e for e in _filler_src if e["obj_folder"] != _tgt_entry["obj_folder"]],
                    k=min(n_objects - 1, len(_filler_src) - 1))
                _all = [_tgt_entry] + _fillers
                scene_obj_names = [e["name"] for e in _all]
                scene_obj_folders = [e["obj_folder"] for e in _all]
                selected_obj_names = list(scene_obj_names)
                selected_obj_folders = list(scene_obj_folders)
                _find_target_name = _tgt_entry["name"]
                # NOTE: do NOT shuffle — buried target must be spawned first
            elif _effective_qtype == "counting":
                # Counting: choose target first, then build scene with controlled
                # count 0-5.  Target instances are placed FIRST in spawn order so
                # they settle at the bottom and are occluded by later distractors.
                # When using annotations_full, the counting target MUST come
                # from target_pool (role='target') and fillers from
                # distractor_pool (role='distractor').  Otherwise, a distractor
                # could accidentally be chosen as the thing being counted.
                scene_obj_names, scene_obj_folders, counting_target_name, counting_target_count = \
                    _prepare_counting_scene(
                        eligible_pool, n_objects, ep_rng,
                        target_pool=target_pool,
                        distractor_pool=distractor_pool,
                    )
                selected_obj_names = list(scene_obj_names)
                selected_obj_folders = list(scene_obj_folders)
                # NOTE: do NOT shuffle — spawn order is intentional
                # ── Drop-height plan (inside the 0.30 m containment walls):
                # targets at x, fillers at x/2, big cover highest (spawned
                # last → lands on top).  Targets spawn FIRST so they settle at
                # the base; the later fillers + big cover pile on and bury
                # them.  Walls keep everything contained (no scatter).
                _COUNT_DROP_X = 0.14   # meters above table for target instances
                if counting_target_count and counting_target_count > 0:
                    _nc = int(counting_target_count)
                    _nf = len(scene_obj_names) - _nc
                    _plan = [_COUNT_DROP_X] * _nc + [_COUNT_DROP_X / 2.0] * _nf
                    if _nf >= 1:   # big cover is the LAST filler → drop highest
                        _plan[-1] = _COUNT_DROP_X + 0.10
                        # Big cover drops onto a settled target instance (the
                        # first _nc spawned) so it reliably buries one.
                        _cover_over_target_count = _nc
                    _spawn_drop_heights = _plan
            elif target_pool is not None:
                # ── annotations_full path: pick targets + distractors ──
                # For same_attribute / relative_size we need ≥2 targets;
                # for existence / under_object ≥1. Pick 2 to cover all types.
                n_targets = min(2, len(target_pool), n_objects)
                n_distractors = n_objects - n_targets
                picked_targets = ep_rng.sample(target_pool, k=n_targets)
                picked_distractors = ep_rng.sample(
                    distractor_pool, k=min(n_distractors, len(distractor_pool)))
                selected_entries = picked_targets + picked_distractors
                selected_obj_names = [e["name"] for e in selected_entries]
                selected_obj_folders = [e["obj_folder"] for e in selected_entries]

                # Shuffle spawn order (keep name↔folder pairing intact)
                order_rng = random.Random(episode_seed)
                paired = list(zip(selected_obj_names, selected_obj_folders))
                order_rng.shuffle(paired)
                scene_obj_names = [n for n, _ in paired]
                scene_obj_folders = [f for _, f in paired]
            else:
                # ── Legacy path: uniform random sample ──
                selected_entries = ep_rng.sample(eligible_pool, k=n_objects)
                selected_obj_names = [e["name"] for e in selected_entries]
                selected_obj_folders = [e["obj_folder"] for e in selected_entries]

                # Shuffle spawn order (keep name↔folder pairing intact)
                order_rng = random.Random(episode_seed)
                paired = list(zip(selected_obj_names, selected_obj_folders))
                order_rng.shuffle(paired)
                scene_obj_names = [n for n, _ in paired]
                scene_obj_folders = [f for _, f in paired]

            # Seed global RNGs for PyBullet spawning
            random.seed(episode_seed)
            np.random.seed(episode_seed)

            # Create episode directory
            ep_dirname = f"ep_{ep:03d}_{ep_question_type}"
            ep_dir = os.path.join(save_dir, ep_dirname)
            os.makedirs(ep_dir, exist_ok=True)
            spawn_dir = os.path.join(ep_dir, "spawn") if args.save_spawn_screenshots else None

            # Save selected objects for reproducibility
            with open(os.path.join(ep_dir, "selected_objects.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "episode": ep,
                    "episode_seed": episode_seed,
                    "question_type": ep_question_type,
                    "selected_obj_names": selected_obj_names,
                    "selected_obj_folders": selected_obj_folders,
                    "scene_bank_scene_id": (
                        bank_row["package"]["scene"].get("scene_id")
                        if bank_row is not None else None),
                    "scene_bank_scene_file": (
                        bank_row["package"]["scene_file"]
                        if bank_row is not None else None),
                }, f, indent=2)

            # Configure for this episode
            cfg.seed = int(episode_seed)
            # Frozen replay (scene_bank OR snapshot) starts with an EMPTY
            # dynamic scene, then restores the stored rigid bodies at their
            # accepted post-settle poses — no spawn, no walls, no settle.
            _frozen_replay = (bank_row is not None) or (snap_ep is not None)
            cfg.n_objects = 0 if _frozen_replay else int(args.n_objects)

            # Create fresh simulation
            _use_planes = bool(args.use_planes) and not _frozen_replay
            _use_basket = bool(args.use_basket) and not _use_planes and not _frozen_replay
            _tspawn0 = time.time()
            demo = RobotEnvUI(
                cfg,
                spawn_dir=spawn_dir,
                selected_obj_names=scene_obj_names,
                selected_obj_folders=scene_obj_folders,
                asset_path=args.asset_path,
                use_basket=_use_basket,
                basket_scale=float(args.basket_scale),
                basket_wall_height=float(args.basket_wall_height),
                use_planes=_use_planes,
                two_pass_spawn=bool(args.two_pass_spawn),
                spawn_drop_heights=_spawn_drop_heights,
                cover_over_target_count=_cover_over_target_count,
            )
            _tspawn1 = time.time()
            print(f"  [timing] PyBullet spawn: {_tspawn1-_tspawn0:.2f}s")
            if bank_row is not None:
                from experiments.marked_size.scene_bank import instantiate_scene_package
                _treplay0 = time.time()
                instantiate_scene_package(demo, bank_row["package"], args.asset_path)
                print(f"  [timing] Frozen scene restore: "
                      f"{time.time()-_treplay0:.2f}s "
                      f"({len(demo.env.obj_ids)} objects; "
                      f"scene_id={bank_row['package']['scene']['scene_id']})",
                      flush=True)

            # Assign GraspGen sampler if loaded
            if graspgen_sampler is not None:
                demo.graspgen_sampler = graspgen_sampler

            # Enable Blender rendering if requested
            if args.renderer == "blender":
                # Tell the Blender renderer which PyBullet bodies are
                # infrastructure (robot / gripper / stand / trays /
                # tables) so it can hide them by ID — NOT by heuristic
                # name matching.  Without this, any spawned asset whose
                # name contains a robot keyword gets mis-hidden
                # (see 2026-04-21 panda_android_figure regression).
                env_ = demo.env
                infra_ids = [
                    getattr(env_, "planeID", None),
                    # NOTE: tableID (main table) intentionally NOT included.
                    # It must be visible in Blender so the wood texture can
                    # be applied to it. Only the target_table (x>0.5) and
                    # other robot infrastructure should be hidden.
                    getattr(env_, "target_table_id", None),
                    getattr(env_, "target_id", None),    # tray
                    getattr(env_, "UR5Stand_id", None),
                    getattr(env_, "robot_id", None),
                    getattr(env_, "basket_id", None),
                ]
                demo.env.camera.set_blender_infrastructure_ids(
                    [i for i in infra_ids if i is not None])
                # Pin the table-texture pick to the episode seed so the
                # same overall --seed yields the same texture sequence
                # regardless of how much global RNG state is consumed
                # by other agentic code paths (perception, prompts, etc.)
                # between episodes.  This makes demo.py and demo_new.py
                # select the same textures for a given seed.
                demo.env.camera.set_table_texture_seed(episode_seed)
                # Lighting modes:
                #   easy — flat ambient (no HDRI, no shadows)
                #   hard — indoor HDRI with directional light (cast shadows)
                _lighting = ("pybullet_match" if args.lighting_mode == "easy"
                             else "random_indoor_hdri")
                _add_dir = False  # No directional Sun lamp in any mode
                _tblend0 = time.time()
                _blender_port = args.blender_port
                if args.worker_id is not None:
                    _blender_port = 9876 + args.worker_id
                demo.env.camera.enable_blender_rendering(
                    physics_client_id=demo.env.physicsClient,
                    engine="EEVEE",
                    lighting=_lighting,
                    # A frozen package carries one exact table material, so it
                    # must be applied regardless of this legacy rollout flag.
                    random_table_texture=(bool(args.random_table_texture)
                                          or _frozen_replay),
                    add_directional_light=_add_dir,
                    port=_blender_port,
                    quiet=True,
                )
                if bank_row is not None:
                    # The package owns a copy of the precise CCO material used
                    # at generation time. Do not depend on the evaluator's
                    # mutable local texture cache or only a selection seed.
                    from experiments.marked_size.scene_bank import packaged_table_texture_path
                    demo.env.camera.set_table_texture_path(
                        packaged_table_texture_path(bank_row["package"]))
                _tblend1 = time.time()
                print(f"  [timing] Blender enable: {_tblend1-_tblend0:.2f}s")
                # No prewarm render — the post-walls render will be the
                # first render, importing all meshes in one shot.  Walls
                # are removed from PyBullet before the first render so
                # they are never imported into Blender at all.
                _tprewarm0 = _tprewarm1 = _tblend1  # zero duration

            if bank_row is not None:
                # The renderer must be enabled first so the texture update is
                # sent to both PyBullet and Blender before any VLM-facing frame.
                from experiments.marked_size.scene_bank import apply_packaged_badges
                apply_packaged_badges(demo, bank_row["package"])

            # ── Snapshot teleport-restore (--from_snapshot) ────────────
            # Load every saved URDF at its settled pose (no drop, no settle),
            # re-apply the packaged table texture + any A/B/C badge textures to
            # BOTH renderers, and rehydrate the GT descriptor with the fresh
            # body IDs.  Must run AFTER Blender is enabled so texture pushes
            # reach the renderer.  ``_restored_q_desc`` feeds the descriptor
            # branch below (bypasses fresh question generation).
            _restored_q_desc = None
            if snap_ep is not None:
                _trs0 = time.time()
                _restore_info = restore_scene_into_env(
                    demo, _snap_json,
                    snapshot_dir=snap_ep["snapshot_dir"],
                    asset_path=args.asset_path,
                    verbose=bool(getattr(args, "verbose", 1)),
                )
                _restored_q_desc = _restore_info.get("question_descriptor")
                _loaded_from_snapshot = True
                print(f"  [timing] Snapshot restore: {time.time()-_trs0:.2f}s "
                      f"({len(demo.env.obj_ids)} objects; "
                      f"scene={snap_ep['ep_dirname']})", flush=True)
                if _restored_q_desc is None:
                    print(f"  ⚠ snapshot {snap_ep['ep_dirname']} has NO "
                          f"annotation.question_descriptor — cannot score "
                          f"(regenerate the bank with the annotation block).")

            # ── Remove containment walls (if using planes) ──────────────
            # Remove walls before the experiment so robot arm and push
            # tool can operate freely.
            _did_walls = demo.env.has_containment_walls()
            _tw0 = _tw1 = _tw2 = _tr0 = _tr1 = time.time()
            if _did_walls:
                _tw0 = time.time()
                demo.env.remove_containment_walls()
                _tw1 = time.time()
                # Settle after wall removal — objects that were resting
                # against walls need time to reach equilibrium.  With 15
                # objects the pile is dense and takes longer to stabilize.
                # Use velocity-based settle (not fixed step count) to avoid
                # objects that appear "hanging" and fall during later steps.
                _vis_saved = demo.env.vis
                demo.env.vis = False
                demo.env.wait_until_all_still(max_wait_epochs=2000)
                demo.env.vis = _vis_saved
                demo.env.remove_out_of_bounds_objs(width=0.8)
                demo.env.update_obj_states()
                _tw2 = time.time()
                print(f"  [timing] wall removal: {_tw1-_tw0:.2f}s, settle: {_tw2-_tw1:.2f}s")

                # Save clean scene image (after walls removed + settled).
                if args.renderer == "blender":
                    _tr0 = time.time()
                    rgb_after, _, _ = demo.env.camera.get_cam_img()
                    _tr1 = time.time()
                    print(f"  [timing] post-walls render: {_tr1-_tr0:.2f}s")

                    # ── Degenerate-frame guard ─────────────────────────
                    # Rarely (~3% of scenes) Blender/EEVEE returns a blank
                    # (fully- or partially-uniform grey) frame even after the
                    # renderer's own re-sync retries — a reproducible render
                    # failure tied to a specific restored scene state, NOT a
                    # data problem (PyBullet renders the same scene fine).
                    # A grey "exact frame" is useless to the VLM / reviewer,
                    # so treat it exactly like the occlusion gate: retry with
                    # a NEW seed (→ a different scene that renders cleanly),
                    # and only skip after MAX_SCENE_RETRIES so we still reach
                    # n_episodes valid candidates.
                    _bad_frame, _bad_why = False, ""
                    try:
                        from robo_blender import BlenderRenderer as _BR
                        _bad_frame, _bad_why = _BR._frame_is_degenerate(rgb_after)
                    except Exception as _degerr:
                        print(f"  ⚠ degenerate-frame check failed "
                              f"({_degerr!r}); assuming frame OK")
                    if _bad_frame:
                        print(f"  ⚠ SKIPPED ep {ep} ({ep_question_type}): "
                              f"Blender render degenerate ({_bad_why}) — "
                              f"blank/grey frame after renderer retries.")
                        safe_close_demo(demo)
                        demo = None
                        free_gpu_memory()
                        if _retry_count < MAX_SCENE_RETRIES:
                            _retry_count += 1
                            print(f"  ↻ Retry {_retry_count}/{MAX_SCENE_RETRIES} "
                                  f"for ep {ep} (new seed: "
                                  f"{base_seed + 1000*ep + _retry_count*137})")
                            continue  # retry same ep with a new seed
                        # All retries exhausted — record skip + advance
                        skip_record = {
                            "episode": ep,
                            "question_type": ep_question_type,
                            "status": "skipped",
                            "skip_reason": "blender_render_degenerate",
                            "detail": _bad_why,
                            "retry_count": _retry_count,
                        }
                        with open(results_path, "a", encoding="utf-8") as rf:
                            rf.write(json.dumps(skip_record) + "\n")
                        _ep_idx += 1
                        _retry_count = 0
                        continue

                    after_path = os.path.join(ep_dir, "after_walls_topdown.png")
                    cv2.imwrite(after_path,
                                cv2.cvtColor(rgb_after, cv2.COLOR_RGB2BGR))
                    # Store for reuse as _gt_scene.png (same image, avoids re-render)
                    _blender_scene_rgb = rgb_after

            # ── Frozen replay clean frame (no walls were present) ──────
            # scene_bank / snapshot replay skips the wall-removal block above,
            # so render the VLM-facing clean top-down frame here (badges +
            # table texture already applied).  Mirrors the post-walls capture.
            if _frozen_replay and args.renderer == "blender" \
                    and _blender_scene_rgb is None:
                _tr0 = time.time()
                rgb_after, _, _ = demo.env.camera.get_cam_img()
                print(f"  [timing] frozen-replay render: {time.time()-_tr0:.2f}s")
                after_path = os.path.join(ep_dir, "after_walls_topdown.png")
                cv2.imwrite(after_path,
                            cv2.cvtColor(np.asarray(rgb_after[:, :, :3]),
                                         cv2.COLOR_RGB2BGR))
                _blender_scene_rgb = np.asarray(rgb_after[:, :, :3])

            # Initialize motion recording if blend export is requested
            if want_blend_export and not is_singlepass:
                demo.env.start_motion_recording(playback_fps=30)

            ## EARLY STOP — timing summary
            # end_time = time.time()
            # print()
            # print("=" * 60)
            # print("  SCENE SETUP TIMING SUMMARY")
            # print("=" * 60)
            # print(f"  PyBullet spawn ({len(demo.env.obj_ids)} objects):   {_tspawn1 - _tspawn0:6.2f}s")
            # if args.renderer == "blender":
            #     print(f"  Blender enable:                {_tblend1 - _tblend0:6.2f}s")
            #     if _tprewarm1 > _tprewarm0:
            #         print(f"  Blender prewarm:               {_tprewarm1 - _tprewarm0:6.2f}s")
            #     else:
            #         print(f"  Blender prewarm:               (skipped — deferred to first render)")
            # if _did_walls:
            #     print(f"  Wall removal:                  {_tw1 - _tw0:6.2f}s")
            #     print(f"  Physics settle:                {_tw2 - _tw1:6.2f}s")
            #     if args.renderer == "blender":
            #         print(f"  Post-walls render (1st render):{_tr1 - _tr0:6.2f}s")
            # print("-" * 60)
            # print(f"  TOTAL:                         {end_time - start_time:6.2f}s")
            # print("=" * 60)
            # import sys
            # sys.exit(0)

            # ── Compute occlusion stats (AFTER spawn + settle) ─────────
            # Target selection is done POST-SPAWN: we pick targets based
            # on which objects ended up most occluded in the physical
            # simulation.  This ensures the benchmark asks about objects
            # that are genuinely hard to see — not ones that happened
            # to land on top.
            _, occ_stats = demo.pick_most_occluded_target()

            # ── Minimum occlusion gate ─────────────────────────────────
            # For non-counting types, require at least one object to be
            # meaningfully occluded.  If the scene is too "easy" (all
            # objects clearly visible), skip and try the next episode
            # seed.  This ensures the agentic benchmark only tests scenes
            # where manipulation would actually help.
            #
            # Threshold logic:
            #   - A/B marker visibility threshold (compute_ab_marker_info)
            #     is 30% by default — an object's A/B point goes blank
            #     when visibility < 30% (i.e. occlusion ≥ 70%).
            #   - We lower the SCENE gate to 60% so we don't skip too
            #     many scenes.  An object at 60% occlusion may still get
            #     a visible A/B point, but the scene is still challenging
            #     enough to require some manipulation.
            #   - For existence / under_object: lighter threshold (≥25%)
            #     since these don't rely on both points being visible.
            _MIN_OCCLUSION_AB = 0.60   # ≥60% — moderate challenge
            _MIN_OCCLUSION_OTHER = 0.25  # ≥25% — some challenge in the scene
            # marked_comparison has its own stricter *label-level* gate inside
            # prepare_marked_episode(): one selected badge must be unreadable
            # now and readable after clutter is parked. A generic object-level
            # >=60% gate would discard valid manipulation scenes where only the
            # painted surface is covered, so it is deliberately not included.
            _SKIP_TYPES = {"existence", "same_attribute", "relative_size",
                            "under_object"}

            # V2 note: gate on the EFFECTIVE type — find→existence,
            # beneath→under_object are gated; rearrangement is NOT gated
            # (it needs a spread-out scene, not an occluded pile).
            # Frozen replay (snap_ep): the scene was already accepted at
            # generation time — never re-gate/skip it.
            if _effective_qtype in _SKIP_TYPES and snap_ep is None:
                occ_values = list(occ_stats.get("occlusion", {}).values())
                max_occ = max(occ_values) if occ_values else 0.0

                # Choose threshold based on question type.  AB_TYPES =
                # (same_attribute, relative_size, marked_comparison) — the
                # multi-target A/B(/C) family that needs a stricter occlusion
                # floor.  ``compare`` maps to ``marked_comparison`` via
                # ``_effective_qtype``.
                if _effective_qtype in AB_TYPES:
                    _min_occ = _MIN_OCCLUSION_AB
                else:
                    _min_occ = _MIN_OCCLUSION_OTHER

                n_above = sum(1 for v in occ_values if v >= _min_occ)
                if max_occ < _min_occ:
                    # Scene is too easy — not enough occlusion for a fair test
                    print(f"  ⚠ SKIPPED ep {ep} ({ep_question_type}): "
                          f"insufficient occlusion — max_occ={max_occ:.1%}, "
                          f"need ≥{_min_occ:.0%}. "
                          f"Objects: {list(demo.env.obj_names)}")
                    safe_close_demo(demo)
                    demo = None
                    free_gpu_memory()
                    if _retry_count < MAX_SCENE_RETRIES:
                        _retry_count += 1
                        print(f"  ↻ Retry {_retry_count}/{MAX_SCENE_RETRIES} "
                              f"for ep {ep} (new seed: "
                              f"{base_seed + 1000*ep + _retry_count*137})")
                        continue  # retry same ep (don't advance _ep_idx)
                    # All retries exhausted — record skip
                    skip_record = {
                        "episode": ep,
                        "question_type": ep_question_type,
                        "status": "skipped",
                        "skip_reason": "insufficient_occlusion",
                        "retry_count": _retry_count,
                        "max_occlusion": round(max_occ, 3),
                        "threshold": _min_occ,
                        "n_objects": len(demo.env.obj_ids) if demo else 0,
                    }
                    with open(results_path, "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(skip_record) + "\n")
                    _ep_idx += 1
                    _retry_count = 0
                    continue
                else:
                    print(f"  ✓ Occlusion gate passed: max_occ={max_occ:.1%}, "
                          f"{n_above}/{len(occ_values)} objects ≥{_min_occ:.0%} "
                          f"(threshold for {ep_question_type})")

            # Compute cover-hidden pairs for under_object questions
            # Uses alpha-transparency rendering: hide each candidate cover,
            # check which objects gain visible area → real spatial "under" pairs.
            # V2 "beneath" demands a STRICT pair: the cover must hide ≥70% of
            # the target's own silhouette (a barely-clipped edge no longer
            # qualifies), so the answer genuinely requires removing the cover.
            _min_cover_frac = 0.70 if (_bench_v2 and ep_question_type == "beneath") else 0.0
            cover_hidden_pairs = demo.compute_cover_hidden_pairs(
                occ_stats=occ_stats, min_cover_fraction=_min_cover_frac)
            if cover_hidden_pairs and bool(args.verbose):
                print(f"[under_object] Found {len(cover_hidden_pairs)} cover→hidden pairs"
                      f" (top: {cover_hidden_pairs[0]['cover_name']} → "
                      f"{cover_hidden_pairs[0]['hidden_name']}, "
                      f"reveal={cover_hidden_pairs[0]['reveal_area']}px)")

            # ── Under-object quality gate (Option A+D) ─────────────────
            # For under_object episodes, require a genuinely hidden object:
            #   1. reveal_area ≥ 500 px (substantial, not a sliver)
            #   2. hidden object initial visibility < 30%
            # If using targeted fallback spawn (Option B), use relaxed
            # thresholds since we forced a pile — accept any valid pair.
            _MIN_UNDER_REVEAL = 200 if _force_under_object_spawn else 500
            _MAX_HIDDEN_VISIBILITY = (
                0.50 if _force_under_object_spawn else V2_OCCLUDED_MAX_VISIBILITY)
            # V2 "beneath": additionally require the chosen cover to hide
            # ≥70% of the target's silhouette.  cover_hidden_pairs is already
            # filtered at source (min_cover_fraction), but enforce here too so
            # the gate is explicit and robust to any relaxation upstream.
            _MIN_COVER_FRAC_GATE = 0.70 if (_bench_v2 and ep_question_type == "beneath") else 0.0
            if _effective_qtype == "under_object" and snap_ep is None:
                _valid_pair = None
                if cover_hidden_pairs:
                    for _cp in cover_hidden_pairs:
                        _h_id = _cp["hidden_id"]
                        _h_vis_area = occ_stats.get("visible_area", {}).get(
                            _h_id, occ_stats.get("visible_area", {}).get(str(_h_id), 0))
                        _h_full_area = occ_stats.get("full_area", {}).get(
                            _h_id, occ_stats.get("full_area", {}).get(str(_h_id), 1))
                        _h_visibility = _h_vis_area / max(_h_full_area, 1)
                        _cp_frac = _cp.get("cover_fraction", 1.0)
                        if (_cp["reveal_area"] >= _MIN_UNDER_REVEAL
                                and _h_visibility < _MAX_HIDDEN_VISIBILITY
                                and _cp_frac >= _MIN_COVER_FRAC_GATE):
                            _valid_pair = _cp
                            break

                if _valid_pair is None:
                    _top_reveal = cover_hidden_pairs[0]["reveal_area"] if cover_hidden_pairs else 0
                    print(f"  ⚠ under_object: no qualifying pair "
                          f"(top reveal={_top_reveal}px, need ≥{_MIN_UNDER_REVEAL}px "
                          f"with hidden vis < {_MAX_HIDDEN_VISIBILITY:.0%})")
                    safe_close_demo(demo)
                    demo = None
                    free_gpu_memory()
                    # Allow MAX_SCENE_RETRIES normal + 1 targeted attempt
                    _max_under = MAX_SCENE_RETRIES + 1
                    if _retry_count < _max_under:
                        _retry_count += 1
                        if _retry_count > MAX_SCENE_RETRIES:
                            print(f"  ↻ Retry {_retry_count}/{_max_under} "
                                  f"(targeted spawn fallback)")
                        else:
                            print(f"  ↻ Retry {_retry_count}/{_max_under}")
                        continue
                    # Targeted spawn also failed — skip this episode
                    skip_record = {
                        "episode": ep,
                        "question_type": ep_question_type,
                        "status": "skipped",
                        "skip_reason": "under_object_no_valid_pair",
                        "retry_count": _retry_count,
                    }
                    with open(results_path, "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(skip_record) + "\n")
                    _ep_idx += 1
                    _retry_count = 0
                    continue
                else:
                    _spawn_mode = "targeted" if _force_under_object_spawn else "natural"
                    if bool(args.verbose):
                        print(f"  ✓ under_object ({_spawn_mode}): valid pair — "
                              f"{_valid_pair['cover_name']} covers "
                              f"{_valid_pair['hidden_name']} "
                              f"(reveal={_valid_pair['reveal_area']}px)")

            # ── find-yes occlusion gate ────────────────────────────────
            # The buried target must be genuinely occluded — visibility below
            # the SHARED V2 bar (same value beneath uses for its hidden
            # object).  Otherwise the target is clearly visible and the
            # question is trivial.  Fail → retry with a new seed, then skip.
            if _find_yes and _find_target_name is not None and snap_ep is None:
                _ftid = occ_stats.get("name_to_id", {}).get(_find_target_name)
                _f_vis = occ_stats.get("visible_area", {}).get(
                    _ftid, occ_stats.get("visible_area", {}).get(str(_ftid), 0))
                _f_full = occ_stats.get("full_area", {}).get(
                    _ftid, occ_stats.get("full_area", {}).get(str(_ftid), 0))
                _f_visibility = _f_vis / max(_f_full, 1)
                # The buried target must genuinely be ON THE TABLE and present.
                # Two failure modes to reject (both make a "yes" answer wrong):
                #   1. fell off the table onto the floor (z well below table top)
                #   2. full_area (own silhouette, all else hidden) ~0 → off-FoV
                _z_top = float(getattr(demo.env, "Z_TABLE_TOP", 0.785))
                _f_z = None
                if _ftid is not None:
                    try:
                        _f_z = float(demo.env.get_obj_pos(_ftid)[2])
                    except Exception:
                        _f_z = None
                _f_on_table = (_f_z is not None and _f_z >= _z_top - 0.05)
                _f_present = _f_full >= 100
                if (_ftid is None or not _f_on_table or not _f_present
                        or _f_visibility >= V2_OCCLUDED_MAX_VISIBILITY):
                    if _ftid is not None and not _f_on_table:
                        _why = "off-table (z=%.3f<%.3f)" % (
                            _f_z if _f_z is not None else -1, _z_top - 0.05)
                    elif _ftid is not None and not _f_present:
                        _why = "absent (full_area=%d<100)" % _f_full
                    else:
                        _why = "not occluded enough"
                    print(f"  ⚠ find-yes: target {_why} "
                          f"({_find_target_name}: vis={_f_visibility:.0%}, "
                          f"full={_f_full}px, need <{V2_OCCLUDED_MAX_VISIBILITY:.0%})")
                    safe_close_demo(demo)
                    demo = None
                    free_gpu_memory()
                    if _retry_count < MAX_SCENE_RETRIES:
                        _retry_count += 1
                        print(f"  ↻ Retry {_retry_count}/{MAX_SCENE_RETRIES}")
                        continue
                    skip_record = {
                        "episode": ep,
                        "question_type": ep_question_type,
                        "status": "skipped",
                        "skip_reason": "find_target_not_occluded",
                        "retry_count": _retry_count,
                        "target_visibility": round(_f_visibility, 3),
                    }
                    with open(results_path, "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(skip_record) + "\n")
                    _ep_idx += 1
                    _retry_count = 0
                    continue
                else:
                    print(f"  ✓ find-yes: target occluded — "
                          f"{_find_target_name} (vis={_f_visibility:.0%} "
                          f"< {V2_OCCLUDED_MAX_VISIBILITY:.0%})")

            # ── count occlusion gate ───────────────────────────────────
            # At least ONE target instance must be genuinely occluded (same
            # bar as find/beneath) so the VLM cannot just count from the clean
            # frame — it must manipulate to be sure.  Only applies when the
            # scene actually contains target instances (GT count > 0); count==0
            # scenes have nothing to occlude and are left as-is.
            if (_bench_v2 and ep_question_type == "count"
                    and counting_target_name is not None
                    and (counting_target_count or 0) > 0 and snap_ep is None):
                # Only PRESENT instances count (on-table AND in-FoV) — the same
                # set the GT recompute keeps.  An off-table or out-of-frame
                # instance renders ~0 px and would falsely read as "occluded";
                # full_area (own silhouette) ≈ 0 marks it out-of-FoV.
                _z_top = float(getattr(demo.env, "Z_TABLE_TOP", 0.785))
                _min_vis = 1.0
                _n_present = 0
                for _tid, _nm in zip(demo.env.obj_ids, demo.env.obj_names):
                    if _nm != counting_target_name:
                        continue
                    _fa = occ_stats.get("full_area", {}).get(
                        _tid, occ_stats.get("full_area", {}).get(str(_tid), 0))
                    try:
                        _z = float(demo.env.get_obj_pos(_tid)[2])
                    except Exception:
                        _z = -1.0
                    if _z < _z_top - 0.01 or _fa < 100:
                        continue   # off-table or out-of-FoV → not a counted instance
                    _n_present += 1
                    _va = occ_stats.get("visible_area", {}).get(
                        _tid, occ_stats.get("visible_area", {}).get(str(_tid), 0))
                    _min_vis = min(_min_vis, _va / max(_fa, 1))
                if _n_present == 0 or _min_vis >= V2_OCCLUDED_MAX_VISIBILITY:
                    print(f"  ⚠ count: no target instance occluded enough "
                          f"({counting_target_name}: best vis={_min_vis:.0%}, "
                          f"need <{V2_OCCLUDED_MAX_VISIBILITY:.0%})")
                    safe_close_demo(demo)
                    demo = None
                    free_gpu_memory()
                    if _retry_count < MAX_SCENE_RETRIES:
                        _retry_count += 1
                        print(f"  ↻ Retry {_retry_count}/{MAX_SCENE_RETRIES}")
                        continue
                    skip_record = {
                        "episode": ep,
                        "question_type": ep_question_type,
                        "status": "skipped",
                        "skip_reason": "count_target_not_occluded",
                        "retry_count": _retry_count,
                        "best_target_visibility": round(_min_vis, 3),
                    }
                    with open(results_path, "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(skip_record) + "\n")
                    _ep_idx += 1
                    _retry_count = 0
                    continue
                else:
                    print(f"  ✓ count: target occluded — {counting_target_name} "
                          f"(most-occluded instance vis={_min_vis:.0%} "
                          f"< {V2_OCCLUDED_MAX_VISIBILITY:.0%})")

            # ── Definitive seg buffer ──────────────────────────────────
            # Capture ONE PyBullet seg after all alpha-modifying operations
            # (pick_most_occluded_target, compute_cover_hidden_pairs) have
            # completed and restored object alphas.  This seg is used for:
            #   1. A/B point visibility check (below)
            #   2. Step 1 of the VLM loop (passed as initial_seg)
            # This eliminates the non-determinism bug where PyBullet's
            # TinyRenderer produced different seg results across calls for
            # the same static scene (objects at grazing visibility edges).
            _, _, _settled_seg = demo.env.camera.get_cam_img_pybullet()

            # ── GT-visibility filter (conversational mode) ────────────
            # Drop objects that are off-table OR invisible in the
            # post-settle camera view BEFORE generating the GT.  The
            # core demo.py keeps the original list (for backward
            # comparability of legacy results); this filter only runs
            # in demo_conversational.py.  See gt_visibility_filter.py.
            _gt_visible_names, _gt_filter_info = filter_visible_obj_names(
                demo, _settled_seg, verbose=bool(args.verbose))
            if _gt_filter_info["n_dropped_off_table"] > 0 \
               or _gt_filter_info["n_dropped_invisible"] > 0:
                print(f"  [gt_filter] kept "
                      f"{_gt_filter_info['n_kept']}/"
                      f"{_gt_filter_info['n_spawned']}  "
                      f"(off-table {_gt_filter_info['n_dropped_off_table']}, "
                      f"invisible {_gt_filter_info['n_dropped_invisible']})")
            # Persist filter results for downstream auditing.
            try:
                with open(os.path.join(ep_dir, "gt_visibility.json"), "w",
                          encoding="utf-8") as _gf:
                    json.dump(_gt_filter_info, _gf, indent=2)
            except Exception as _e:
                print(f"  ⚠ Could not write gt_visibility.json: {_e}")

            # Counting scenes plan a specific target_count up-front, but a
            # target-instance can roll off-frame / off-table after settle.
            # Recompute so GT matches what is actually PRESENT in the scene.
            #
            # IMPORTANT: counting must include instances that are fully
            # occluded (buried under another object) — digging those out is
            # the whole point of the task.  So we count every instance that
            # is on-table AND inside the FoV (unoccluded silhouette via the
            # alpha-hide-all-others trick), NOT just the currently-visible
            # ones.  Using the plain visible-list recompute here caused GT to
            # under-count (e.g. ep_004: 4 mice spawned, 2 buried under a
            # plate → GT wrongly said 2).
            if counting_target_name is not None:
                _new_count = recompute_counting_target_count_present(
                    counting_target_name, demo, _settled_seg,
                    verbose=bool(args.verbose))
                if _new_count != counting_target_count:
                    print(f"  [gt_filter] counting_target_count (present): "
                          f"{counting_target_count} → {_new_count}  "
                          f"({counting_target_name})")
                    counting_target_count = _new_count

            # Generate question (type assigned by schedule)
            # Target objects are now chosen by the question generators
            # based on occlusion stats — the MOST occluded objects become
            # targets, ensuring the VLM must explore/manipulate to answer.
            q_rng = random.Random(episode_seed)
            # When using annotations_full, pass the designated target names
            # so existence questions only query target-role objects.
            _designated = None
            if target_pool is not None:
                _designated = [e["name"] for e in target_pool]

            # ── V2 rearrangement: per-object image centroids ───────────
            # Compute (row, col) centroid for every spawned object from the
            # definitive settled seg buffer.  The rearrangement generator
            # uses these to build the left/right/above/below GT in IMAGE
            # space (never the world frame).
            _pixel_centroids = None
            if _bench_v2 and ep_question_type == "rearrangement":
                _pixel_centroids = {}
                for _oid in demo.env.obj_ids:
                    _ys, _xs = np.where(_settled_seg == _oid)
                    if _ys.size > 0:
                        _pixel_centroids[int(_oid)] = (
                            float(_ys.mean()), float(_xs.mean()))

            # ── Frozen snapshot replay: reuse the saved GT descriptor ──
            # A --from_snapshot episode already carries its full question +
            # GT (rehydrated with restored body IDs by restore_scene_into_env).
            # Bypass all fresh question generation.  For marked_comparison the
            # badges were re-applied during restore; the clean frame was
            # rendered above with them present.
            _marked_info = None
            if snap_ep is not None:
                q_desc = _restored_q_desc
                if q_desc is None:
                    # No annotation → cannot score.  Record + skip (advance).
                    skip_record = {
                        "episode": ep,
                        "question_type": ep_question_type,
                        "status": "skipped",
                        "skip_reason": "snapshot_missing_annotation",
                    }
                    with open(results_path, "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(skip_record) + "\n")
                    print(f"  ⏭ ep {ep:03d}: skipped (snapshot has no "
                          f"annotation.question_descriptor)")
                    safe_close_demo(demo)
                    demo = None
                    free_gpu_memory()
                    _ep_idx += 1
                    _retry_count = 0
                    continue
                if _effective_qtype == "marked_comparison":
                    _marked_info = {"frozen_snapshot": snap_ep["ep_dirname"]}

            # ── compare / marked_comparison: badge-baked 3-way task ────
            # ``compare`` (V2) maps to the runtime ``marked_comparison`` type
            # via ``_effective_qtype``.  Its descriptor is produced by Mika's
            # marker pipeline (paints A/B/C badges onto 3 targets), NOT by the
            # generic descriptor generator.  A frozen bank replays a saved
            # scene; otherwise we paint fresh badges for the settled scene.
            elif _effective_qtype == "marked_comparison":
                if bank_row is not None:
                    from experiments.marked_size.scene_bank import (
                        hydrate_marked_descriptor)
                    q_desc = hydrate_marked_descriptor(demo, bank_row["package"])
                    _marked_info = {
                        "frozen_scene_id": bank_row["package"]["scene"]["scene_id"]}
                    print(f"  ✓ frozen marked scene restored: "
                          f"{_marked_info['frozen_scene_id']}", flush=True)
                else:
                    from experiments.marked_size.integrate import (
                        prepare_marked_episode)
                    _tex_dir = os.path.join(save_dir, "_marked_tex",
                                            f"ep{ep:03d}")
                    # ── largest / smallest question polarity ───────────────
                    # Each compare episode asks either "largest" or "smallest"
                    # volume.  The choice is deterministic per episode index
                    # (even→largest, odd→smallest) so a full random schedule is
                    # exactly balanced 50/50 and reproducible.  The physical
                    # scene, badges and occlusion tier are IDENTICAL either way
                    # — only the answer letter differs (see _finalize_triple).
                    _extremum = "smallest" if (ep % 2 == 1) else "largest"
                    q_desc, _marked_info = prepare_marked_episode(
                        demo,
                        dict(zip(scene_obj_names, scene_obj_folders)),
                        args.asset_path, gt_lookup, q_rng, _tex_dir,
                        occ_stats=occ_stats,
                        # Keep every quality-valid triple and record whether its
                        # largest target is least/middle/most occluded. A frozen
                        # scene-bank manifest later stratifies those realised
                        # tiers; do not throw away good scenes to target a rank.
                        wanted_largest_rank=None,
                        extremum=_extremum)
                    if q_desc is None:
                        print(f"  ⚠ marked_comparison unavailable for ep {ep}: "
                              f"{_marked_info}")
                    else:
                        print(f"  ✓ badges painted [{_extremum}]: "
                              f"{_marked_info['occlusions']} "
                              f"(eligible sites: {_marked_info['n_eligible']})")
                # ── Re-render the clean frame WITH badges baked in ─────────
                # The clean Blender frame (after_walls_topdown.png +
                # _blender_scene_rgb) was captured BEFORE badges were painted,
                # so the VLM-facing image showed no A/B/C labels.  Both marked
                # paths (fresh prepare_marked_episode + frozen-bank hydrate)
                # push the badge textures to Blender, so re-render now and
                # overwrite the clean frame + its cached copy.  This is the
                # image the VLM actually sees, so the labels MUST be present.
                if q_desc is not None and args.renderer == "blender":
                    try:
                        _rgb_badged, _, _ = demo.env.camera.get_cam_img()
                        _rgb_badged = np.asarray(_rgb_badged[:, :, :3])
                        after_path = os.path.join(ep_dir,
                                                  "after_walls_topdown.png")
                        cv2.imwrite(after_path,
                                    cv2.cvtColor(_rgb_badged.astype(np.uint8),
                                                 cv2.COLOR_RGB2BGR))
                        _blender_scene_rgb = _rgb_badged
                        print("  ✓ clean frame re-rendered with badges "
                              "(VLM-facing image now shows A/B/C)")
                    except Exception as _re:
                        print(f"  ⚠ badge re-render failed (non-fatal): {_re}")
                # ── LLM phrasing for compare (stem only; A/B/C preserved) ──
                # compare bypasses generate_question_descriptor(), so apply the
                # optional LLM phrasing here.  Frozen-bank replays keep their
                # saved (already-phrased) question verbatim.
                if (q_desc is not None and bank_row is None
                        and getattr(args, "question_phrasing", "template") == "llm"):
                    try:
                        from scene_gen.llm_phrasing import phrase_question
                        _pm = getattr(args, "phrasing_model",
                                      getattr(args, "model_name", None))
                        # Pass the episode's polarity so the LLM writes a
                        # "largest" or "smallest" stem to match the GT.
                        _cmp_ext = q_desc.get("comparison_extremum", "largest")
                        _res = phrase_question(
                            "compare", {"extremum": _cmp_ext},
                            model_name=_pm, use_llm=True)
                        q_desc["question"] = _res["question"]
                        q_desc["question_stem"] = _res["stem"]
                        q_desc["phrasing_source"] = _res["source"]
                        q_desc["template_variant"] = f"llm_{_res['source']}"
                        q_desc["template_category"] = "llm"
                    except Exception as _pe:
                        print(f"  ⚠ compare LLM phrasing failed (non-fatal): {_pe}")
            elif snap_ep is None:
                q_desc = generate_question_descriptor(
                    question_type=ep_question_type,
                    scene_obj_names=_gt_visible_names,  # visibility-filtered
                    gt_lookup=gt_lookup,
                    occ_stats=occ_stats,
                    rng=q_rng,
                    cover_hidden_pairs=cover_hidden_pairs,
                    counting_target_name=counting_target_name,
                    counting_target_count=counting_target_count,
                    designated_targets=_designated,
                    benchmark_version=getattr(args, "benchmark_version", "v1"),
                    pixel_centroids=_pixel_centroids,
                    question_phrasing=getattr(args, "question_phrasing", "template"),
                    phrasing_model=getattr(args, "phrasing_model",
                                           getattr(args, "model_name", None)),
                    # V2 find: force polarity + query the pre-buried target
                    # (occluded, so absent from the visible-name list).
                    find_forced_is_yes=(_find_yes if (_bench_v2 and
                                        ep_question_type == "find") else None),
                    find_forced_target=_find_target_name,
                )

            if q_desc is None:
                print(f"  ⚠ Could not generate {ep_question_type} question for ep {ep}")
                safe_close_demo(demo)
                demo = None
                free_gpu_memory()
                _max_q_retries = (MARKED_SCENE_RETRIES if
                                  ep_question_type == "marked_comparison"
                                  else MAX_SCENE_RETRIES)
                if _retry_count < _max_q_retries:
                    _retry_count += 1
                    print(f"  ↻ Retry {_retry_count}/{_max_q_retries}")
                    continue
                # Record the exhausted slot.  Without this the episode is absent
                # from results.jsonl, so on worker recycle
                # orchestrator._get_completed_episodes() treats it as unfinished
                # and re-assigns it -- forever, at MAX_SCENE_RETRIES scene
                # spawns per attempt.  (14/40 slots were stuck in this loop,
                # which is what made a 40-slot run take ~20 hours.)
                skip_record = {
                    "episode": ep,
                    "question_type": ep_question_type,
                    "status": "skipped",
                    "skip_reason": "question_generation_failed",
                    "retry_count": _retry_count,
                }
                with open(results_path, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(skip_record) + "\n")
                print(f"  ⏭ ep {ep:03d}: recorded skip "
                      f"(question_generation_failed after {_retry_count} retries)")
                _ep_idx += 1
                _retry_count = 0
                continue

            question = q_desc["question"]
            target_name = q_desc.get("target_name")
            resolved_qtype = q_desc["question_type"]
            gt_answer = q_desc.get("gt_answer")
            answer_options = q_desc.get("answer_options") or []

            # ── A/B point visibility validation ────────────────────────
            # For same_attribute / relative_size: verify that at least one
            # of the A/B target points will be BLANK (object too occluded
            # for identification).  Uses the definitive _settled_seg —
            # same buffer that step 1 of the VLM loop will use, so the
            # check result is guaranteed to match the prompt content.
            if resolved_qtype in AB_COORD_TYPES:
                _m_ids = q_desc.get("marker_obj_ids")
                _m_fa = q_desc.get("marker_full_area")
                if _m_ids and _m_fa:
                    _ab_check = compute_ab_marker_info(
                        _settled_seg, _m_ids, _m_fa,
                        visibility_threshold=0.30)
                    _a_vis = _ab_check.get("A", {}).get("visible", False)
                    _b_vis = _ab_check.get("B", {}).get("visible", False)
                    if _a_vis and _b_vis:
                        # Both targets fully visible → VLM doesn't need to
                        # manipulate → scene is too easy for this question type
                        _a_pct = _ab_check["A"].get("visibility", 0) * 100
                        _b_pct = _ab_check["B"].get("visibility", 0) * 100
                        print(f"  ⚠ SKIPPED ep {ep} ({resolved_qtype}): "
                              f"both A/B targets visible — "
                              f"A={q_desc.get('marker_obj_names',{}).get('A','?')} "
                              f"({_a_pct:.0f}% vis), "
                              f"B={q_desc.get('marker_obj_names',{}).get('B','?')} "
                              f"({_b_pct:.0f}% vis). "
                              f"Need at least one point blank.")
                        safe_close_demo(demo)
                        demo = None
                        free_gpu_memory()
                        if _retry_count < MAX_SCENE_RETRIES:
                            _retry_count += 1
                            print(f"  ↻ Retry {_retry_count}/{MAX_SCENE_RETRIES}")
                            continue
                        skip_record = {
                            "episode": ep,
                            "question_type": resolved_qtype,
                            "status": "skipped",
                            "skip_reason": "both_targets_visible",
                            "retry_count": _retry_count,
                        }
                        with open(results_path, "a", encoding="utf-8") as rf:
                            rf.write(json.dumps(skip_record) + "\n")
                        _ep_idx += 1
                        _retry_count = 0
                        continue
                    else:
                        _blank = "A" if not _a_vis else "B"
                        print(f"  ✓ A/B check: point {_blank} is blank → "
                              f"VLM must manipulate to reveal it")

            n_obj = len(demo.env.obj_names)
            if target_name:
                target_occ = occ_stats["occlusion"].get(
                    occ_stats.get("name_to_id", {}).get(target_name), 0.0)
            else:
                target_occ = 0.0

            # File paths
            safe_target = (target_name or "scene").replace(" ", "_")[:40]
            stem = f"{ep_dirname}_tgt_{safe_target}"
            base_path = os.path.join(ep_dir, stem)
            log_path = os.path.join(ep_dir, f"{stem}_log.log")

            # Save question descriptor, occlusion stats
            with open(os.path.join(ep_dir, f"{stem}_question.json"), "w", encoding="utf-8") as f:
                json.dump(q_desc, f, indent=2)
            occ_save = dict(occ_stats)
            occ_save["cover_hidden_pairs"] = cover_hidden_pairs
            with open(os.path.join(ep_dir, f"{stem}_occlusion.json"), "w", encoding="utf-8") as f:
                json.dump(occ_save, f, indent=2, default=str)

            # Save GT scene image — Blender render with target/A/B annotations.
            # The base image matches step_1_img1.png (what VLM sees);
            # annotations show which objects the question is about.
            gt_img_path = os.path.join(ep_dir, f"{stem}_gt_scene.png")
            if args.renderer == "blender" and _blender_scene_rgb is not None:
                _gt_annotated = _annotate_gt_scene(
                    _blender_scene_rgb, _settled_seg, q_desc, occ_stats,
                    camera=demo.env.camera)
                cv2.imwrite(gt_img_path, _gt_annotated)
            elif args.renderer == "blender":
                _gt_rgb, _, _ = demo.env.camera.get_cam_img()
                _gt_annotated = _annotate_gt_scene(
                    _gt_rgb, _settled_seg, q_desc, occ_stats,
                    camera=demo.env.camera)
                cv2.imwrite(gt_img_path, _gt_annotated)
            else:
                # PyBullet-only: use the settled seg for annotation
                _pb_rgb, _, _ = demo.env.camera.get_cam_img_pybullet()
                _gt_annotated = _annotate_gt_scene(
                    _pb_rgb, _settled_seg, q_desc, occ_stats,
                    camera=demo.env.camera)
                cv2.imwrite(gt_img_path, _gt_annotated)

            # Save target segmentation overlay — shows ONLY the 1-2 target
            # objects for this question (A/B, counting target, existence
            # target, etc.) with their seg mask highlighted. For heavily
            # occluded targets, draws the projected AABB as a dashed box.
            _seg_base_rgb = (
                _blender_scene_rgb if (_blender_scene_rgb is not None)
                else demo.env.camera.get_cam_img_pybullet()[0]
            )
            _render_target_seg_overlay(
                _seg_base_rgb, _settled_seg, q_desc, occ_stats,
                demo.env, os.path.join(ep_dir, f"{stem}_seg_overlay.png"),
                camera=demo.env.camera,
            )

            # ── Frozen scene snapshot (settled poses + table texture) ──
            # Save the exact initial world so the finalized benchmark can be
            # restored deterministically (teleport, no physics replay) without
            # re-spawning.  Skipped when this episode was itself restored from
            # a snapshot (nothing new to freeze).
            if bool(getattr(args, "save_snapshot", 1)) and not _loaded_from_snapshot:
                try:
                    save_scene_snapshot(
                        demo, ep_dir,
                        episode=ep,
                        episode_seed=episode_seed,
                        question_type=resolved_qtype,
                        benchmark_version=getattr(args, "benchmark_version", "v1"),
                        asset_path=args.asset_path,
                        copy_texture=True,
                        question_descriptor=q_desc,
                        verbose=bool(getattr(args, "verbose", 1)),
                    )
                except Exception as _snap_e:
                    print(f"  ⚠ scene snapshot failed (non-fatal): {_snap_e}")

            # ── Snapshot-only mode: write review.json and STOP ─────────
            # No VLM, no vlm_steps/, no sim-gt-stats/.  This episode's job
            # is done once the scene + question + review manifest are saved.
            if snapshots_only:
                _img_files = {
                    "clean": "after_walls_topdown.png",
                    "gt_scene": os.path.basename(gt_img_path),
                    "seg_overlay": f"{stem}_seg_overlay.png",
                }
                try:
                    write_review_manifest(
                        demo, q_desc, occ_stats, ep_dir, demo.env.camera,
                        episode=ep,
                        episode_seed=episode_seed,
                        question_type=resolved_qtype,
                        benchmark_version=getattr(args, "benchmark_version", "v1"),
                        image_files=_img_files,
                        seg=_settled_seg,
                    )
                except Exception as _rev_e:
                    print(f"  ⚠ review.json write failed (non-fatal): {_rev_e}")

                _snap_rec = {
                    "episode": int(ep),
                    "question_type": resolved_qtype,
                    "question": question,
                    "target_name": target_name,
                    "gt_answer": gt_answer,
                    "answer_options": answer_options,
                    "status": "snapshot_saved",
                    "episode_seed": int(episode_seed),
                    "n_objects": int(n_obj),
                }
                with open(results_path, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(_snap_rec) + "\n")
                    rf.flush()
                    os.fsync(rf.fileno())
                print(f"  📸 ep={ep+1}/{args.n_episodes} [{resolved_qtype}] "
                      f"snapshot + review.json saved ({n_obj} objects)")
                _ep_idx += 1
                _retry_count = 0
                continue

            # Run VLM reasoning loop — Tee captures ALL output (header + loop) to log 
            with open(log_path, "w", encoding="utf-8") as log_file:
                tee_stdout = Tee(sys.__stdout__, log_file)
                tee_stderr = Tee(sys.__stderr__, log_file)

                with contextlib.redirect_stdout(tee_stdout), \
                     contextlib.redirect_stderr(tee_stderr):

                    # Log episode header into the file too
                    print(f"\n{'='*60}")
                    print(f"  Episode {ep+1}/{args.n_episodes}  [{resolved_qtype}]  ({n_obj} objects)")
                    print(f"  Q: {question}")
                    print(f"  Options: {answer_options}")
                    if resolved_qtype == "rearrangement":
                        # Rearrangement has no text GT — surface the removal
                        # set + reference so the log is debuggable.
                        _rm_ids = q_desc.get("gt_removal_ids", [])
                        _rm_names = q_desc.get("gt_removal_names", [])
                        _ref_id = q_desc.get("reference_obj_id")
                        _ref_name = q_desc.get("target_name")
                        _direction = q_desc.get("direction", "?")
                        print(f"  GT: remove {len(_rm_ids)} object(s) "
                              f"to the {_direction} of reference")
                        print(f"  GT reference (KEEP): id={_ref_id} "
                              f"'{_ref_name}'")
                        for _i, (_rid, _rn) in enumerate(zip(_rm_ids, _rm_names)):
                            print(f"  GT remove #{_i+1}: id={_rid} '{_rn}'")
                    else:
                        print(f"  GT: {gt_answer}")
                    if target_name:
                        print(f"  Target: {target_name} ({target_occ:.0%} occluded)")
                    print(f"  Objects: {list(demo.env.obj_names)}")
                    print(f"  Seed: {episode_seed}")
                    # Log marker details for same_attribute / relative_size (log file only)
                    m_names = q_desc.get("marker_obj_names")
                    if m_names and resolved_qtype in ("same_attribute", "relative_size"):
                        log_file.write(f"  ── Marker Details ──\n")
                        log_file.write(f"  Marker A: {m_names.get('A')}\n")
                        log_file.write(f"  Marker B: {m_names.get('B')}\n")
                        if resolved_qtype == "same_attribute":
                            color_a = gt_lookup.get(m_names.get("A"), {}).get("color", "?")
                            color_b = gt_lookup.get(m_names.get("B"), {}).get("color", "?")
                            log_file.write(f"  Color A: {color_a}\n")
                            log_file.write(f"  Color B: {color_b}\n")
                        elif resolved_qtype == "relative_size":
                            vol_a = q_desc.get("gt_volume_a", "?")
                            vol_b = q_desc.get("gt_volume_b", "?")
                            gt_ratio = q_desc.get("gt_ratio", "?")
                            log_file.write(f"  Volume A: {vol_a} cm³\n")
                            log_file.write(f"  Volume B: {vol_b} cm³\n")
                            log_file.write(f"  GT ratio (A/B): {gt_ratio}\n")
                        log_file.flush()
                    print(f"{'='*60}")

                    try:
                        if is_singlepass:
                            # Single-pass: one image, one VLM call, no tools.
                            # Capture Blender RGB + PyBullet seg atomically
                            # (see FIXES_FOR_April18 P1-1) so the A/B
                            # visibility computed from seg matches the RGB
                            # the VLM sees.
                            rgb_clean, _, seg_sp = demo.env.camera.get_cam_img()
                            rgb = rgb_clean.copy()

                            # A/B target info (text, not image markers) for
                            # same_attribute / relative_size.  Image stays clean.
                            m_ids = q_desc.get("marker_obj_ids")
                            m_fa = q_desc.get("marker_full_area")
                            ab_info_sp: Dict[str, Dict] = {}
                            if m_ids is not None:
                                ab_info_sp = compute_ab_marker_info(
                                    seg_sp, m_ids, m_fa,
                                    visibility_threshold=0.30)

                            # Save to vlm_steps/ (same structure as agentic mode)
                            vlm_steps_dir = os.path.join(ep_dir, "vlm_steps")
                            os.makedirs(vlm_steps_dir, exist_ok=True)

                            # Save exactly what the VLM sees (clean RGB)
                            rgb_path = os.path.join(vlm_steps_dir, "step_1_img1.png")
                            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                            # Build prompt (single source of truth).
                            # Pass ab_info so the prompt can render A/B points
                            # as text for same_attribute / relative_size.
                            sp_user_prompt = build_singlepass_prompt(
                                question=question,
                                question_type=resolved_qtype,
                                answer_options=answer_options,
                                q_desc=q_desc,
                                ab_info=ab_info_sp,
                            )
                            sp_prompt_path = os.path.join(vlm_steps_dir, "step_1_prompt.txt")
                            with open(sp_prompt_path, "w", encoding="utf-8") as spf:
                                spf.write(sp_user_prompt)
                            sp_sys_path = os.path.join(vlm_steps_dir, "system_prompt.txt")
                            with open(sp_sys_path, "w", encoding="utf-8") as ssf:
                                ssf.write(SINGLEPASS_SYSTEM_PROMPT)

                            # Write detailed prompt + image info to log file only
                            log_file.write(f"\n  [LOG] System prompt saved → vlm_steps/system_prompt.txt\n")
                            log_file.write(f"  [LOG] Image sent to VLM:\n")
                            log_file.write(f"    → vlm_steps/step_1_img1.png\n")
                            log_file.write(f"  [LOG] Prompt ({len(sp_user_prompt)} chars) saved → vlm_steps/step_1_prompt.txt\n")
                            log_file.write(f"  ┌── USER PROMPT {'─'*50}\n")
                            for pline in sp_user_prompt.splitlines():
                                log_file.write(f"  │ {pline}\n")
                            log_file.write(f"  └{'─'*68}\n")
                            log_file.flush()

                            _two_stage = (bool(getattr(args, "sp_identify_stage", 0))
                                          and resolved_qtype == "marked_comparison")
                            if _two_stage:
                                from mgvqa.singlepass import run_singlepass_identify
                                vlm_result = run_singlepass_identify(
                                    rgb=rgb, question=question,
                                    question_type=resolved_qtype,
                                    answer_options=answer_options,
                                    q_desc=q_desc,
                                    model_name=args.model_name,
                                    verbose=bool(args.verbose),
                                )
                                # persist the identification stage alongside
                                # the answer stage for the error decomposition
                                _ids = vlm_result.get("identify_stage") or {}
                                with open(os.path.join(vlm_steps_dir,
                                        "step_0_identify_prompt.txt"), "w") as _f:
                                    _f.write(_ids.get("prompt", ""))
                                with open(os.path.join(vlm_steps_dir,
                                        "step_0_identify_response.json"), "w") as _f:
                                    json.dump(_ids.get("report", {}), _f, indent=2)
                                if vlm_result.get("stage2_prompt"):
                                    with open(sp_prompt_path, "w") as _f:
                                        _f.write(vlm_result["stage2_prompt"])
                            else:
                                vlm_result = run_singlepass(
                                    rgb=rgb,
                                    user_prompt=sp_user_prompt,
                                    model_name=args.model_name,
                                    verbose=bool(args.verbose),
                                    question_type=resolved_qtype,
                                )

                            # Write full VLM response to log file only
                            log_file.write(f"  ┌── VLM RESPONSE {'─'*49}\n")
                            resp_dump = {
                                "answer": vlm_result.get("answer"),
                                # keep the identification report on disk; it was
                                # being dropped by this fixed key list
                                "badge_points": vlm_result.get("badge_points"),
                                "reasoning": vlm_result.get("reasoning", ""),
                                "point": vlm_result.get("point"),
                                "ratio": vlm_result.get("ratio"),
                                "success": vlm_result.get("success"),
                            }
                            for rline in json.dumps(resp_dump, indent=2).splitlines():
                                log_file.write(f"  │ {rline}\n")
                            log_file.write(f"  └{'─'*68}\n")
                            log_file.flush()
                            # Save VLM response as JSON file
                            resp_path = os.path.join(vlm_steps_dir, "step_1_response.json")
                            with open(resp_path, "w", encoding="utf-8") as rf:
                                json.dump(resp_dump, rf, indent=2)

                            record = {
                                "status": "answered" if vlm_result["success"] else "error",
                                "answer": vlm_result.get("answer"),
                                "predicted_point": vlm_result.get("point"),
                                "predicted_ratio": vlm_result.get("ratio"),
                                "question": question,
                                "target_name": target_name,
                                "total_vlm_steps": 1,
                                "manipulation_actions": 0,
                                "grasp_attempts": 0,
                                "grasp_successes": 0,
                                "push_attempts": 0,
                                "push_successes": 0,
                                "objects_removed": [],
                                "action_history": [],
                                "fail_reason": None if vlm_result["success"] else "VLM parse error",
                                "all_detected_labels": [],
                                "reasoning": vlm_result.get("reasoning", ""),
                                "mode": ("singlepass_identify" if _two_stage
                                          else "singlepass"),
                                "badge_points": vlm_result.get("badge_points"),
                            }
                        else:
                            # Few-shot exemplar for this question type (if --few_shot)
                            _fs_msgs = None
                            if _few_shot_bundle is not None:
                                _fs_msgs = _few_shot_bundle.get(resolved_qtype)
                                if _fs_msgs is None and bool(args.verbose):
                                    print(f"  ⚠ No few-shot exemplar for "
                                          f"qtype={resolved_qtype}; running 0-shot.")
                            record = run_vlm_reasoning_loop(
                                demo=demo,
                                q_desc=q_desc,
                                base_path=base_path,
                                molmo_processor=molmo_processor,
                                molmo_model=molmo_model,
                                sam_model=sam_model,
                                graspgen_sampler=graspgen_sampler,
                                max_actions=args.max_actions,
                                model_name=args.model_name,
                                verbose=bool(args.verbose),
                                log_file=log_file,
                                initial_seg=_settled_seg,
                                occ_stats=occ_stats,
                                sft_max_steps=getattr(args, "sft_max_steps", 0) or 0,
                                few_shot_messages=_fs_msgs,
                                rearrange_visible_thresh=getattr(
                                    args, "rearrange_visible_thresh", 0.10),
                                marked_condition=getattr(
                                    args, "marked_condition", "agentic"),
                                mode=_mode,
                                scene_model=args.scene_model,
                            )
                            if resolved_qtype == "marked_comparison":
                                _marked_ids = q_desc.get("marker_obj_ids") or {}
                                _removed_ids = {int(x) for x in
                                                record.get("removed_object_ids", [])}
                                record["marked_targets_removed"] = [
                                    label for label, oid in _marked_ids.items()
                                    if int(oid) in _removed_ids]

                    except Exception as e:
                        print(f"  ✗ Episode crashed: {e}")
                        import traceback
                        traceback.print_exc()
                        record = {
                            "status": "error",
                            "answer": None,
                            "question": question,
                            "target_name": target_name,
                            "total_vlm_steps": 0,
                            "manipulation_actions": 0,
                            "grasp_attempts": 0,
                            "grasp_successes": 0,
                            "push_attempts": 0,
                            "push_successes": 0,
                            "objects_removed": [],
                            "action_history": [],
                            "fail_reason": repr(e),
                        }

            # --- Existence point check: get current seg mask before closing ---
            # The VLM answers "A" or "B"; resolve to "yes"/"no" via option_map.
            # NOTE: count answers are bare ints (no letter/options), so coerce
            # to str before string ops to avoid AttributeError.
            point_on_mask = None
            vlm_answer_raw = str(record.get("answer") or "").strip().upper()
            option_map = q_desc.get("option_map", {})
            vlm_answer_resolved = option_map.get(vlm_answer_raw, vlm_answer_raw).lower()
            gt_answer_text = q_desc.get("gt_answer_text", q_desc.get("gt_answer", ""))

            if (resolved_qtype in ("existence", "find")
                    and vlm_answer_resolved == "yes"
                    and record.get("predicted_point") is not None):
                try:
                    target_obj_id = q_desc.get("target_obj_id")
                    pred_pt = record["predicted_point"]  # [row, col] in [0, 1000]
                    if (isinstance(pred_pt, (list, tuple)) and len(pred_pt) == 2):
                        rgb_pb, _, seg = demo.env.camera.get_cam_img_pybullet()
                        h, w = seg.shape[:2]
                        row_px = int(round(pred_pt[0] / 1000.0 * (h - 1)))
                        col_px = int(round(pred_pt[1] / 1000.0 * (w - 1)))
                        row_px = max(0, min(row_px, h - 1))
                        col_px = max(0, min(col_px, w - 1))

                        if target_obj_id is not None and gt_answer_text == "yes":
                            # Tolerant hit-test (FIXES_FOR_April20 note):
                            #   The VLM's (row, col) is only good to within
                            #   a few percent of image width — a point that
                            #   visually lies on the target can easily fall
                            #   1-3 % off the mask because of occluded
                            #   boundaries, shadows, or tiny gaps between
                            #   stacked boxes.  A strict single-pixel seg
                            #   lookup produces false negatives.
                            #
                            #   We now accept a hit if any pixel in a small
                            #   neighbourhood (radius ≈ 2.5 % of min(h, w),
                            #   ≈ 25 px at 1024²) belongs to the target.
                            #   Calibrated on the 20260420 existence
                            #   regression: the VLM gave point [579, 466]
                            #   for a box whose visible mask centre is at
                            #   [580, 490] — 24 px off in col.  With a
                            #   radius of 25 px the check now passes.
                            #   For logging, we still record the *centre*
                            #   pixel's id under point_hit_id — that's what
                            #   the report surfaces when there's a miss.
                            radius_hit = max(8, min(h, w) // 40)
                            r0 = max(0, row_px - radius_hit)
                            r1 = min(h, row_px + radius_hit + 1)
                            c0 = max(0, col_px - radius_hit)
                            c1 = min(w, col_px + radius_hit + 1)
                            patch = seg[r0:r1, c0:c1]
                            hit_center = int(seg[row_px, col_px])
                            point_on_mask = bool((patch == target_obj_id).any())
                            record["point_hit_id"] = hit_center
                            record["point_target_id"] = target_obj_id
                            record["point_hit_radius_px"] = int(radius_hit)
                        record["point_px"] = [row_px, col_px]

                        # --- Debug image: show GT (green) and predicted
                        # (blue/red depending on hit) side-by-side on the
                        # final RGB.  This is what the report uses when
                        # the letter was correct but the point missed
                        # (partial success) so a human can see exactly
                        # where the VLM pointed vs where the object is.
                        pt_img = np.ascontiguousarray(rgb_pb.copy())
                        radius = max(6, min(h, w) // 60)

                        # GT centroid (from seg), if object is present
                        gt_cx = gt_cy = None
                        if target_obj_id is not None:
                            _ys, _xs = np.where(seg == target_obj_id)
                            if _ys.size > 0:
                                gt_cy = int(_ys.mean())
                                gt_cx = int(_xs.mean())
                                cv2.circle(pt_img, (gt_cx, gt_cy),
                                           radius + 4, (0, 255, 0), 2)
                                cv2.drawMarker(pt_img, (gt_cx, gt_cy),
                                               (0, 255, 0), cv2.MARKER_CROSS,
                                               radius * 2, 2)
                                cv2.putText(pt_img, "GT",
                                            (gt_cx + radius + 6, gt_cy - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                            (0, 255, 0), 2, cv2.LINE_AA)

                        # Predicted point
                        color_ok = (0, 220, 0)    # green = hit
                        color_bad = (255, 50, 50) # red = miss
                        color = color_ok if point_on_mask else color_bad
                        cv2.circle(pt_img, (col_px, row_px), radius, color, 3)
                        cv2.circle(pt_img, (col_px, row_px), 2, color, -1)
                        label = "HIT" if point_on_mask else "MISS"
                        if point_on_mask is None:
                            color = (255, 220, 0)  # yellow if GT is "no"
                            label = "POINT"
                        cv2.putText(pt_img, f"PRED ({label})",
                                    (col_px + radius + 6, row_px + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                                    cv2.LINE_AA)

                        # Connecting line when both present and miss
                        if (gt_cx is not None and point_on_mask is False):
                            cv2.line(pt_img, (gt_cx, gt_cy), (col_px, row_px),
                                     (200, 200, 200), 1, cv2.LINE_AA)

                        pt_path = os.path.join(ep_dir, f"{stem}_point_check.png")
                        cv2.imwrite(pt_path, cv2.cvtColor(pt_img, cv2.COLOR_RGB2BGR))

                        if args.verbose:
                            status_str = "✓ HIT" if point_on_mask else (
                                f"✗ MISS (hit id={record.get('point_hit_id')})"
                                if point_on_mask is False else "POINT (gt=no)")
                            print(f"  Point check: ({row_px},{col_px}) → {status_str}")
                            print(f"  Saved point debug image: {pt_path}")
                except Exception as e:
                    if args.verbose:
                        print(f"  Point check failed: {e}")
            record["point_on_mask"] = point_on_mask

            # --- Blender animation export (before closing the environment) ---
            if want_blend_export and not is_singlepass and args.renderer == "blender":
                motion_log = demo.env.get_motion_log()
                if motion_log:
                    try:
                        br = demo.env.camera._blender_renderer
                        pc = demo.env.physicsClient

                        # Load robot + insert keyframes ONCE
                        print(f"  Loading robot into Blender & inserting "
                              f"{len(motion_log)} keyframes...")
                        br.load_robot_into_scene(pc)
                        kf_result = br.insert_keyframes(motion_log, fps=30)
                        print(f"  Keyframes inserted: {kf_result}")

                        blend_path = os.path.join(
                            ep_dir, f"{stem}_manipulation.blend")
                        br.save_blend(blend_path)
                        print(f"  Saved .blend: {blend_path}")
                        print(f"  (render to MP4 offline: python render_blend_to_mp4.py {blend_path})")
                    except Exception as e:
                        print(f"  Blend export failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    if args.verbose:
                        print("  No manipulation frames recorded, skipping blend export.")

                # Clear motion log for next episode
                demo.env.clear_motion_log()

            safe_close_demo(demo)
            demo = None
            free_gpu_memory()

            # Ground-truth evaluation
            vlm_answer = record.get("answer")

            # ── V2 rearrangement: world-state F1 (no text answer) ──────
            if resolved_qtype == "rearrangement":
                gt_removal = set(q_desc.get("gt_removal_ids") or [])
                orig = set(record.get("original_obj_ids") or [])
                # Visibility-based removal: an object is "removed from view"
                # when ≤10% of its spawn footprint still renders in the FoV
                # (pushed off-frame / off the table / carried to the tray).
                # Falls back to obj_ids presence if the visible set is absent.
                remaining = set(record.get("remaining_visible_obj_ids")
                                or record.get("remaining_obj_ids") or [])
                removed = orig - remaining
                tp = len(removed & gt_removal)
                precision = tp / len(removed) if removed else (
                    1.0 if not gt_removal else 0.0)
                recall = tp / len(gt_removal) if gt_removal else 1.0
                f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
                is_correct = (abs(f1 - 1.0) < 1e-9)
                record["rearrange_precision"] = round(precision, 4)
                record["rearrange_recall"] = round(recall, 4)
                record["rearrange_f1"] = round(f1, 4)
                record["gt_removal_ids"] = sorted(gt_removal)
                record["removed_obj_ids"] = sorted(removed)
                record["extra_removed_ids"] = sorted(removed - gt_removal)
                record["missed_ids"] = sorted(gt_removal - removed)
                # Diagnostics: which IDs still physically exist in the sim
                # vs. which are counted removed purely because they left the
                # FoV (pushed off-frame / off table / trayed but still a body).
                record["still_present_obj_ids"] = sorted(remaining)
                record["removal_method"] = "visibility_fov"
                record["reference_obj_id"] = q_desc.get("reference_obj_id")
                record["reference_removed"] = (
                    q_desc.get("reference_obj_id") in removed)
                if args.verbose:
                    print(f"  Rearrangement F1={f1:.2f} "
                          f"(P={precision:.2f} R={recall:.2f})  "
                          f"removed={sorted(removed)} gt={sorted(gt_removal)}"
                          f"{'  ⚠REF-REMOVED' if record['reference_removed'] else ''}")
                record["ground_truth"] = q_desc.get("gt_removal_names")
                record["is_correct"] = is_correct
                record["question_descriptor"] = q_desc
                _v2_scored = True
            else:
                _v2_scored = False

            if _v2_scored:
                pass  # already scored above
            elif resolved_qtype == "count" and vlm_answer is not None:
                # ── V2 count: exact integer match ─────────────────────
                import re as _re
                _m = _re.search(r"-?\d+", str(vlm_answer))
                pred_int = int(_m.group()) if _m else None
                try:
                    gt_int = int(gt_answer)
                except (TypeError, ValueError):
                    gt_int = None
                is_correct = (pred_int is not None and gt_int is not None
                              and pred_int == gt_int)
                record["predicted_int"] = pred_int
                record["gt_int"] = gt_int
                if args.verbose:
                    print(f"  Count: pred={pred_int}  gt={gt_int}  "
                          f"{'✓' if is_correct else '✗'}")
                record["ground_truth"] = gt_answer
                record["is_correct"] = is_correct
                record["question_descriptor"] = q_desc
            elif gt_answer is not None and vlm_answer is not None:
                # Defensive: coerce to str (count answers are ints, though
                # count is scored in its own branch above and never reaches
                # here) so string canonicalisation can't AttributeError.
                vlm_ans_clean = str(vlm_answer).strip()

                # ── Canonicalise the VLM's answer ─────────────────────────
                # We accept three possible forms:
                #   (a) a single letter     ("A", "b")
                #   (b) the option *text*   ("yes", "no", "3", "none of the above")
                #   (c) stray whitespace/punctuation around either
                # and convert both to a single canonical letter whenever an
                # ``option_map`` is available.  This closes the P0-2 bug
                # where an existence answer "yes" was compared directly to
                # the GT letter "B" and silently scored wrong.
                vlm_ans_letter = None     # "A"/"B"/… or None
                vlm_ans_resolved = vlm_ans_clean   # text-form fallback

                ans_upper = vlm_ans_clean.upper()
                if len(ans_upper) == 1 and ans_upper in "ABCDEFGHIJ":
                    vlm_ans_letter = ans_upper
                    letter_idx = ord(ans_upper) - ord("A")
                    if option_map and ans_upper in option_map:
                        vlm_ans_resolved = option_map[ans_upper]
                    elif letter_idx < len(answer_options):
                        vlm_ans_resolved = answer_options[letter_idx]
                elif option_map:
                    # Text form — look it up in the option_map values.
                    # Match case-insensitively; strip any leading "A: " prefix
                    # the VLM may have included.
                    _needle = ans_upper.lstrip("ABCDEFGHIJ: ").strip().lower()
                    for _k, _v in option_map.items():
                        if str(_v).strip().lower() == _needle or \
                           str(_v).strip().lower() == ans_upper.lower():
                            vlm_ans_letter = _k
                            vlm_ans_resolved = _v
                            break

                gt_letter = gt_answer.strip().upper()
                record["vlm_answer_letter"] = vlm_ans_letter
                record["vlm_answer_resolved"] = vlm_ans_resolved

                if resolved_qtype in ("existence", "find"):
                    # Letter-based comparison (canonicalised).
                    text_match = (vlm_ans_letter is not None
                                  and vlm_ans_letter == gt_letter)
                    record["letter_correct"] = text_match
                    gt_answer_text_val = q_desc.get("gt_answer_text", "")
                    if gt_answer_text_val == "yes":
                        is_correct = text_match and (point_on_mask is True)
                    else:
                        is_correct = text_match
                elif resolved_qtype == "same_attribute":
                    is_correct = (vlm_ans_letter is not None
                                  and vlm_ans_letter == gt_letter)
                elif resolved_qtype == "relative_size":
                    # Part 1: Which is larger? (letter match, canonicalised)
                    letter_correct = (vlm_ans_letter is not None
                                      and vlm_ans_letter == gt_letter)
                    # Part 2: Volume ratio A/B.
                    # Tolerance is now a symmetric log-ratio band (|log(pred/gt)|
                    # < 0.35 ≈ ±41 %).  Tight bands like ±10 % were too
                    # strict for humans/VLMs estimating volume by eye
                    # (see FIXES_FOR_April18 P1-6).  We also record
                    # ``ratio_error`` and ``ratio_error_log`` so the threshold
                    # can be retuned post hoc.
                    gt_ratio = q_desc.get("gt_ratio")
                    pred_ratio = record.get("predicted_ratio")
                    ratio_correct = False
                    ratio_error = None
                    ratio_error_log = None
                    if gt_ratio is not None and pred_ratio is not None \
                            and gt_ratio > 0:
                        try:
                            pred_ratio_f = float(pred_ratio)
                            if pred_ratio_f > 0:
                                ratio_error = pred_ratio_f / gt_ratio
                                import math as _math
                                ratio_error_log = abs(_math.log(ratio_error))
                                ratio_correct = ratio_error_log < 0.35
                        except (ValueError, TypeError):
                            ratio_correct = False
                    record["letter_correct"] = letter_correct
                    record["ratio_correct"] = ratio_correct
                    record["gt_ratio"] = gt_ratio
                    record["ratio_error"] = ratio_error
                    record["ratio_error_log"] = ratio_error_log
                    # Primary accuracy = letter only (which object is larger).
                    # Ratio estimation is tracked separately as a diagnostic
                    # but is NOT required for correctness — volume ratios are
                    # too hard for VLMs to estimate reliably from a single
                    # top-down view.
                    is_correct = letter_correct
                else:
                    # counting, interactive_counting, under_object, beneath:
                    # compare resolved option text.
                    vlm_lower = vlm_ans_resolved.strip().lower()
                    is_correct = vlm_lower == gt_answer.strip().lower()
                    # For under_object/beneath: also accept alternative valid
                    # answers (other objects substantially revealed by the same
                    # cover).  V2 beneath only populates also_valid in the 80%
                    # branch where the hidden object IS an option.
                    if not is_correct and resolved_qtype in ("under_object", "beneath"):
                        also_valid = q_desc.get("also_valid", [])
                        if any(vlm_lower == av.strip().lower() for av in also_valid):
                            is_correct = True
                            record["matched_also_valid"] = True
            else:
                is_correct = False
            record["ground_truth"] = gt_answer
            record["is_correct"] = is_correct
            record["question_descriptor"] = q_desc

            # Enrich record with metadata
            record.update({
                "episode": int(ep),
                "episode_seed": int(episode_seed),
                "selected_obj_names": list(selected_obj_names),
                "selected_obj_folders": list(selected_obj_folders),
                "question_type": resolved_qtype,
                "question_type_cli": args.question_type,
                "model_name": args.model_name,
                "ep_dir": ep_dir,
            })

            # Append full record to in-memory list (for summary stats)
            results.append(record)

            # Write a compact summary to results.jsonl (human-readable per line)
            jsonl_record = {
                "episode": int(ep),
                "question_type": resolved_qtype,
                "question": question,
                "answer_options": answer_options,
                "gt_answer": gt_answer,
                "vlm_answer": record.get("answer"),
                "is_correct": is_correct,
                "status": record.get("status"),
                "total_vlm_steps": record.get("total_vlm_steps"),
                "manipulation_actions": record.get("manipulation_actions", 0),
                "grasp_attempts": record.get("grasp_attempts", 0),
                "grasp_successes": record.get("grasp_successes", 0),
                "push_attempts": record.get("push_attempts", 0),
                "push_successes": record.get("push_successes", 0),
                "objects_removed": record.get("objects_removed", []),
                "model_name": args.model_name,
                "pointing_model": args.pointing_model,
                # Always record the pointer-VLM string so we never have
                # to guess which model produced the points. When
                # pointing_model is roborefer/molmo this is informational.
                "pointing_vlm_model": getattr(
                    args, "pointing_vlm_model",
                    "gcp/google/gemini-3.1-pro-preview"),
                "grasp_model": "graspgen",
                "episode_seed": int(episode_seed),
                "mode": _mode,          # normalized: direct / perception / agentic
                "mode_raw": args.mode,  # as passed (may be 'singlepass' alias)
            }
            if resolved_qtype == "marked_comparison":
                jsonl_record.update({
                    "marked_condition": getattr(args, "marked_condition", "agentic"),
                    "badge_points": record.get("badge_points"),
                    "badge_reports": record.get("badge_reports", []),
                    "marked_targets_removed": record.get("marked_targets_removed", []),
                })
            # Add question-type-specific fields
            if resolved_qtype == "relative_size":
                jsonl_record["letter_correct"] = record.get("letter_correct")
                jsonl_record["ratio_correct"] = record.get("ratio_correct")
                jsonl_record["predicted_ratio"] = record.get("predicted_ratio")
                jsonl_record["gt_ratio"] = record.get("gt_ratio")
                jsonl_record["ratio_error"] = record.get("ratio_error")
                jsonl_record["ratio_error_log"] = record.get("ratio_error_log")
            if resolved_qtype in ("existence", "find"):
                jsonl_record["letter_correct"] = record.get("letter_correct")
                jsonl_record["predicted_point"] = record.get("predicted_point")
                jsonl_record["point_on_mask"] = record.get("point_on_mask")
                jsonl_record["point_hit_id"] = record.get("point_hit_id")
                jsonl_record["point_target_id"] = record.get("point_target_id")
                jsonl_record["point_hit_radius_px"] = record.get("point_hit_radius_px")
            if resolved_qtype in ("under_object", "beneath"):
                jsonl_record["also_valid"] = q_desc.get("also_valid", [])
                jsonl_record["matched_also_valid"] = \
                    bool(record.get("matched_also_valid", False))
            if resolved_qtype == "count":
                jsonl_record["predicted_int"] = record.get("predicted_int")
                jsonl_record["gt_int"] = record.get("gt_int")
            if resolved_qtype == "rearrangement":
                jsonl_record["rearrange_f1"] = record.get("rearrange_f1")
                jsonl_record["rearrange_precision"] = record.get("rearrange_precision")
                jsonl_record["rearrange_recall"] = record.get("rearrange_recall")
                jsonl_record["gt_removal_ids"] = record.get("gt_removal_ids")
                jsonl_record["removed_obj_ids"] = record.get("removed_obj_ids")
                jsonl_record["extra_removed_ids"] = record.get("extra_removed_ids")
                jsonl_record["missed_ids"] = record.get("missed_ids")
                jsonl_record["reference_removed"] = record.get("reference_removed")
            if record.get("forced_answer"):
                jsonl_record["forced_answer"] = True
            if record.get("fail_reason"):
                jsonl_record["fail_reason"] = record["fail_reason"]
            # ── Instrumentation (E15) ──
            if record.get("belief_at_turn_1"):
                jsonl_record["belief_at_turn_1"] = record["belief_at_turn_1"]
            jsonl_record["perception_belief_changes"] = \
                record.get("perception_belief_changes", 0)
            jsonl_record["tool_disagreement_count"] = \
                record.get("tool_disagreement_count", 0)
            # Track prompt variant if applied
            if q_desc.get("template_variant"):
                jsonl_record["template_variant"] = q_desc["template_variant"]
                jsonl_record["template_category"] = q_desc.get("template_category", "")

            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(jsonl_record) + "\n")
                f.flush()
                os.fsync(f.fileno())

            status_icon = "✓" if is_correct else "✗"
            _g_att = record.get("grasp_attempts", 0)
            _g_suc = record.get("grasp_successes", 0)
            _p_att = record.get("push_attempts", 0)
            _p_suc = record.get("push_successes", 0)
            manip_parts = []
            if _g_att:
                manip_parts.append(f"grasps={_g_suc}/{_g_att}")
            if _p_att:
                manip_parts.append(f"pushes={_p_suc}/{_p_att}")
            manip_str = " | ".join(manip_parts) if manip_parts else "no actions"
            result_line = (
                f"\n  {status_icon} ep={ep+1}/{args.n_episodes} "
                f"| {resolved_qtype} | "
                f"answer={record.get('answer')} | "
                f"GT={gt_answer} | "
                f"{manip_str}"
            )
            if resolved_qtype == "relative_size":
                r_icon = "✓" if record.get("ratio_correct") else "✗"
                result_line += (
                    f" | ratio={record.get('predicted_ratio')} "
                    f"(GT={record.get('gt_ratio')}) {r_icon}"
                )
            print(result_line)

            # Write correctness verdict to log file (visible in both agentic & singlepass)
            # log_file from the `with` block is closed by now, so reopen in append mode.
            with open(log_path, "a", encoding="utf-8") as lf:
                # If the episode terminated abnormally, log the reason clearly
                ep_fail = record.get("fail_reason")
                if ep_fail:
                    lf.write(f"\n  ── TERMINATION ──\n")
                    lf.write(f"  Reason: {ep_fail}\n")
                    lf.write(f"  Steps used:  {record.get('total_vlm_steps', '?')}\n")
                    lf.write(f"  Actions used: {record.get('manipulation_actions', '?')}"
                             f" / {args.max_actions}\n")

                lf.write(f"\n  ── EVALUATION ──\n")
                lf.write(f"  VLM answer: {record.get('answer')}\n")
                lf.write(f"  GT answer:  {gt_answer}\n")
                lf.write(f"  Correct:    {'YES' if is_correct else 'NO'}\n")
                if resolved_qtype == "relative_size":
                    lf.write(f"  Letter:     {'YES' if record.get('letter_correct') else 'NO'}\n")
                    gt_r = record.get('gt_ratio') or 0
                    # Log-ratio band: exp(±0.35) ≈ ×/÷ 1.42
                    import math as _mlog
                    lo = gt_r * _mlog.exp(-0.35) if gt_r else 0
                    hi = gt_r * _mlog.exp(0.35) if gt_r else 0
                    err_log = record.get('ratio_error_log')
                    err_log_str = f"{err_log:.3f}" if err_log is not None else "—"
                    lf.write(f"  Ratio:      {record.get('predicted_ratio')} "
                             f"(GT={record.get('gt_ratio')}, "
                             f"log-band=[{lo:.2f}, {hi:.2f}], "
                             f"|log(pred/gt)|={err_log_str}) → "
                             f"{'YES' if record.get('ratio_correct') else 'NO'}\n")
                if resolved_qtype in ("existence", "find") and record.get("predicted_point"):
                    lf.write(f"  Point:      {record.get('predicted_point')} → "
                             f"{'HIT' if record.get('point_on_mask') else 'MISS'}\n")
                g_att = record.get("grasp_attempts", 0)
                g_suc = record.get("grasp_successes", 0)
                p_att = record.get("push_attempts", 0)
                p_suc = record.get("push_successes", 0)
                if g_att or p_att:
                    lf.write(f"\n  ── MANIPULATION ──\n")
                    lf.write(f"  Total actions: {record.get('manipulation_actions', 0)}\n")
                    if g_att:
                        lf.write(f"  Grasps:  {g_suc}/{g_att} = "
                                 f"{g_suc/g_att*100:.1f}%\n")
                    if p_att:
                        lf.write(f"  Pushes:  {p_suc}/{p_att} = "
                                 f"{p_suc/p_att*100:.1f}%\n")

            # ── Episode completed successfully — advance to next ──
            _ep_idx += 1
            _retry_count = 0

        except Exception as _ep_exc:
            # Safety net for any exception that escaped the inner handlers.
            # Log it, write a minimal failure record, then continue.
            import traceback as _tb
            print(f"  ✗ Episode {ep} aborted with uncaught exception: {_ep_exc}")
            _tb.print_exc()
            try:
                _crash_rec = {
                    "episode": int(ep),
                    "question_type": ep_question_type,
                    "status": "error",
                    "is_correct": False,
                    "fail_reason": f"uncaught: {_ep_exc!r}",
                }
                with open(results_path, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(_crash_rec) + "\n")
                    _f.flush()
            except Exception:
                pass
            # Advance past failed episode (don't retry crashes)
            _ep_idx += 1
            _retry_count = 0
        finally:
            # Guarantee env teardown + GPU memory release before next episode.
            safe_close_demo(demo)
            demo = None
            free_gpu_memory()

            # ── Worker recycling: exit after N episodes to free PyBullet's
            # leaked mesh memory. The orchestrator (demo_parallel.py) will
            # re-launch us with only the remaining episodes.
            _episodes_completed_this_subprocess += 1
            if (_max_eps_per_sub > 0
                    and _episodes_completed_this_subprocess >= _max_eps_per_sub
                    and _ep_idx < len(_ep_list)):
                _remaining = _ep_list[_ep_idx:]
                print(f"\n{'='*60}")
                print(f"[worker-recycle] Completed {_episodes_completed_this_subprocess} "
                      f"episodes (limit={_max_eps_per_sub}). "
                      f"Exiting for memory reclamation.")
                print(f"[worker-recycle] {len(_remaining)} episodes remaining "
                      f"(will be re-launched by orchestrator).")
                print(f"{'='*60}\n")
                sys.exit(0)

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    def _is_ep_correct(r):
        """For relative_size: letter_correct only. All others: is_correct."""
        from mgvqa.metrics import is_answer_correct
        return is_answer_correct(r)

    n_total = len(results)
    n_answered = sum(1 for r in results if r.get("status") == "answered")
    n_correct = sum(1 for r in results if _is_ep_correct(r))
    # Non-answer terminations counted as failures (max-steps, empty answer,
    # perception-loop auto-termination, …).  Excluded from n_answered → not
    # in either overall or per-type accuracy. n_total is inventory only.
    n_fail = sum(1 for r in results
                 if r.get("status") in ("fail", "perception_loop_failure"))
    n_perc_loop_fail = sum(1 for r in results
                           if r.get("status") == "perception_loop_failure")
    n_error = sum(1 for r in results if r.get("status") == "error")
    # Headline accuracy = correct / answered. Episodes that crashed or
    # hit max-steps are NOT penalised (they never produced an answer).
    accuracy = (n_correct / n_answered * 100) if n_answered else 0.0
    accuracy_answered = accuracy  # same — primary metric
    answered_actions = [r.get("manipulation_actions", 0) for r in results
                        if r.get("status") == "answered"]
    avg_actions = np.mean(answered_actions) if answered_actions else 0.0

    # Same answer-only policy for overall and per-type accuracy.
    from mgvqa.metrics import answer_metrics
    answer_stats = answer_metrics(results, scale=100, digits=2)
    qt_stats = answer_stats["per_type"]

    # Relative size: track letter-only and ratio-only side metrics
    rs_results = [r for r in results if r.get("question_type") == "relative_size"
                  and r.get("status") == "answered"]
    n_letter_correct = sum(1 for r in rs_results if r.get("letter_correct"))
    n_ratio_correct = sum(1 for r in rs_results if r.get("ratio_correct"))
    n_ratio_answered = len(rs_results)

    # Manipulation statistics (across all episodes) — separate grasp vs push
    total_manip_actions = sum(r.get("manipulation_actions", 0) for r in results)
    total_grasp_attempts = sum(r.get("grasp_attempts", 0) for r in results)
    total_grasp_successes = sum(r.get("grasp_successes", 0) for r in results)
    total_push_attempts = sum(r.get("push_attempts", 0) for r in results)
    total_push_successes = sum(r.get("push_successes", 0) for r in results)
    grasp_rate = (total_grasp_successes / total_grasp_attempts * 100) \
        if total_grasp_attempts else 0.0
    push_rate = (total_push_successes / total_push_attempts * 100) \
        if total_push_attempts else 0.0

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({n_total} scenes, {args.model_name})")
    print(f"{'='*60}")
    print(f"  Answered: {n_answered}/{n_total}  |  "
          f"Correct: {n_correct}/{n_answered}  |  "
          f"Accuracy: {accuracy:.1f}% (of answered: {accuracy_answered:.1f}%)  |  "
          f"Avg actions: {avg_actions:.1f}")
    if total_manip_actions > 0:
        parts = []
        if total_grasp_attempts:
            parts.append(f"Grasps: {total_grasp_successes}/{total_grasp_attempts} = "
                         f"{grasp_rate:.1f}%")
        if total_push_attempts:
            parts.append(f"Pushes: {total_push_successes}/{total_push_attempts} = "
                         f"{push_rate:.1f}%")
        print(f"  Manipulations: {total_manip_actions}  |  " + "  |  ".join(parts))
    if n_fail or n_error:
        _ploop = (f" (perception-loop: {n_perc_loop_fail})"
                  if n_perc_loop_fail else "")
        print(f"  Failed: {n_fail}{_ploop}  |  Errors: {n_error}")
    for reason, count in sorted(answer_stats["non_answer_failures_by_reason"].items()):
        print(f"  Non-answer: {count} — {reason}")
    print()
    print(f"  {'Question Type':<20s} {'Total':>6s} {'Ans':>6s} {'Correct':>8s} {'Acc':>7s} {'AccAns':>8s}")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*8}")
    for qt in (QUESTION_TYPES + QUESTION_TYPES_V2):
        s = qt_stats.get(qt, {"total": 0, "answered": 0, "correct": 0})
        acc_tot = (s["correct"] / s["answered"] * 100) if s["answered"] else 0.0
        acc_ans = (s["correct"] / s["answered"] * 100) if s["answered"] else 0.0
        print(f"  {qt:<20s} {s['total']:>6d} {s['answered']:>6d} {s['correct']:>8d} "
              f"{acc_tot:>6.1f}% {acc_ans:>7.1f}%")
    if n_ratio_answered:
        letter_acc = n_letter_correct / n_ratio_answered * 100
        ratio_acc = n_ratio_correct / n_ratio_answered * 100
        print(f"\n  Relative size — letter (primary): "
              f"{n_letter_correct}/{n_ratio_answered} = {letter_acc:.1f}%")
        print(f"  Relative size — ratio (diagnostic, |log(pred/gt)|<0.35): "
              f"{n_ratio_correct}/{n_ratio_answered} = {ratio_acc:.1f}%")
    print(f"{'='*60}")

    # Write summary to file
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_name,
            "question_type_cli": args.question_type,
            "n_total": n_total,
            "n_answered": n_answered,
            "n_correct": n_correct,
            "n_fail": n_fail,
            "n_perception_loop_failure": n_perc_loop_fail,
            "n_error": n_error,
            "accuracy": round(accuracy, 2),
            "accuracy_answered": round(accuracy_answered, 2),
            "_accuracy_notes": {
                "accuracy": "n_correct / n_answered (primary metric)",
                "relative_size": "Uses letter_correct only; ratio is diagnostic",
            },
            "avg_actions": round(avg_actions, 2),
            "per_question_type": qt_stats,
            "status_counts": answer_stats["status_counts"],
            "non_answer_failures_by_reason": answer_stats["non_answer_failures_by_reason"],
            "relative_size_detail": {
                "n_answered": n_ratio_answered,
                "letter_correct": n_letter_correct,
                "letter_accuracy": round(n_letter_correct / n_ratio_answered * 100, 2)
                                   if n_ratio_answered else 0.0,
                "ratio_correct": n_ratio_correct,
                "ratio_accuracy": round(n_ratio_correct / n_ratio_answered * 100, 2)
                                  if n_ratio_answered else 0.0,
                "_note": "letter_accuracy is used for headline; ratio is diagnostic only",
            },
            "manipulation": {
                "total_actions": total_manip_actions,
                "grasps": {
                    "model": "graspgen",
                    "attempts": total_grasp_attempts,
                    "successes": total_grasp_successes,
                    "success_rate": round(grasp_rate, 2),
                },
                "pushes": {
                    "attempts": total_push_attempts,
                    "successes": total_push_successes,
                    "success_rate": round(push_rate, 2),
                },
            },
        }, f, indent=2)

