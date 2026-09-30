"""
core.reasoning_loop — Conversational agentic reasoning loop.

This module contains the multi-turn VLM reasoning loop that powers agentic mode.
The VLM iteratively:
  1. Receives the current scene image + conversation history
  2. Selects a tool (perceive_scene, perceive_find_object, grasp_and_remove,
     push, or answer_question)
  3. The tool result is added to the conversation
  4. Repeat until answer_question or budget exhausted

Extracted verbatim from demo_conversational.py.
"""

import os
import json
import time
import copy
import numpy as np
import cv2
from typing import Dict, List, Optional
from PIL import Image

from owg_robot.ui import RobotEnvUI
from ultralytics import SAM

from models import call_model, call_vlm, parse_vlm_response, VLM_MODEL_NAME
from tools import (
    perceive_scene_v2, perceive_find_object, perceive_find_object_at_point,
    format_scene_graph_block, draw_scene_graph_overlay,
    execute_grasp_and_remove, execute_push,
    compute_ab_marker_info, format_ab_targets_block,
    build_sam_occupancy, draw_pointing_debug,
    build_perception_meta, format_perception_yaml,
    mental_map_new,
)
from mgvqa.utils import MAX_ACTIONS_PER_SCENE, reencode_video
from mgvqa.gt_helpers import _capture_sim_gt_state
# load_prompt MUST come from mgvqa.prompts (reads mgvqa/prompt_templates/), NOT
# from tools.load_prompt (which reads the root prompts/vlm_demo/). The mgvqa
# package owns its own prompt texts.
from mgvqa.prompts import load_prompt, build_user_prompt, format_previous_tool_response
from mgvqa.verification import pick_forced_answer
from mgvqa.tool_registry import canonical_name, execute_backend, normalize_planner_call


def run_vlm_reasoning_loop(
    demo: RobotEnvUI,
    q_desc: dict,
    base_path: str,
    molmo_processor,
    molmo_model,
    sam_model: SAM,
    graspgen_sampler=None,
    max_actions: int = MAX_ACTIONS_PER_SCENE,
    model_name: str = VLM_MODEL_NAME,
    verbose: bool = True,
    log_file=None,
    initial_seg=None,
    occ_stats: dict = None,
    sft_max_steps: int = 0,
    few_shot_messages: list = None,
    rearrange_visible_thresh: float = 0.10,
    marked_condition: str = "agentic",
    mode: str = "agentic",
    scene_model: str = "gcp/google/gemini-3.6-flash",
) -> dict:
    """
    Run the VLM-driven reasoning loop for a single scene.

    Args:
        demo:       Initialized RobotEnvUI instance with objects spawned.
        q_desc:     Question descriptor dict from generate_question_descriptor().
        base_path:  Path prefix for saving images and logs.
        max_actions: Maximum manipulation actions allowed.
        model_name: VLM model identifier.
        verbose:    Whether to print detailed logs.
        sft_max_steps: If >0, terminate the episode early when VLM step
                    count exceeds this limit. Used for SFT data generation
                    to avoid wasting API calls on episodes that will be
                    filtered out by build_sft_dataset.py (default max=15).

    Returns:
        A result dict with keys: status, answer, steps, actions_taken,
        objects_removed, fail_reason, etc.
    """
    # Unpack descriptor
    question = q_desc["question"]
    question_type = q_desc.get("question_type", "")
    target_name = q_desc.get("target_name")  # None for counting, existence-no
    target_display = target_name or ""
    answer_options = q_desc.get("answer_options") or []
    gt_answer = q_desc.get("gt_answer", "")
    require_target_visible = q_desc.get("require_target_visible", False)

    all_detected_labels: set = set()  # accumulates labels across all perception calls

    # A/B marker info for same_attribute / relative_size
    marker_obj_ids = q_desc.get("marker_obj_ids")        # {"A": id, "B": id} or None
    marker_full_area = q_desc.get("marker_full_area")     # {"A": px, "B": px} or None
    # The coordinate block is ONLY for the types that describe their targets in
    # text.  marked_comparison paints A/B onto the objects instead, so emitting
    # a point block would re-introduce exactly the leak it removes.
    from questions import AB_COORD_TYPES
    uses_ab_markers = (marker_obj_ids is not None
                       and question_type in AB_COORD_TYPES)

    ep_dir = os.path.dirname(base_path)
    os.makedirs(ep_dir, exist_ok=True)

    viz_dir = os.path.join(ep_dir, "vlm_steps")
    os.makedirs(viz_dir, exist_ok=True)

    # Load prompts — CONVERSATIONAL variant.  The system prompt is the
    # only "static" prompt; per-turn user content is built directly in
    # the loop below as a {role, text, images} dict appended to the
    # ``conversation`` list (which is then passed via ``extra_messages``).
    # System-prompt selection by question type:
    #   • marked_comparison (V2 `compare`) → Mika's patched marked prompt
    #     (badge A/B/C task, per-arm tool restrictions).
    #   • other V2 tasks (count/find/beneath) → the V2 system prompt
    #     (integer answers, etc.).
    #   • V1 (6-task) → the original conversational prompt, unchanged.
    # Detected by question_type so no extra arg needs threading through.
    from questions import QUESTION_TYPES_V2
    # V2 grounding-by-point workflow: perceive_scene returns PRECISE per-object
    # 2D points, and perceive_find_object takes a point (not an object name).
    # ALL four V2 tasks — including marked_comparison (compare) — now share the
    # single V2 system prompt and the point-based grounding workflow.  The
    # compare-specific badge guidance is injected per-turn via the type
    # constraint (see below), NOT via a separate system prompt.
    # NOTE: the runtime question_type for the V2 `compare` task is the string
    # "marked_comparison" (the V2 vocabulary name `compare` is only used in the
    # schedule); it is NOT literally in QUESTION_TYPES_V2, so include it here.
    _is_v2_task = (question_type in QUESTION_TYPES_V2
                   or question_type == "marked_comparison")
    is_v2_points = _is_v2_task
    # perception_only fires if EITHER the top-level --mode is 'perception'
    # (applies to ALL four V2 tasks) OR the compare-specific ablation requests
    # it via --marked_condition.  When True, grasp_and_remove / push are refused
    # and the system prompt advertises only the perception tools.
    perception_only = (
        mode == "perception"
        or (question_type == "marked_comparison" and
            marked_condition in {"perception", "perception_suggest",
                                 "perception_report"}))
    marked_suggest = (question_type == "marked_comparison" and
                      marked_condition in {"perception_suggest", "perception_report"})
    # Keep optional badge-point diagnostics matched across the manipulation
    # comparison and both post-baseline agentic prompt interventions. These
    # points remain nullable diagnostics—not a gate on the A/B/C answer—and
    # do not disclose any evaluator-side badge coordinates to the model.
    marked_report = (question_type == "marked_comparison" and
                     marked_condition in {"perception_report", "agentic",
                                          "agentic_instructed", "agentic_ledger"})
    marked_instructed = (question_type == "marked_comparison" and
                         marked_condition == "agentic_instructed")
    marked_ledger = (question_type == "marked_comparison" and
                     marked_condition == "agentic_ledger")
    if _is_v2_task:
        # ALL V2 tasks (count/find/beneath/compare) share the SINGLE V2 system
        # prompt.  (compare's runtime question_type is "marked_comparison".)
        # The V2 prompt is already compare-aware (it documents the painted
        # A/B/C identity badges), so compare no longer needs the old V1 prompt
        # + Mika's patch_agentic_system_prompt — that path carried stale
        # "always answer with a letter" and existence/relative_size references.
        # Compare-specific answering guidance is injected per-turn via the type
        # constraint further below.
        from mgvqa.prompts import load_system_prompt
        system_prompt = load_system_prompt(
            "perception" if perception_only else "agentic")
    else:
        system_prompt = load_prompt("system_prompt_conversational.txt")
    # Conversation history: a list of unified-format messages
    #   {"role": "user"|"assistant", "text": str, "images": [np.ndarray]}.
    # Grows by 2 entries per turn.
    conversation: list = []

    # State tracking
    action_history = []       # list of action dicts for prompt context
    objects_removed = []      # list of (name, step_removed) tuples
    actions_taken = 0         # count of manipulation actions (grasp + push)
    grasp_attempts = 0        # count of grasp_and_remove calls
    grasp_successes = 0       # count of successful grasp_and_remove
    push_attempts = 0         # count of push calls
    push_successes = 0        # count of successful pushes
    total_steps = 0           # total VLM calls
    badge_reports = []        # retained in results schema for old-run compatibility
    final_answer = None
    predicted_point = None    # [row, col] for existence questions
    predicted_ratio = None    # numeric for relative_size questions
    final_badge_points = None # optional marked A/B/C diagnostic on final answer
    removed_object_ids = []   # simulator IDs; preserves target-removal auditability
    status = "fail"
    fail_reason = None

    # Working memory — plan/belief carried forward from VLM
    last_plan = ""
    last_belief = ""

    # Track objects that failed grasping — warn VLM not to retry them
    failed_grasp_objects = set()  # set of object names that failed grasp

    # ── Instrumentation (E15) ──
    # belief_at_turn_1: the VLM's belief/plan from step 1 (before any tool results)
    belief_at_turn_1 = None
    # pre_vs_post_perception_delta: how many times perception changed the belief
    perception_belief_changes = 0
    _prev_belief_before_perception = None
    # tool_disagreement_count: how many times the VLM's belief contradicted a tool
    tool_disagreement_count = 0

    # Current perception state (V2: scene graph + grounding separated)
    current_marker_map = {}        # marker_id → object label (from find_object)
    current_labelled_img = None    # SAM overlay, sent after successful find_object
    current_detections = []        # detection dicts from last find_object call
    has_grounding = False          # whether we have valid marker IDs for manipulation
    latest_perception_meta = None  # structured meta for grounding YAML
    current_scene_graph = None     # V2: YAML scene graph from perceive_scene_v2
    scene_graph_fresh = False      # becomes False after manipulation
    scene_graph_step = 0           # which step produced the current scene graph
    target_visible = False         # whether perceive_find_object found the target
    last_tool = None               # track last tool used
    # Persistent mental map (kept for longterm summary, but scene graph
    # is the primary mental representation now).
    mental_map = mental_map_new()
    PERCEPTION_TOOLS = {"perceive_scene", "perceive_find_object"}
    # Perception-loop guard.
    #   - WARN_IDLE_PERCEPTIONS:          from this many consecutive idle
    #                                     perceptions, the turn prompt shows a
    #                                     countdown warning.
    #   - MAX_CONSECUTIVE_PERCEPTIONS:    on this many consecutive perception-
    #                                     only calls the episode TERMINATES
    #                                     immediately and is scored as a
    #                                     FAILURE (no synthesized answer).
    perception_calls_without_action = 0
    perception_loop_failure = False  # set True on hard perception-loop termination
    WARN_IDLE_PERCEPTIONS = 3
    MAX_CONSECUTIVE_PERCEPTIONS = 5
    # ── Perception mode: manipulation is IMPOSSIBLE, so "consecutive
    # perception calls" is the expected steady state — it must NOT be scored
    # as a failure (that would nuke every perception-mode episode).  Disable
    # the hard failure guard; the idle-perception WARN countdown still nudges
    # the model to answer, and max_total_steps (below) is the real terminator.
    if perception_only:
        MAX_CONSECUTIVE_PERCEPTIONS = 10 ** 9   # effectively off
    budget_rejections = 0          # consecutive times VLM tried manipulation after budget=0
    MAX_BUDGET_REJECTIONS = 3      # hard-break after this many

    # Start video recording (PyBullet — continuous, fast)
    vid_path = f"{base_path}_vlm_video.mp4"
    demo.env.camera.start_recording(vid_path)

    def _log(msg: str):
        """Write a line to the log file only (not terminal)."""
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

    try:
        # ---- Initial setup ----
        demo.env.reset_robot()
        obs = demo.update()

        # Recapture the definitive seg AFTER robot reset + update.
        # reset_robot() runs 420+ physics steps and update() adds 20 more,
        # which can shift objects in the pile.  The seg captured here
        # reflects the true static state that the VLM will see.
        if initial_seg is not None:
            _, _, initial_seg = demo.env.camera.get_cam_img_pybullet()

        # Save system prompt to disk (once per episode) for reproducibility
        sys_prompt_path = os.path.join(viz_dir, "system_prompt.txt")
        with open(sys_prompt_path, "w", encoding="utf-8") as spf:
            spf.write(system_prompt)
        if verbose:
            print(f"\n  System prompt saved → {os.path.relpath(sys_prompt_path, ep_dir)}")

        # Snapshot the originally-spawned PyBullet body IDs.  Used by the
        # per-step sim-gt-stats dumper to compute which IDs have been
        # removed since spawn (env.obj_ids loses entries on grasp success).
        original_obj_ids = list(demo.env.obj_ids)

        # Maximum total loop iterations (perceive + grasp + answer attempts)
        max_total_steps = max_actions * 3 + 5
        # When generating SFT data, cap at sft_max_steps to avoid wasting
        # API calls on episodes that build_sft_dataset.py will discard.
        if sft_max_steps > 0:
            max_total_steps = min(max_total_steps, sft_max_steps)

        while total_steps < max_total_steps:
            total_steps += 1

            # Save simulator ground-truth state for this step.  Used
            # downstream by reward / SFT scoring scripts that need to
            # know which objects were target vs distractor at every
            # decision point in the trajectory.  See _capture_sim_gt_state
            # docstring for schema details.
            try:
                _capture_sim_gt_state(
                    demo, q_desc, occ_stats, total_steps,
                    original_obj_ids, ep_dir, demo.env.camera,
                )
            except Exception as _e:
                # Never let GT-stats dumping break the agentic loop —
                # this is purely an offline-analysis artifact.
                _log(f"  [LOG] sim-gt-stats capture failed at step "
                     f"{total_steps}: {_e}")

            # Capture Blender RGB for the VLM.  For segmentation, use the
            # cached ``initial_seg`` until a manipulation (grasp/push)
            # actually changes the scene.  PyBullet's TinyRenderer in
            # DIRECT mode is non-deterministic at object visibility edges:
            # calling getCameraImage() twice on the same static scene can
            # produce different seg masks for objects at grazing angles.
            # The definitive seg captured after robot reset guarantees
            # consistency across all pre-manipulation steps.
            rgb, _depth_pb, _seg_fresh = demo.env.camera.get_cam_img()

            if initial_seg is not None and actions_taken == 0:
                # Scene is unchanged from setup — use definitive seg.
                seg_pb = initial_seg
            else:
                # A manipulation has occurred — scene has changed, use
                # fresh seg from current PyBullet state.
                seg_pb = _seg_fresh

            # --- A/B target info for same_attribute / relative_size ---
            # Images sent to the VLM are ALWAYS clean — A/B identity is
            # communicated as text (normalized [row, col] on a [0, 1000] grid,
            # or blank if the object is below the visibility threshold).
            # See FIXES_FOR_April16.MD Issue #1 and April18 P1-1.
            markers_visible: Dict[str, bool] = {}
            ab_info: Dict[str, Dict] = {}
            if uses_ab_markers:
                ab_info = compute_ab_marker_info(
                    seg_pb, marker_obj_ids, marker_full_area,
                    visibility_threshold=0.30,
                )
                markers_visible = {k: v.get("visible", False)
                                    for k, v in ab_info.items()}

            # Published user template: status, previous result (except turn 1),
            # original question/task/options, and current image on every turn.
            # No task-specific search guidance or privileged target fields.
            images_for_vlm = [rgb]
            # Only the immediately preceding successful grounding contributes a
            # second image. Never reuse a prior mask after a failed/rejected call
            # or a manipulation. The overlay is SAM output, not simulator GT.
            has_segmentation_overlay = (
                last_tool == "perceive_find_object"
                and bool(action_history and action_history[-1].get("success"))
                and has_grounding and current_labelled_img is not None
            )
            if has_segmentation_overlay:
                images_for_vlm.append(current_labelled_img)
            remaining_budget = max_actions - actions_taken
            previous_response = ""
            if total_steps > 1:
                previous_response = format_previous_tool_response(
                    action_history[-1] if action_history else {"tool": last_tool},
                    scene_graph_text=(format_scene_graph_block(
                        current_scene_graph, exact_points=is_v2_points)
                        if current_scene_graph is not None else ""),
                    has_grounding=has_grounding,
                )
            user_prompt = build_user_prompt(
                "perception" if perception_only else "agentic",
                question=question, question_type=question_type,
                answer_options=answer_options, step=total_steps,
                max_steps=max_total_steps, last_tool=last_tool,
                previous_tool_response=previous_response,
                remaining_budget=remaining_budget, max_actions=max_actions,
                perception_calls=perception_calls_without_action,
                has_segmentation_overlay=has_segmentation_overlay,
            )

            # ── Save prompt + image to disk (same on-disk layout as v2) ──
            step_prompt_path = os.path.join(viz_dir, f"step_{total_steps}_prompt.txt")
            with open(step_prompt_path, "w", encoding="utf-8") as pf:
                pf.write(user_prompt)
            step_img_paths = []
            for image_index, image in enumerate(images_for_vlm, 1):
                image_path = os.path.join(
                    viz_dir, f"step_{total_steps}_img{image_index}.png")
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                step_img_paths.append(image_path)

            if verbose:
                print(f"\n── Step {total_steps} │ Actions: {actions_taken}/{max_actions} │ Last: {last_tool or 'none'} ──")

            _log(f"\n  [LOG] Images sent to VLM (this turn):")
            for ip in step_img_paths:
                _log(f"    → {os.path.relpath(ip, ep_dir)}")
            _log(f"  [LOG] Conversation history so far: {len(conversation)} prior turns")
            _log(f"  [LOG] Prompt ({len(user_prompt)} chars) saved → {os.path.relpath(step_prompt_path, ep_dir)}")
            _log(f"  ┌── USER PROMPT {'─'*50}")
            for line in user_prompt.splitlines():
                _log(f"  │ {line}")
            _log(f"  └{'─'*68}")

            # ── Call VLM with the full conversation history ──
            # `extra_messages` carries every prior {user, assistant} pair
            # (including the prior images).  The new user turn is
            # `user_prompt` + `images_for_vlm`.  Backend converters
            # (to_gemini_contents / to_openai_messages) handle role and
            # content shape per provider.
            # Published protocol retains all earlier images and tool responses
            # equally across models; do not silently truncate by model family.
            _extra = conversation if conversation else None

            vlm_response = call_vlm(
                images=images_for_vlm,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model_name=model_name,
                temperature=0.0,
                max_tokens=16384,
                extra_messages=_extra,
            )

            # ── Append the new (user, assistant) pair to the conversation
            # so the NEXT call sees this turn's image and JSON response. ──
            conversation.append({
                "role": "user",
                "text": user_prompt,
                "images": list(images_for_vlm),  # preserve both attachments in history
            })
            assistant_text = json.dumps(vlm_response, ensure_ascii=False)
            conversation.append({
                "role": "assistant",
                "text": assistant_text,
                "images": [],
            })

            if verbose:
                tool_str = vlm_response.get("tool", "?")
                reason_str = vlm_response.get("reasoning", "")
                repaired = vlm_response.get("_truncated_and_repaired", False)
                if tool_str == "error":
                    raw = vlm_response.get("raw_response", "(empty)")
                    print(f"  VLM → PARSE ERROR — could not extract JSON from response:")
                    for line in raw[:500].splitlines():
                        print(f"    | {line}")
                    if len(raw) > 500:
                        print(f"    | ... ({len(raw)} chars total)")
                elif repaired:
                    # Response was truncated (usually by max_tokens) but we
                    # salvaged the first few fields.  Flag it so the user
                    # can see whether to bump the budget further.
                    print(f"  VLM → {tool_str}: {reason_str}  "
                          f"[WARN: response truncated, repaired JSON]")
                else:
                    print(f"  VLM → {tool_str}: {reason_str}")

            # Save VLM response
            response_path = os.path.join(viz_dir, f"step_{total_steps}_response.json")
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(vlm_response, f, indent=2)

            # Write full VLM response to log file only (not terminal)
            _log(f"  ┌── VLM RESPONSE {'─'*49}")
            for rline in json.dumps(vlm_response, indent=2).splitlines():
                _log(f"  │ {rline}")
            _log(f"  └{'─'*68}")

            vlm_response = normalize_planner_call(vlm_response)
            tool = vlm_response["tool"]
            reasoning = vlm_response.get("reasoning", "")

            # Carry plan / belief forward (fall back to previous if VLM omitted).
            # Some models (Gemini 3 Flash, Gemini-Robotics-ER-1.6, ...) return
            # ``plan`` / ``belief`` as a JSON list of bullet strings rather than
            # a single newline-separated string.  Coerce defensively so we
            # don't crash on .strip().  See rollout_vlm_20260525_052016
            # post-mortem (148/150 episodes failed with
            # AttributeError("'list' object has no attribute 'strip'")
            # until this landed).
            def _coerce_str(v):
                if v is None:
                    return ""
                if isinstance(v, list):
                    return "\n".join(
                        str(x).strip() for x in v if str(x).strip())
                return str(v)
            new_plan = _coerce_str(vlm_response.get("plan", "")).strip()
            new_belief = _coerce_str(vlm_response.get("belief", "")).strip()
            if new_plan:
                last_plan = new_plan
            if new_belief:
                last_belief = new_belief

            # ── Instrumentation (E15) ──
            if total_steps == 1:
                belief_at_turn_1 = new_belief or new_plan or reasoning
            # Track belief changes across perception calls
            if tool in ("perceive_scene", "perceive_find_object"):
                _prev_belief_before_perception = last_belief
            elif _prev_belief_before_perception and new_belief:
                if new_belief != _prev_belief_before_perception:
                    perception_belief_changes += 1
                _prev_belief_before_perception = None

            # ---- Handle tool calls ----

            if tool in PERCEPTION_TOOLS:
                # Perception-loop guard.  Count consecutive perception calls
                # without a manipulation action.  From WARN_IDLE_PERCEPTIONS
                # on, the turn prompt shows a countdown warning.  On the
                # MAX_CONSECUTIVE_PERCEPTIONS-th consecutive perception call
                # the EPISODE ENDS IMMEDIATELY and is scored as a FAILURE
                # (no answer is synthesized — the VLM had ample warning).
                perception_calls_without_action += 1
                if perception_calls_without_action >= MAX_CONSECUTIVE_PERCEPTIONS:
                    fail_reason = (
                        f"Perception-loop failure: {perception_calls_without_action} "
                        f"consecutive perception-only calls (limit "
                        f"{MAX_CONSECUTIVE_PERCEPTIONS}); episode terminated "
                        f"without an answer."
                    )
                    if verbose:
                        print(f"[GUARD] {fail_reason}")
                    _log(f"  [GUARD] {fail_reason}")
                    status = "perception_loop_failure"
                    perception_loop_failure = True
                    final_answer = None
                    predicted_point = None
                    predicted_ratio = None
                    break

                if tool == "perceive_scene":
                    if verbose:
                        print("  Running V2 scene-graph perception...")
                    # V2: main VLM produces rich scene graph (no SAM grounding).
                    # SAM grounding is only triggered by perceive_find_object.
                    # Use pointer model (Gemini) for scene graph generation
                    # since the local SFT model was trained to receive scene
                    # graphs, not generate them.
                    _sg_model = scene_model
                    scene_graph, labelled_img_sg, mm_sg, dets_sg = \
                        execute_backend("perceive_scene", mode=mode,
                            rgb=rgb,
                            vlm_model_name=_sg_model,
                            sam_model=None,  # No grounding here — find_object handles it
                            verbose=verbose,
                            # V2: ask for PRECISE per-object points that the
                            # planner will pass to perceive_find_object.
                            precise_points=is_v2_points,
                        )

                    n_sg_objects = len(scene_graph.get("objects", []))
                    if n_sg_objects:
                        current_scene_graph = scene_graph
                        scene_graph_fresh = True
                        scene_graph_step = total_steps
                        if verbose:
                            print(f"  ✓ Scene graph: {n_sg_objects} objects, "
                                  f"summary: "
                                  f"{scene_graph.get('scene_summary', '')[:120]}")
                        # Save scene graph YAML to disk for inspection
                        sg_path = os.path.join(
                            viz_dir, f"step_{total_steps}_scene_graph.yaml")
                        try:
                            import yaml as _yaml
                            with open(sg_path, "w", encoding="utf-8") as sgf:
                                _yaml.safe_dump(scene_graph, sgf,
                                                default_flow_style=False,
                                                allow_unicode=True,
                                                sort_keys=False)
                        except Exception:
                            pass
                        # Save debug overlay: approximate points drawn on RGB.
                        # NOT sent to the VLM — debug-only, for analysis of
                        # where the scene-graph VLM placed each object.
                        # Separate name: imgN files are reserved for actual
                        # planner attachments, never debug scene-graph points.
                        try:
                            sg_overlay = draw_scene_graph_overlay(
                                rgb, scene_graph)
                            sg_overlay_path = os.path.join(
                                viz_dir,
                                f"step_{total_steps}_scene_graph_overlay.png")
                            cv2.imwrite(sg_overlay_path,
                                        cv2.cvtColor(sg_overlay,
                                                     cv2.COLOR_RGB2BGR))
                        except Exception as _e:
                            if verbose:
                                print(f"  (scene graph overlay skipped: "
                                      f"{_e})")
                    else:
                        if verbose:
                            print(f"  ✗ Scene graph empty (VLM parse failed)")

                    action_history.append({
                        "step_number": total_steps,
                        "tool": "perceive_scene",
                        "reasoning": reasoning,
                        "success": bool(n_sg_objects),
                        "n_objects_detected": n_sg_objects,
                        "scene_summary": scene_graph.get(
                            "scene_summary", "")[:200],
                    })
                    if scene_graph.get("objects"):
                        all_detected_labels.update(
                            o["name"] for o in scene_graph["objects"])

                elif tool == "perceive_find_object" and is_v2_points:
                    # A new selection request supersedes the previous one,
                    # including malformed/rejected points (no stale feedback).
                    has_grounding = False
                    current_labelled_img = None
                    current_marker_map = {}
                    current_detections = []
                    latest_perception_meta = None
                    # ── V2 grounding-by-point ──
                    # The planner reads the scene graph (precise per-object
                    # points) and passes the [row, col] point (on [0, 1000])
                    # of the object it wants to grasp/push.  SAM3 segments at
                    # that pixel directly — no pointing model, no object name.
                    requested_point = vlm_response.get("point", None)

                    if not (isinstance(requested_point, (list, tuple))
                            and len(requested_point) >= 2):
                        action_history.append({
                            "step_number": total_steps,
                            "tool": "perceive_find_object",
                            "point": requested_point,
                            "reasoning": reasoning,
                            "success": False,
                            "status": "no_point",
                            "note": ("No valid point provided. "
                                     "perceive_find_object requires "
                                     "\"point\": [row, col] on the [0, 1000] "
                                     "grid — take it from the scene graph "
                                     "object you want to grasp/push. Call "
                                     "perceive_scene first if you have not."),
                        })
                        if verbose:
                            print("  ✗ perceive_find_object (v2) called "
                                  "without a valid point — ignored")
                        last_tool = tool
                        continue

                    if verbose:
                        print(f"  Grounding at point: {requested_point}")
                    labelled_img, marker_map, detections, find_status = \
                        execute_backend("perceive_find_object", mode=mode,
                            rgb=rgb,
                            point=requested_point,
                            sam_model=sam_model,
                            scene_graph=current_scene_graph,
                            verbose=verbose,
                        )
                    n_instances = len(detections) if detections else 0
                    _pt_label = (detections[0].get("label")
                                 if detections else "object")

                    # target_visible is a name-based V1 concept; in the
                    # point-based V2 flow the point IS the target designation.
                    target_visible = (find_status == "found")

                    if find_status == "found" and detections:
                        current_labelled_img = labelled_img
                        current_marker_map = marker_map
                        current_detections = detections
                        has_grounding = True
                        latest_perception_meta = build_perception_meta(
                            "find_object", marker_map, detections,
                            rgb.shape[:2],
                            query=str(requested_point), status=find_status)
                        latest_perception_meta["labelled_img"] = labelled_img
                        if verbose:
                            print(f"  ✓ Segmented object at "
                                  f"{requested_point} → '{_pt_label}' "
                                  f"(marker 1)")
                    else:
                        has_grounding = False
                        current_labelled_img = None
                        current_marker_map = {}
                        current_detections = []
                        latest_perception_meta = build_perception_meta(
                            "find_object", {}, [], rgb.shape[:2],
                            query=str(requested_point), status="not_found")
                        if verbose:
                            print(f"  ✗ No object segmented at "
                                  f"{requested_point} (status={find_status})")

                    action_history.append({
                        "step_number": total_steps,
                        "tool": "perceive_find_object",
                        "point": list(requested_point[:2]),
                        "object_name": _pt_label,
                        "reasoning": reasoning,
                        "success": find_status == "found",
                        "status": find_status,
                        "n_instances": n_instances,
                        "n_objects_detected": n_instances,
                        "marker_map": dict(marker_map) if marker_map else {},
                    })
                    if marker_map:
                        all_detected_labels.update(marker_map.values())

                    # Save tool artifacts; the SAM overlay is also attached
                    # alongside clean RGB on the next planner turn.
                    if current_detections:
                        pt_img = draw_pointing_debug(
                            rgb, current_detections, current_marker_map)
                        cv2.imwrite(
                            os.path.join(viz_dir,
                                         f"step_{total_steps}_point.png"),
                            cv2.cvtColor(pt_img, cv2.COLOR_RGB2BGR))
                    if current_labelled_img is not None:
                        cv2.imwrite(
                            os.path.join(viz_dir,
                                         f"step_{total_steps}_seg.png"),
                            cv2.cvtColor(current_labelled_img,
                                         cv2.COLOR_RGB2BGR))
                    last_tool = tool
                    continue

                elif tool == "perceive_find_object":
                    has_grounding = False
                    current_labelled_img = None
                    current_marker_map = {}
                    current_detections = []
                    latest_perception_meta = None
                    requested_name = vlm_response.get("object_name", "").strip()

                    if not requested_name:
                        # V2: no object name → advise the VLM to call
                        # perceive_scene first.  We do NOT silently run a
                        # full scene scan via the pointer — that would
                        # bring back the old entangled pipeline.
                        action_history.append({
                            "step_number": total_steps,
                            "tool": "perceive_find_object",
                            "object_name": "",
                            "reasoning": reasoning,
                            "success": False,
                            "status": "no_object_name",
                            "note": ("No object_name provided. In V2, "
                                     "perceive_find_object requires an "
                                     "object name for precise grounding. "
                                     "Use perceive_scene for holistic "
                                     "scene understanding."),
                        })
                        if verbose:
                            print("  ✗ perceive_find_object called without "
                                  "object_name — ignored")
                        last_tool = tool
                        continue

                    if verbose:
                        print(f"  Grounding: '{requested_name}'")
                    labelled_img, marker_map, detections, find_status = \
                        perceive_find_object(
                            rgb=rgb,
                            target_name=requested_name,
                            molmo_processor=molmo_processor,
                            molmo_model=molmo_model,
                            sam_model=sam_model,
                            verbose=verbose,
                            target_display_name=requested_name,
                        )
                    n_instances = len(detections) if detections else 0

                    target_visible = (find_status == "found"
                                      and target_name
                                      and requested_name == target_name)

                    if find_status == "found" and detections:
                        current_labelled_img = labelled_img
                        current_marker_map = marker_map
                        current_detections = detections
                        has_grounding = True
                        latest_perception_meta = build_perception_meta(
                            "find_object", marker_map, detections,
                            rgb.shape[:2],
                            query=requested_name, status=find_status)
                        latest_perception_meta["labelled_img"] = labelled_img
                        if verbose:
                            print(f"  ✓ Grounded {n_instances} instance(s) of "
                                  f"'{requested_name}'")
                    else:
                        has_grounding = False
                        current_labelled_img = None
                        current_marker_map = {}
                        current_detections = []
                        latest_perception_meta = build_perception_meta(
                            "find_object", {}, [], rgb.shape[:2],
                            query=requested_name, status="not_found")
                        if verbose:
                            print(f"  ✗ '{requested_name}' not visible — "
                                  f"grounding model could not locate it")

                    action_history.append({
                        "step_number": total_steps,
                        "tool": "perceive_find_object",
                        "object_name": requested_name,
                        "reasoning": reasoning,
                        "success": find_status == "found",
                        "status": find_status,
                        "n_instances": n_instances,
                        "n_objects_detected": n_instances,
                        "marker_map": dict(marker_map) if marker_map else {},
                    })
                    if marker_map:
                        all_detected_labels.update(marker_map.values())

                else:
                    # Unknown perception tool (e.g. legacy perceive_get_obstructors)
                    action_history.append({
                        "step_number": total_steps,
                        "tool": tool,
                        "reasoning": reasoning,
                        "success": False,
                        "note": f"Unknown perception tool '{tool}'",
                    })

                # Save debug images (NOT sent to VLM)
                if current_detections:
                    # Point debug: raw pointing-model output before SAM
                    pt_img = draw_pointing_debug(
                        rgb, current_detections, current_marker_map)
                    pt_path = os.path.join(
                        viz_dir, f"step_{total_steps}_point.png")
                    cv2.imwrite(pt_path, cv2.cvtColor(pt_img, cv2.COLOR_RGB2BGR))
                if current_labelled_img is not None:
                    # Seg debug: SAM masks with numbered labels
                    seg_path = os.path.join(
                        viz_dir, f"step_{total_steps}_seg.png")
                    cv2.imwrite(seg_path, cv2.cvtColor(
                        current_labelled_img, cv2.COLOR_RGB2BGR))

                last_tool = tool
                continue

            elif tool == "grasp_and_remove":
                if perception_only:
                    action_history.append({
                        "step_number": total_steps, "tool": "grasp_and_remove",
                        "reasoning": reasoning, "success": False,
                        "note": "Manipulation is unavailable in this perception-only condition. "
                                "Use perception tools or answer from the unchanged scene.",
                    })
                    last_tool = "grasp_and_remove"
                    continue
                if actions_taken >= max_actions:
                    budget_rejections += 1
                    action_history.append({
                        "step_number": total_steps,
                        "tool": "grasp_and_remove",
                        "reasoning": reasoning,
                        "success": False,
                        "note": "REJECTED — manipulation budget exhausted. "
                                "You must call answer_question now.",
                    })
                    if budget_rejections >= MAX_BUDGET_REJECTIONS:
                        fail_reason = (
                            f"VLM requested manipulation {MAX_BUDGET_REJECTIONS} times "
                            f"after budget was exhausted. Terminating."
                        )
                        if verbose:
                            print(f"  [GUARD] {fail_reason}")
                        break
                    last_tool = "grasp_and_remove"
                    if verbose:
                        print(f"  ✗ Budget exhausted — action rejected "
                              f"({budget_rejections}/{MAX_BUDGET_REJECTIONS})")
                    continue

                if not has_grounding:
                    action_history.append({
                        "tool": "grasp_and_remove",
                        "reasoning": reasoning,
                        "success": False,
                        "note": (
                            "No grounding available. "
                            "Call perceive_find_object with the \"point\" "
                            "[row, col] of the object you want to grasp "
                            "(from the scene graph) — this returns marker "
                            "object_id = 1 to pass to grasp_and_remove."
                            if is_v2_points else
                            "No grounding available. "
                            "Call perceive_find_object with the object name "
                            "you want to grasp — this gives you a valid "
                            "marker ID to pass to grasp_and_remove."
                        ),
                    })
                    last_tool = "grasp_and_remove"
                    continue

                marker_id = vlm_response.get("object_id", None)
                if marker_id is None:
                    action_history.append({
                        "tool": "grasp_and_remove",
                        "reasoning": reasoning,
                        "success": False,
                        "note": "No object_id provided",
                    })
                    pass  # no object_id
                    last_tool = "grasp_and_remove"
                    continue

                marker_id = int(marker_id)
                obj_name_from_marker = current_marker_map.get(marker_id, None)

                if obj_name_from_marker is None:
                    action_history.append({
                        "tool": "grasp_and_remove",
                        "object_id": marker_id,
                        "reasoning": reasoning,
                        "success": False,
                        "note": f"Marker ID {marker_id} not found in current perception",
                    })
                    pass  # invalid marker
                    last_tool = "grasp_and_remove"
                    continue

                # (name-match guard removed — protection is by obj ID below)

                # Find closest env object to the detection point
                # The detection point from Molmo gives us a rough location
                det_for_marker = None
                for det_i, det in enumerate(current_detections):
                    det_marker_idx = det_i + 1
                    if det_marker_idx == marker_id:
                        det_for_marker = det
                        break

                # Try to match the detected object to an env object by
                # finding which env object's center is closest to the
                # detection point, or whose segmentation mask overlaps
                # with the SAM3 mask
                actual_obj_id = None
                actual_obj_name = obj_name_from_marker

                if det_for_marker is not None and "mask" in det_for_marker:
                    det_mask = det_for_marker["mask"]
                    seg = obs.get("seg", None)
                    if seg is not None:
                        # Find the env object ID with highest overlap
                        best_overlap = 0
                        for oid, oname in zip(demo.env.obj_ids, demo.env.obj_names):
                            env_mask = (seg == oid)
                            overlap = int((det_mask & env_mask).sum())
                            if overlap > best_overlap:
                                best_overlap = overlap
                                actual_obj_id = oid
                                actual_obj_name = oname

                if actual_obj_id is None:
                    # Fallback: try to match by name
                    for oid, oname in zip(demo.env.obj_ids, demo.env.obj_names):
                        if obj_name_from_marker.lower().replace(" ", "_") in oname.lower():
                            actual_obj_id = oid
                            actual_obj_name = oname
                            break

                if actual_obj_id is None:
                    action_history.append({
                        "tool": "grasp_and_remove",
                        "object_id": marker_id,
                        "object_name": obj_name_from_marker,
                        "reasoning": reasoning,
                        "success": False,
                        "note": "Could not match detected object to env object",
                    })
                    pass  # no env match
                    last_tool = "grasp_and_remove"
                    continue

                if verbose:
                    print(f"  Grasping '{actual_obj_name}'...(ID from annotation, might differ from VLM label)")

                # ── 6-DOF GraspGen (3 attempts, angle-first), SAM3 masks ──
                sam_mask = det_for_marker.get("mask") if det_for_marker else None
                grasp_success = False

                if sam_mask is not None and graspgen_sampler is not None:
                    # Enable motion recording for manipulation
                    demo.env.set_recording_motion(True)

                    grasp_result = execute_backend("grasp_and_remove", mode=mode,
                        demo=demo,
                        obj_id=actual_obj_id,
                        sam_mask=sam_mask,
                        sam_model=sam_model,
                        graspgen_sampler=graspgen_sampler,
                        verbose=verbose,
                    )
                    grasp_success = grasp_result["success"]

                    # Stop motion recording
                    demo.env.set_recording_motion(False)
                else:
                    if verbose:
                        reason = "no SAM mask" if sam_mask is None else "no GraspGen"
                        print(f"  ✗ Cannot grasp: {reason}")

                actions_taken += 1
                grasp_attempts += 1
                # Three-way physical outcome (the tool returns two bits):
                #   placed          → lifted AND resting in the bin  → success
                #   grasped_slipped → lifted but slipped before the bin
                #                     (object is still somewhere in the scene)
                #   failed          → never got a hold on the object
                _lifted = bool(grasp_result.get("grasped", False)) \
                    if (sam_mask is not None and graspgen_sampler is not None) else False
                if grasp_success:
                    grasp_outcome = "placed"
                elif _lifted:
                    grasp_outcome = "grasped_slipped"
                else:
                    grasp_outcome = "failed"

                action_entry = {
                    "step_number": total_steps,
                    "tool": "grasp_and_remove",
                    "object_id": marker_id,
                    "object_name": actual_obj_name,
                    "reasoning": reasoning,
                    "success": grasp_success,
                    "grasp_outcome": grasp_outcome,
                }
                if sam_mask is not None and graspgen_sampler is not None:
                    action_entry["n_grasp_attempts"] = grasp_result.get("n_attempts", 0)
                    action_entry["grasped"] = _lifted
                action_history.append(action_entry)

                if grasp_success:
                    grasp_successes += 1
                    objects_removed.append((actual_obj_name, total_steps))
                    removed_object_ids.append(int(actual_obj_id))
                    failed_grasp_objects.discard(actual_obj_name)  # clear if previously failed
                    if verbose:
                        print(f"  ✓ removed → bin ({grasp_successes}/{grasp_attempts} grasps)")
                elif grasp_outcome == "grasped_slipped":
                    # The grasp mechanics worked (object was lifted) — do NOT
                    # blacklist it as "cannot grasp".  It merely slipped and
                    # has likely moved; the VLM should re-perceive and may
                    # retry the grasp or push it off instead.
                    if verbose:
                        print(f"  ⚠ grasped but slipped before bin — object "
                              f"moved, still in scene ({grasp_successes}/{grasp_attempts} placed)")
                else:
                    failed_grasp_objects.add(actual_obj_name)
                    if verbose:
                        print(f"  ✗ failed ({grasp_successes}/{grasp_attempts} grasps)")

                # Update observation after action
                demo.env.reset_robot()
                obs = demo.update()

                # Detect whether the scene actually changed by comparing
                # object pixels in the seg buffer before vs after.
                _, _, _post_seg = demo.env.camera.get_cam_img_pybullet()
                _first_obj_id = min(demo.env.obj_ids) if demo.env.obj_ids else 6
                _pre_obj_mask = (seg_pb >= _first_obj_id)
                _post_obj_mask = (_post_seg >= _first_obj_id)
                _scene_changed = not np.array_equal(
                    seg_pb * _pre_obj_mask, _post_seg * _post_obj_mask)

                if _scene_changed:
                    # Invalidate grounding + scene graph
                    has_grounding = False
                    current_marker_map = {}
                    current_labelled_img = None
                    current_detections = []
                    target_visible = False
                    perception_calls_without_action = 0
                    mental_map = mental_map_new()
                    if latest_perception_meta is not None:
                        latest_perception_meta["fresh"] = False
                    scene_graph_fresh = False
                    # Use fresh seg from now on
                    initial_seg = None
                else:
                    # Scene unchanged — keep scene graph fresh, reuse seg
                    if verbose:
                        print(f"  ℹ Scene unchanged after grasp — "
                              f"keeping scene graph fresh")

                last_tool = "grasp_and_remove"
                continue

            elif tool == "push":
                # ── Push tool: slide an object in a direction ──
                if perception_only:
                    action_history.append({
                        "step_number": total_steps, "tool": "push",
                        "reasoning": reasoning, "success": False,
                        "note": "Manipulation is unavailable in this perception-only condition. "
                                "Use perception tools or answer from the unchanged scene.",
                    })
                    last_tool = "push"
                    continue
                if actions_taken >= max_actions:
                    budget_rejections += 1
                    action_history.append({
                        "step_number": total_steps,
                        "tool": "push",
                        "reasoning": reasoning,
                        "success": False,
                        "note": "REJECTED — manipulation budget exhausted. "
                                "You must call answer_question now.",
                    })
                    if budget_rejections >= MAX_BUDGET_REJECTIONS:
                        fail_reason = (
                            f"VLM requested manipulation {MAX_BUDGET_REJECTIONS} times "
                            f"after budget was exhausted. Terminating."
                        )
                        if verbose:
                            print(f"  [GUARD] {fail_reason}")
                        break
                    last_tool = "push"
                    if verbose:
                        print(f"  ✗ Budget exhausted — action rejected "
                              f"({budget_rejections}/{MAX_BUDGET_REJECTIONS})")
                    continue

                if not has_grounding:
                    action_history.append({
                        "tool": "push",
                        "reasoning": reasoning,
                        "success": False,
                        "note": (
                            "No grounding available. "
                            "Call perceive_find_object with the \"point\" "
                            "[row, col] of the object you want to push "
                            "(from the scene graph) — this returns marker "
                            "object_id = 1 to pass to push."
                            if is_v2_points else
                            "No grounding available. "
                            "Call perceive_find_object with the object name "
                            "you want to push — this gives you a valid "
                            "marker ID to pass to push."
                        ),
                    })
                    last_tool = "push"
                    continue

                push_marker_id = vlm_response.get("object_id", None)
                push_direction = vlm_response.get("direction", "").strip().lower()
                push_distance_cm = vlm_response.get("distance_cm", None)  # optional

                if push_marker_id is None:
                    action_history.append({
                        "tool": "push",
                        "reasoning": reasoning,
                        "success": False,
                        "note": "No object_id provided",
                    })
                    last_tool = "push"
                    continue

                if push_direction not in ("left", "right", "up", "down"):
                    action_history.append({
                        "tool": "push",
                        "object_id": push_marker_id,
                        "reasoning": reasoning,
                        "success": False,
                        "note": (
                            f"Invalid direction '{push_direction}'. "
                            f"Use one of: left, right, up, down."
                        ),
                    })
                    last_tool = "push"
                    continue

                push_marker_id = int(push_marker_id)
                push_obj_name = current_marker_map.get(push_marker_id, None)

                if push_obj_name is None:
                    action_history.append({
                        "tool": "push",
                        "object_id": push_marker_id,
                        "reasoning": reasoning,
                        "success": False,
                        "note": f"Marker ID {push_marker_id} not found in current perception",
                    })
                    last_tool = "push"
                    continue

                # Find the detection for this marker
                push_det = None
                for det_i, det in enumerate(current_detections):
                    if det_i + 1 == push_marker_id:
                        push_det = det
                        break

                # Match marker to PyBullet object via SAM mask overlap
                push_obj_id = None
                push_actual_name = push_obj_name

                if push_det is not None and "mask" in push_det:
                    push_det_mask = push_det["mask"]
                    push_seg = obs.get("seg", None)
                    if push_seg is not None:
                        best_ov = 0
                        for oid, oname in zip(demo.env.obj_ids, demo.env.obj_names):
                            ov = int((push_det_mask & (push_seg == oid)).sum())
                            if ov > best_ov:
                                best_ov = ov
                                push_obj_id = oid
                                push_actual_name = oname

                if push_obj_id is None:
                    # Fallback: name match
                    for oid, oname in zip(demo.env.obj_ids, demo.env.obj_names):
                        if push_obj_name.lower().replace(" ", "_") in oname.lower():
                            push_obj_id = oid
                            push_actual_name = oname
                            break

                if push_obj_id is None:
                    action_history.append({
                        "tool": "push",
                        "object_id": push_marker_id,
                        "object_name": push_obj_name,
                        "reasoning": reasoning,
                        "success": False,
                        "note": "Could not match to env object",
                    })
                    last_tool = "push"
                    continue

                if verbose:
                    print(f"  Pushing '{push_actual_name}' {push_direction}...")

                # Build target mask + occupancy mask from SAM detections
                push_target_mask = push_det.get("mask") if push_det else None
                if push_target_mask is None:
                    action_history.append({
                        "tool": "push",
                        "object_id": push_marker_id,
                        "object_name": push_actual_name,
                        "direction": push_direction,
                        "reasoning": reasoning,
                        "success": False,
                        "note": "No SAM mask available for this object",
                    })
                    last_tool = "push"
                    continue

                push_occupancy = build_sam_occupancy(
                    current_detections, push_target_mask.shape[:2])

                # Enable motion recording
                demo.env.set_recording_motion(True)

                push_result = execute_backend("push", mode=mode,
                    demo=demo,
                    obj_id=push_obj_id,
                    direction=push_direction,
                    target_mask=push_target_mask,
                    occupancy_mask=push_occupancy,
                    distance_cm=push_distance_cm,
                    verbose=verbose,
                )

                demo.env.set_recording_motion(False)

                actions_taken += 1
                push_attempts += 1
                push_success = push_result["success"]
                if push_success:
                    push_successes += 1

                action_history.append({
                    "step_number": total_steps,
                    "tool": "push",
                    "object_id": push_marker_id,
                    "object_name": push_actual_name,
                    "direction": push_direction,
                    "distance_cm": push_result.get("distance_cm"),
                    "reasoning": reasoning,
                    "success": push_success,
                    "n_push_attempts": push_result.get("n_attempts", 0),
                })

                if verbose:
                    if push_success:
                        print(f"  ✓ pushed ({actions_taken}/{max_actions})")
                    else:
                        print(f"  ✗ push failed ({actions_taken}/{max_actions})")

                # Update observation after action
                demo.env.reset_robot()
                obs = demo.update()

                # Detect whether the scene actually changed by comparing
                # object pixels in the seg buffer before vs after.
                _, _, _post_seg = demo.env.camera.get_cam_img_pybullet()
                _first_obj_id = min(demo.env.obj_ids) if demo.env.obj_ids else 6
                _pre_obj_mask = (seg_pb >= _first_obj_id)
                _post_obj_mask = (_post_seg >= _first_obj_id)
                _scene_changed = not np.array_equal(
                    seg_pb * _pre_obj_mask, _post_seg * _post_obj_mask)

                if _scene_changed:
                    # Invalidate grounding + scene graph
                    has_grounding = False
                    current_marker_map = {}
                    current_labelled_img = None
                    current_detections = []
                    target_visible = False
                    perception_calls_without_action = 0
                    mental_map = mental_map_new()
                    if latest_perception_meta is not None:
                        latest_perception_meta["fresh"] = False
                    scene_graph_fresh = False
                    # Use fresh seg from now on
                    initial_seg = None
                else:
                    # Scene unchanged — keep scene graph fresh, reuse seg
                    if verbose:
                        print(f"  ℹ Scene unchanged after push — "
                              f"keeping scene graph fresh")

                last_tool = "push"
                continue

            elif tool == "report_badges":
                # Optional, nullable diagnostic. The harness never tells the
                # model whether a point is right and never makes it an answer
                # prerequisite; final answer accuracy remains the main outcome.
                action_history.append({
                    "step_number": total_steps, "tool": "report_badges",
                    "reasoning": reasoning, "success": False,
                    "note": "report_badges is not an available tool in this experiment. "
                            "Put nullable badge_points in answer_question instead.",
                })
                last_tool = "report_badges"
                continue

            elif tool == "answer_question":
                submitted = execute_backend("answer_question", mode=mode,
                                            response=vlm_response)
                final_answer = submitted["answer"]
                predicted_point = submitted["point"]
                predicted_ratio = submitted["ratio"]
                if question_type == "marked_comparison":
                    _bp = vlm_response.get("badge_points")
                    # Accept either the documented nested object or direct
                    # A/B/C fields on answer_question; both are diagnostics.
                    if not isinstance(_bp, dict):
                        _bp = ({L: vlm_response.get(L) for L in ("A", "B", "C")}
                               if any(L in vlm_response for L in ("A", "B", "C"))
                               else None)
                    final_badge_points = _bp

                action_entry = {
                    "tool": "answer_question",
                    "answer": final_answer,
                    "reasoning": reasoning,
                }
                if predicted_point is not None:
                    action_entry["point"] = predicted_point
                if predicted_ratio is not None:
                    action_entry["ratio"] = predicted_ratio
                if final_badge_points is not None:
                    action_entry["badge_points"] = final_badge_points
                if badge_reports:
                    action_entry["badge_reports"] = badge_reports
                action_history.append(action_entry)

                if final_answer is not None:
                    status = "answered"
                elif question_type == "rearrangement":
                    # V2 rearrangement is scored on the FINAL WORLD STATE,
                    # not a text answer.  Calling answer_question is the
                    # explicit "I'm done rearranging" signal — treat it as
                    # answered even with no answer text.
                    status = "answered"
                    final_answer = ""  # sentinel: terminated intentionally
                else:
                    fail_reason = "VLM called answer_question but provided no answer."

                if verbose:
                    # Resolve letter to option text for readable output.
                    # NOTE: count answers are bare ints (no options / letters),
                    # so guard the letter path against non-string answers.
                    resolved = final_answer
                    _fa_str = final_answer if isinstance(final_answer, str) else None
                    if _fa_str and len(_fa_str.strip()) == 1 and _fa_str.strip().upper() in "ABCDEFGHIJ":
                        idx = ord(_fa_str.strip().upper()) - ord("A")
                        if idx < len(answer_options):
                            resolved = f"{final_answer} ({answer_options[idx]})"
                    pt_str = f"  point={predicted_point}" if predicted_point else ""
                    ratio_str = f"  ratio={predicted_ratio}" if predicted_ratio is not None else ""
                    print(f"  Answer: {resolved}  (GT: {gt_answer}){pt_str}{ratio_str}")
                break

            elif tool == "error":
                raw = vlm_response.get("raw_response", "")
                action_history.append({
                    "step_number": total_steps,
                    "tool": "error",
                    "note": "VLM response could not be parsed as JSON. "
                            "Expected {\"tool\": \"...\", ...}. "
                            f"Got: {raw[:200]}",
                })
                last_tool = "error"
                continue

            else:
                action_history.append({"step_number": total_steps, "tool": tool, "reasoning": reasoning})
                last_tool = tool
                continue

        else:
            # Loop exhausted without answer
            fail_reason = f"Maximum VLM steps exceeded ({total_steps} steps)."

    finally:
        demo.env.camera.stop_recording()
        reencode_video(vid_path)

    if status != "answered" and fail_reason is None:
        fail_reason = "Episode ended without a final answer."

    if fail_reason and verbose:
        print(f"\n  ── TERMINATED: {fail_reason}")
        print(f"     Steps: {total_steps} | Actions: {actions_taken}/{max_actions} "
              f"| Grasps: {grasp_successes}/{grasp_attempts} "
              f"| Pushes: {push_successes}/{push_attempts}")

    # Final world state for V2 rearrangement F1 scoring: which originally
    # spawned PyBullet body IDs are still present at answer time.  Captured
    # here (before the finally-block closes the env) so the worker can
    # compute removed = original − remaining without re-querying PyBullet.
    try:
        _remaining_obj_ids = [int(o) for o in demo.env.obj_ids]
    except Exception:
        _remaining_obj_ids = []

    # ── V2 rearrangement: visibility-based removal (FoV boundary + 10%) ──
    # An object counts as "removed from view" when ≤ ``rearrange_visible_thresh``
    # of its spawn-time unoccluded footprint still renders in the current
    # top-down FoV.  This credits objects pushed off-frame / off the table /
    # carried into the tray, WITHOUT falsely crediting an object that is
    # merely covered by a still-present neighbour (its unoccluded footprint
    # stays large).  See RobotEnvUI.compute_infov_footprint.
    _remaining_visible_obj_ids = list(_remaining_obj_ids)
    if question_type == "rearrangement":
        try:
            _spawn_full = (occ_stats or {}).get("full_area", {}) or {}
            _now_fp = demo.compute_infov_footprint(_remaining_obj_ids)
            _still_present = []
            for _oid in _remaining_obj_ids:
                _full = _spawn_full.get(_oid, _spawn_full.get(str(_oid), 0))
                _now = _now_fp.get(_oid, 0)
                # If we lack a spawn footprint, fall back to "present in
                # obj_ids" == present.  Otherwise use the visibility ratio.
                if _full and _full > 0:
                    _ratio = _now / float(_full)
                    if _ratio > rearrange_visible_thresh:
                        _still_present.append(_oid)
                else:
                    _still_present.append(_oid)
            _remaining_visible_obj_ids = _still_present
            if verbose:
                _removed_by_vis = sorted(set(_remaining_obj_ids)
                                         - set(_still_present))
                print(f"  [rearrange] visibility removal (≤"
                      f"{rearrange_visible_thresh:.0%} of spawn footprint): "
                      f"removed_from_view={_removed_by_vis}")
        except Exception as _e:
            if verbose:
                print(f"  [rearrange] visibility-removal check failed "
                      f"({_e}); falling back to obj_ids presence.")
            _remaining_visible_obj_ids = list(_remaining_obj_ids)

    return {
        "status": status,
        "answer": final_answer,
        "predicted_point": predicted_point,
        "predicted_ratio": predicted_ratio,
        "question": question,
        "target_name": target_name,
        "total_vlm_steps": total_steps,
        "manipulation_actions": actions_taken,
        "grasp_attempts": grasp_attempts,
        "grasp_successes": grasp_successes,
        "push_attempts": push_attempts,
        "push_successes": push_successes,
        "objects_removed": [name for name, _ in objects_removed],
        "original_obj_ids": [int(o) for o in original_obj_ids],
        "remaining_obj_ids": _remaining_obj_ids,
        "remaining_visible_obj_ids": [int(o) for o in _remaining_visible_obj_ids],
        "removed_object_ids": removed_object_ids,
        "badge_points": final_badge_points,
        "badge_reports": badge_reports,
        "action_history": action_history,
        "fail_reason": fail_reason,
        "perception_loop_failure": perception_loop_failure,
        "all_detected_labels": sorted(all_detected_labels),
        # ── Instrumentation (E15) ──
        "belief_at_turn_1": belief_at_turn_1,
        "perception_belief_changes": perception_belief_changes,
        "tool_disagreement_count": tool_disagreement_count,
    }
