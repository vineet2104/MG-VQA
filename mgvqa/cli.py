"""
core.cli — Command-line argument parsing for MG-VQA-Bench workers.

This module defines the full CLI for single-worker mode (used by run.py
and demo_conversational.py). The orchestrator (orchestrator/parallel.py)
has its own lighter arg parser that forwards most flags to the worker.
"""

import argparse
import json
import os
from models import VLM_MODEL_NAME
from questions import QUESTION_TYPES, QUESTION_TYPES_V2, EXTRA_QUESTION_TYPES
from mgvqa.utils import MAX_ACTIONS_PER_SCENE

# ─── Benchmark defaults ────────────────────────────────────────────────
# These encode the frozen MG-VQA-Bench 150-task evaluation settings.
# Override via CLI only for SFT data generation or debugging.
BENCHMARK_EPISODE_SUBSET = ""  # the frozen snapshot bank defines the evaluation set
BENCHMARK_SEED = 1000000
BENCHMARK_ANNOTATIONS = "owg_robot/assets/object_annotations_eval.json"


def _infer_n_episodes_from_subset(subset_path: str) -> int:
    """Read n_episodes_total from a benchmark JSON, or return a safe default."""
    try:
        with open(subset_path, "r") as f:
            doc = json.load(f)
        return doc.get("n_episodes_total", 380)
    except (OSError, json.JSONDecodeError, TypeError):
        return 380


def normalize_mode(mode: str) -> str:
    """Canonicalize the evaluation mode.

    'singlepass' is a deprecated alias for 'direct' (kept for back-compat with
    older shell scripts and the SFT tooling).  Returns one of
    {'direct', 'perception', 'agentic'}.
    """
    return "direct" if mode == "singlepass" else mode


def parse_args():
    p = argparse.ArgumentParser(
        description="MG-VQA-Bench: VLM-driven agentic/singlepass evaluation "
                    "on cluttered tabletop scenes."
    )
    p.add_argument("--mode", type=str, default="agentic",
                   choices=["direct", "perception", "agentic", "singlepass"],
                   help="Evaluation mode / tool-access regime: "
                        "'direct' = one image, no tools (was 'singlepass'); "
                        "'perception' = perceive_scene + perceive_find_object "
                        "only, cannot manipulate the scene; "
                        "'agentic' = full tool loop (perception + "
                        "grasp_and_remove + push). "
                        "'singlepass' is a DEPRECATED alias for 'direct'.")
    p.add_argument("--model_name", type=str, default=VLM_MODEL_NAME,
                   help="VLM model name for planning. "
                        "NVIDIA-routed: 'azure/openai/gpt-5.2', "
                        "'gcp/google/gemini-3-flash-preview'. "
                        "Google direct: 'gemini-*'. "
                        "OpenAI direct: 'gpt-4o'. "
                        "Local: 'local/<model_id>'.")
    p.add_argument("--n_workers", type=int, default=None,
                   help="Number of parallel workers. When set, routes to "
                        "the orchestrator. Default: None (single-process).")
    p.add_argument("--save_blend_anim", type=int, default=0,
                   help="Save .blend animation file per episode. Default: 0.")
    p.add_argument("--save_snapshot", type=int, default=1,
                   help="Save a frozen scene snapshot (settled poses + table "
                        "texture) per episode after physics settles, before "
                        "the first VLM call. Skipped automatically when the "
                        "episode was itself loaded from a snapshot. Default: 1.")
    p.add_argument("--from_snapshot", type=str, default="",
                   help="Directory of pre-saved scene snapshots to restore "
                        "from instead of spawning (teleport-restore, no "
                        "physics). Expects <dir>/ep_NNN_*/scene_snapshot/. "
                        "Default: '' (spawn fresh).")
    p.add_argument("--snapshots_only", type=int, default=0,
                   help="Snapshot-generation mode: spawn + settle each scene, "
                        "save question.json / occlusion.json / gt_scene.png / "
                        "seg_overlay.png / scene_snapshot/ / review.json, then "
                        "STOP — no VLM, no perception/grasp models, no "
                        "vlm_steps/, no sim-gt-stats/. Used to build the "
                        "human-review question bank. Default: 0.")

    # ── Benchmark settings (rarely need to override) ──
    p.add_argument("--n_episodes", type=int, default=None,
                   help="Total episode slots. Auto-inferred from "
                        "--episode_subset JSON if not set (default: 380).")
    p.add_argument("--seed", type=int, default=BENCHMARK_SEED,
                   help="Master seed. Default: 1000000 (frozen eval benchmark).")
    p.add_argument("--episode_subset", type=str,
                   default=BENCHMARK_EPISODE_SUBSET,
                   help="Benchmark episode list JSON. Set to '' to disable.")
    p.add_argument("--annotations_full", type=str,
                   default=BENCHMARK_ANNOTATIONS,
                   help="Object annotations file. Default: EVAL pool.")
    from models.catalog import DEFAULT_SCENE_MODEL
    p.add_argument("--scene_model", default=DEFAULT_SCENE_MODEL,
                   help="Fixed perceive_scene backend, independent of planner.")
    p.add_argument("--pointing_model", type=str, default="vlm",
                   choices=["roborefer", "molmo", "vlm"],
                   help="Pointing / grounding model. Default: vlm.")
    p.add_argument("--pointing_vlm_model", type=str,
                   default="gcp/google/gemini-3.1-pro-preview",
                   help="VLM model for pointing (when --pointing_model=vlm).")
    p.add_argument("--max_eps_per_subprocess", type=int, default=8,
                   help="Max episodes before worker recycle (0=unlimited). "
                        "Default: 8 (avoids PyBullet memory leak OOM).")

    # ── Advanced settings ──
    p.add_argument("--n_objects", type=int, default=15,
                   help="Number of objects to spawn per scene.")
    p.add_argument("--save_dir", type=str, default="rollout_vlm",
                   help="Base directory for saving results.")
    p.add_argument("--vis", type=int, default=1,
                   help="Enable Blender scene-sync + video capture (1=on, 0=off).")
    p.add_argument("--save_spawn_screenshots", type=int, default=0,
                   help="Save per-object spawn screenshots + video.")
    p.add_argument("--verbose", type=int, default=1,
                   help="Verbosity level (0=quiet, 1=normal).")
    p.add_argument("--asset_path", type=str,
                   default="./owg_robot/assets/unified_objects",
                   help="Path to object asset directory.")
    p.add_argument("--annotations", type=str, default=None,
                   help="Path to annotations.json (legacy).")
    p.add_argument("--min_dim_cm", type=float, default=2.0,
                   help="Min smallest dimension (cm) for objects.")
    p.add_argument("--max_graspable_dim_cm", type=float, default=8.0,
                   help="Max smallest dimension (cm) for graspability filter.")
    p.add_argument("--question_type", type=str, default="random",
                   choices=QUESTION_TYPES + QUESTION_TYPES_V2
                           + EXTRA_QUESTION_TYPES + ["random"],
                   help="Question type. 'random' distributes equally across "
                        "the active benchmark version's types.")
    p.add_argument("--benchmark_version", type=str, default="v1",
                   choices=["v1", "v2"],
                   help="Benchmark suite: 'v1' (6-task) or 'v2' (4-task: "
                        "count/find/beneath/compare). Set by run_v2.py.")
    p.add_argument("--question_phrasing", type=str, default="template",
                   choices=["template", "llm"],
                   help="How question text is phrased. 'template' (default): "
                        "hand-written variant pool. 'llm': natural phrasing "
                        "from --phrasing_model for the V2 tasks "
                        "(count/find/beneath/compare); ground truth and answer-"
                        "format suffix are unchanged. Cached to "
                        ".cache/llm_question_phrasings.json.")
    p.add_argument("--phrasing_model", type=str,
                   default="gcp/google/gemini-3.1-pro-preview",
                   help="Model used when --question_phrasing=llm.")
    p.add_argument("--find_polarity", type=str, default="mixed",
                   choices=["mixed", "yes", "no"],
                   help="V2 'find' answer polarity. 'mixed' (default): even "
                        "episodes are yes (buried+gated), odd are no. 'yes': "
                        "force every find episode to yes (buried target + "
                        "occlusion gate) — use to top up the scarcer yes half. "
                        "'no': force every find episode to no.")
    p.add_argument("--exclude_qtype", type=str, default="",
                   help="Comma-separated question types to skip.")
    p.add_argument("--max_actions", type=int, default=MAX_ACTIONS_PER_SCENE,
                   help="Maximum manipulation actions per scene.")
    p.add_argument("--rearrange_visible_thresh", type=float, default=0.10,
                   help="V2 rearrangement: an object counts as 'removed from "
                        "view' when the fraction of its spawn-time unoccluded "
                        "footprint still visible in the top-down FoV drops to "
                        "or below this value (default 0.10 = 10%%).")
    p.add_argument("--sft_max_steps", type=int, default=0,
                   help="Max VLM steps before early termination (0=unlimited).")
    p.add_argument("--sp_identify_stage", type=int, default=0, choices=[0, 1],
                   help="Optional legacy marked control: make a first no-tool VLM "
                        "call that reports nullable badge locations, then answer "
                        "from the same image. Not an answer-validity gate.")
    p.add_argument("--marked_condition", type=str, default="agentic",
                   choices=["direct", "perception", "perception_suggest",
                            "perception_report", "agentic", "agentic_instructed",
                            "agentic_ledger"],
                   help="Marked-comparison ablation arm. direct=one image/no "
                        "tools; perception*=interactive perception but no "
                        "manipulation; agentic=full tools; agentic_instructed "
                        "and agentic_ledger are separate prompt interventions. "
                        "Ignored by other types.")
    p.add_argument("--scene_bank", type=str, default=None,
                   help="Frozen marked scene-bank manifest. Each episode replays "
                        "a saved settled scene and texture-baked badges instead of "
                        "generating/retrying a new scene. Only valid with "
                        "--question_type marked_comparison.")
    p.add_argument("--verify_answer", action="store_true",
                   help="Run answer verification with a second VLM call.")

    # Renderer
    p.add_argument("--renderer", type=str, default="blender",
                   choices=["pybullet", "blender"],
                   help="Rendering backend. Default: blender.")
    p.add_argument("--random_table_texture", type=int, default=1,
                   choices=[0, 1],
                   help="Apply random PBR wood texture per episode.")
    p.add_argument("--lighting_mode", type=str, default="easy",
                   choices=["easy", "hard"],
                   help="Lighting mode. 'easy': ambient only. 'hard': + sun lamp.")

    # Spawn containment
    p.add_argument("--use_planes", type=int, default=1, choices=[0, 1],
                   help="Invisible containment planes during spawn.")
    p.add_argument("--use_basket", type=int, default=0, choices=[0, 1],
                   help="Spawn inside basket (legacy).")
    p.add_argument("--basket_scale", type=float, default=1.0)
    p.add_argument("--basket_wall_height", type=float, default=0.15)

    # Grasp model
    p.add_argument("--graspgen_config", type=str,
                   default="checkpoints/graspgen/graspgen_robotiq_2f_140.yml")
    p.add_argument("--finger_offset_deg", type=float, default=0.0,
                   help="Deprecated — ignored.")
    p.add_argument("--num_pc_points", type=int, default=4000)

    # Episode selection
    p.add_argument("--only_ep", type=int, default=None,
                   help="Run only this episode index.")
    p.add_argument("--episodes", type=str, default=None,
                   help="Episode indices: comma-separated or ranges (e.g. '180-379').")
    p.add_argument("--few_shot", action="store_true",
                   help="Inject few-shot exemplars. Agentic mode only.")
    p.add_argument("--few_shot_rollout", type=str,
                   default="rollouts/rollout_vlm_20260516_063429")

    # Worker internals (set by orchestrator, not user)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--blender_port", type=int, default=9876)
    p.add_argument("--worker_id", type=int, default=None)
    p.add_argument("--two_pass_spawn", type=int, default=0)

    args = p.parse_args()

    # ── Auto-infer n_episodes from episode_subset JSON ──
    if args.n_episodes is None:
        if args.episode_subset and os.path.isfile(args.episode_subset):
            args.n_episodes = _infer_n_episodes_from_subset(args.episode_subset)
        else:
            args.n_episodes = 380  # default benchmark size

    # ── Treat empty string --episode_subset as None ──
    if args.episode_subset == "":
        args.episode_subset = None

    return args
