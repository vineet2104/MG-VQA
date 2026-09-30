#!/usr/bin/env python3
"""
orchestrator.parallel — Parallel episode orchestrator for MG-VQA-Bench.

Launches N workers, each running a subset of episodes via run.py (or
demo_conversational.py for backward compatibility).
All workers share the same rollout directory, each writing to its own
subdirectory. Once all workers complete, results are merged.

Usage:
    BLENDER=$HOME/blender/blender DISPLAY=:100 CUDA_VISIBLE_DEVICES=0 \
    python demo_parallel.py \
        --n_workers 6 \
        --model_name gcp/google/gemini-3-flash-preview \
        --n_episodes 30 \
        --seed 42 \
        --mode agentic \
        --pointing_model vlm

All standard demo.py flags are supported and passed through to workers.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Wall-time estimation (per-qtype calibration)
# ---------------------------------------------------------------------------
#
# Per-qtype min/ep numbers measured from Gemini-3.1-Pro at 6 workers on a
# single L40S (rollout_vlm_20260516_063429 / rollout_singlepass_20260516_051217,
# 180 episodes each, seed 1000000). See CLAUDE.md "Per-qtype agentic timings".
#
# These are SINGLE-WORKER-EQUIVALENT minutes per episode (i.e. wall time
# divided by n_workers from the source run).
#
# To get parallel wall-time for a planned run:
#     wall = sum(n_per_type[q] * MIN_PER_EP[mode][q]) / (n_workers * scaling)
# where `scaling` ≈ 0.85 (singlepass) / 0.95 (agentic) accounts for
# Blender startup + API tail latencies. We pick conservative values so the
# estimate errs on the SLOW side rather than promising too much.

MIN_PER_EP = {
    "agentic": {
        "counting":             7.63,
        "interactive_counting": 9.59,
        "existence":            5.52,
        "same_attribute":      24.48,
        "relative_size":       20.26,
        "under_object":        13.81,
    },
    "singlepass": {
        "counting":             1.60,
        "interactive_counting": 1.87,
        "existence":            1.88,
        "same_attribute":       2.54,
        "relative_size":        2.39,
        "under_object":         2.71,
    },
}

# Scaling efficiency observed when going from 6 → 12 workers
# (singlepass canary: 1.19h*6 / (60ep)/12wk ≈ 1.9× scaling at 12 workers).
# These are the FRACTION of ideal speed (1.0 = perfect scaling).
SCALING_EFF = {
    "singlepass": 0.85,
    "agentic":    0.95,
}


def estimate_schedule(n_episodes: int, question_type: str,
                      excluded: set, benchmark_version: str = "v1") -> "dict[str, int]":
    """Reproduce the worker's episode_schedule construction logic to count
    how many episodes per question type will actually run after applying
    --exclude_qtype. Returns {qtype: n_eps_to_run}.

    We do NOT seed the shuffle here because we only need counts, not
    ordering.
    """
    if benchmark_version == "v2":
        # Must match questions.QUESTION_TYPES_V2 (rearrangement was dropped from
        # the default schedule — still runnable via --question_type
        # rearrangement, but not in the "random" pool; `compare` was added as
        # the 4th V2 task).
        QTYPES = ["count", "find", "beneath", "compare"]
    else:
        QTYPES = [
            "counting", "interactive_counting", "existence",
            "same_attribute", "relative_size", "under_object",
        ]
    if question_type == "random":
        n_per_type = -(-n_episodes // len(QTYPES))  # ceil division
        full_schedule = []
        for qt in QTYPES:
            full_schedule.extend([qt] * n_per_type)
        full_schedule = full_schedule[:n_episodes]
    else:
        full_schedule = [question_type] * n_episodes

    # Apply exclusion (slots are skipped, not removed — but for time
    # estimation we count only the running ones; skipped slots cost ~0s).
    counts = {}
    for qt in full_schedule:
        if qt in excluded:
            continue
        counts[qt] = counts.get(qt, 0) + 1
    return counts


def print_time_estimate(args, n_workers: int) -> None:
    """Print an upfront wall-time estimate for the planned run.

    Uses per-qtype min/ep calibrated from Gemini-3.1-Pro 6-worker baseline
    runs (see MIN_PER_EP) and a scaling-efficiency factor to extrapolate
    to the requested n_workers.
    """
    excluded = set()
    if getattr(args, "exclude_qtype", ""):
        excluded = {q.strip() for q in args.exclude_qtype.split(",") if q.strip()}

    counts = estimate_schedule(args.n_episodes, args.question_type, excluded,
                               getattr(args, "benchmark_version", "v1"))
    if not counts:
        print(f"\n  ⚠ Time estimate: no runnable episodes after exclusion.")
        return

    # ── Snapshot-only mode: no VLM, no perception/grasp models. ──
    # Each episode is just spawn → settle → render → save.  Measured
    # ~0.5 min/ep single-worker (spawn ~14s + settle ~3s + 2 Blender
    # renders ~10s).  Use a flat rate across all tasks and skip the
    # per-qtype agentic calibration entirely.
    if bool(getattr(args, "snapshots_only", 0)):
        eff = 0.90
        SNAP_MIN_PER_EP = 0.6
        n_eff = sum(counts.values())
        seq_min = n_eff * SNAP_MIN_PER_EP
        par_min = seq_min / max(1, n_workers) / eff
        n_skipped = args.n_episodes - n_eff
        print(f"\n  ┌─ Wall-time estimate (snapshots-only, {n_workers} workers) ───")
        print(f"  │ Effective episodes: {n_eff} "
              f"(skipping {n_skipped} excluded slots)")
        print(f"  │ Rate: {SNAP_MIN_PER_EP:.2f} min/ep single-worker "
              f"(spawn + settle + 2 renders, NO VLM)")
        for qt in ("count", "find", "beneath", "compare", "rearrangement"):
            if counts.get(qt, 0):
                print(f"  │   {qt:<22s} {counts[qt]:>4d} ep")
        print(f"  │ Sequential : {seq_min:>7.0f} min "
              f"({seq_min/60:>5.1f} h) on 1 worker")
        print(f"  │ Parallel   : {par_min:>7.0f} min "
              f"({par_min/60:>5.1f} h) on {n_workers} workers "
              f"(eff={eff:.0%})")
        eta = datetime.now().timestamp() + par_min * 60
        print(f"  │ ETA        : "
              f"{datetime.fromtimestamp(eta).strftime('%Y-%m-%d %H:%M')}")
        print(f"  └─ (snapshot generation — recycles every "
              f"{getattr(args, 'max_eps_per_subprocess', 8)} eps)")
        return

    # Normalize mode for the timing table: 'direct'/'singlepass' share the
    # singlepass calibration; 'perception' has no calibration table yet, so it
    # falls back to the agentic rates scaled down (fewer turns, no manipulation).
    _mode_key = {"direct": "singlepass", "singlepass": "singlepass",
                 "perception": "agentic",  # closest available; over-estimates
                 "agentic": "agentic"}.get(args.mode, args.mode)
    rates = MIN_PER_EP.get(_mode_key)
    if rates is None:
        return

    eff = SCALING_EFF.get(_mode_key, 0.9)
    # V2 tasks (and the marked/compare type) aren't in the calibration
    # table; fall back to a rough per-task estimate so the summary never
    # KeyErrors.  ``compare`` / ``marked_comparison`` are perception+
    # manipulation heavy, so estimate like a rearrangement-class task.
    _V2_FALLBACK = {"count": 8.0, "find": 6.0, "beneath": 14.0,
                    "rearrangement": 16.0, "compare": 16.0,
                    "marked_comparison": 16.0}
    def _rate(q):
        return rates.get(q) if q in rates else _V2_FALLBACK.get(q, 10.0)
    seq_min = sum(counts[q] * _rate(q) for q in counts)
    par_min = seq_min / max(1, n_workers) / eff

    n_skipped = args.n_episodes - sum(counts.values())

    print(f"\n  ┌─ Wall-time estimate ({args.mode}, {n_workers} workers) ───")
    print(f"  │ Effective episodes: {sum(counts.values())} "
          f"(skipping {n_skipped} excluded slots)")
    for qt in ("counting", "interactive_counting", "existence",
               "same_attribute", "relative_size", "under_object",
               "count", "find", "beneath", "rearrangement"):
        if counts.get(qt, 0):
            mins = counts[qt] * _rate(qt)
            print(f"  │   {qt:<22s} {counts[qt]:>4d} ep × {_rate(qt):>5.2f} min "
                  f"= {mins:>7.1f} min single-worker")
    if excluded:
        excluded_skipped = sorted(args.n_episodes // 6 if q in
                                  ("counting","interactive_counting","existence",
                                   "same_attribute","relative_size","under_object")
                                  else 0 for q in excluded)
        print(f"  │   excluded (fast-skip): {sorted(excluded)} "
              f"(~{n_skipped} slots, <1 min total)")
    hrs = par_min / 60
    print(f"  │ Sequential : {seq_min:>7.0f} min "
          f"({seq_min/60:>5.1f} h) on 1 worker")
    print(f"  │ Parallel   : {par_min:>7.0f} min "
          f"({hrs:>5.1f} h) on {n_workers} workers "
          f"(eff={eff:.0%})")
    eta = datetime.now().timestamp() + par_min * 60
    eta_str = datetime.fromtimestamp(eta).strftime("%Y-%m-%d %H:%M")
    print(f"  │ ETA        : {eta_str}")
    print(f"  └─ (calibration: Gemini-3.1-Pro 6-worker May-16 baseline)")


# ---------------------------------------------------------------------------
# Episode assignment
# ---------------------------------------------------------------------------

def assign_episodes(n_episodes: int, n_workers: int,
                     target_episodes: list = None) -> list:
    """Round-robin episode assignment.

    Returns a list of lists: assignments[worker_id] = [ep_indices...].
    When ``target_episodes`` is provided, only those indices are
    distributed (others are silently dropped from the schedule).
    """
    assignments = [[] for _ in range(n_workers)]
    if target_episodes:
        eps = sorted(int(e) for e in target_episodes)
    else:
        eps = list(range(n_episodes))
    for i, ep in enumerate(eps):
        assignments[i % n_workers].append(ep)
    return assignments


# ---------------------------------------------------------------------------
# Worker launch
# ---------------------------------------------------------------------------

def build_worker_cmd(args, worker_id: int, episodes: list,
                     worker_dir: str, pause_file: str) -> list:
    """Build the command to launch a worker subprocess (run.py or demo_conversational.py).

    Uses the same Python interpreter that's running the orchestrator.
    When launched from within a conda env, sys.executable points to the
    conda Python (e.g. /home/vineet/.conda/envs/agentic_mqa/bin/python).
    """
    # Workers must use the caller's environment, irrespective of its name.
    python_exe = sys.executable
    cmd = [
        python_exe, os.environ.get(
            "MGVQA_WORKER_SCRIPT", str(Path(__file__).resolve().parents[1] / "worker_entry.py")),
        "--model_name", args.model_name,
        "--n_episodes", str(args.n_episodes),
        "--seed", str(args.seed),
        "--mode", args.mode,
        "--n_objects", str(args.n_objects),
        "--scene_model", args.scene_model,
        "--pointing_model", args.pointing_model,
        "--pointing_vlm_model", getattr(
            args, "pointing_vlm_model", "gcp/google/gemini-3.1-pro-preview"),
        "--renderer", args.renderer,
        "--lighting_mode", args.lighting_mode,
        "--vis", str(args.vis),
        "--save_spawn_screenshots", str(args.save_spawn_screenshots),
        "--max_actions", str(args.max_actions),
        "--rearrange_visible_thresh",
        str(getattr(args, "rearrange_visible_thresh", 0.10)),
        "--question_type", args.question_type,
        "--benchmark_version", getattr(args, "benchmark_version", "v1"),
        "--question_phrasing", getattr(args, "question_phrasing", "template"),
        "--phrasing_model", getattr(args, "phrasing_model",
                                    "gcp/google/gemini-3.1-pro-preview"),
        "--find_polarity", getattr(args, "find_polarity", "mixed"),
        "--save_blend_anim", str(args.save_blend_anim),
        "--save_snapshot", str(getattr(args, "save_snapshot", 1)),
        "--snapshots_only", str(getattr(args, "snapshots_only", 0)),
        "--random_table_texture", str(args.random_table_texture),
        "--use_planes", str(args.use_planes),
        "--use_basket", str(args.use_basket),
        "--max_graspable_dim_cm", str(args.max_graspable_dim_cm),
        "--num_pc_points", str(args.num_pc_points),
        "--two_pass_spawn", str(args.two_pass_spawn),
        "--sp_identify_stage", str(getattr(args, "sp_identify_stage", 0) or 0),
        "--marked_condition", str(getattr(args, "marked_condition", "agentic")),
        # Parallel-specific flags
        "--output_dir", worker_dir,
        "--worker_id", str(worker_id),
        "--blender_port", str(9876 + worker_id),
        "--episodes", ",".join(str(e) for e in episodes),
        "--max_eps_per_subprocess", str(
            getattr(args, "max_eps_per_subprocess", 0) or 0),
        "--sft_max_steps", str(
            getattr(args, "sft_max_steps", 0) or 0),
    ]
    # Pass optional flags
    if args.asset_path:
        cmd.extend(["--asset_path", args.asset_path])
    if args.graspgen_config:
        cmd.extend(["--graspgen_config", args.graspgen_config])
    if getattr(args, "annotations_full", None):   # non-empty string
        cmd.extend(["--annotations_full", args.annotations_full])
    if getattr(args, "min_dim_cm", None) is not None:
        cmd.extend(["--min_dim_cm", str(args.min_dim_cm)])
    if getattr(args, "exclude_qtype", ""):
        cmd.extend(["--exclude_qtype", args.exclude_qtype])
    if getattr(args, "from_snapshot", ""):
        cmd.extend(["--from_snapshot", args.from_snapshot])
    if getattr(args, "few_shot", False):
        cmd.append("--few_shot")
        if getattr(args, "few_shot_rollout", ""):
            cmd.extend(["--few_shot_rollout", args.few_shot_rollout])
    # Forward explicit subset decisions, including disabling the filter.
    # The frozen bank is the default evaluation set.
    _subset = getattr(args, "episode_subset", None)
    cmd.extend(["--episode_subset", _subset if _subset else ""])
    return cmd


def launch_workers(args, assignments: list, save_dir: str,
                   pause_file: str) -> list:
    """Launch worker subprocesses and return (proc, worker_dir) pairs.

    GPU assignment:
      - n_gpus == 1 (default): every worker inherits CUDA_VISIBLE_DEVICES
        from the orchestrator's environment (legacy behaviour — single GPU
        controlled by the calling shell).
      - n_gpus > 1: workers are pinned round-robin to GPU IDs 0..(n_gpus-1)
        via per-process CUDA_VISIBLE_DEVICES override. This requires that
        the orchestrator is run with all desired GPUs visible (e.g. unset
        CUDA_VISIBLE_DEVICES or set CUDA_VISIBLE_DEVICES=0,1).
    """
    workers = []
    base_env = os.environ.copy()
    base_env["AGENTIC_MQA_PAUSE_FILE"] = pause_file
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"

    n_gpus = max(1, int(getattr(args, "n_gpus", 1)))

    for wid, episodes in enumerate(assignments):
        if not episodes:
            continue
        worker_dir = os.path.join(save_dir, f"worker_{wid}")
        os.makedirs(worker_dir, exist_ok=True)

        cmd = build_worker_cmd(args, wid, episodes, worker_dir, pause_file)
        log_path = os.path.join(worker_dir, "worker.log")

        # Per-worker env (copy so each worker has its own CUDA assignment)
        worker_env = dict(base_env)
        if n_gpus > 1:
            gpu_id = wid % n_gpus
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_label = f" → GPU {gpu_id}"
        else:
            gpu_label = ""

        print(f"  Worker {wid}: episodes {episodes}{gpu_label} → {worker_dir}")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=worker_env,
            cwd=os.getcwd(),
        )
        workers.append({
            "proc": proc,
            "worker_id": wid,
            "episodes": episodes,
            "worker_dir": worker_dir,
            "log_file": log_file,
            "log_path": log_path,
        })

    return workers


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def _get_completed_episodes(worker_dir: str) -> set:
    """Read results.jsonl to find which episodes were completed."""
    results_path = os.path.join(worker_dir, "results.jsonl")
    completed = set()
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "episode" in rec:
                        completed.add(int(rec["episode"]))
                except (json.JSONDecodeError, ValueError):
                    pass
    return completed


def monitor_workers(workers: list, args=None) -> dict:
    """Wait for all workers to complete, recycling on clean exit.

    When a worker exits with code 0 but has remaining episodes (i.e. it
    hit --max_eps_per_subprocess), it is re-launched with only the
    unfinished episodes. This works around PyBullet's mesh memory leak
    by giving each subprocess a bounded lifetime.

    Returns {worker_id: final_returncode}.
    """
    results = {}
    active = list(workers)
    max_eps = (getattr(args, "max_eps_per_subprocess", 0) or 0) if args else 0

    # Build base env for re-launches (same as initial launch)
    base_env = os.environ.copy()
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"
    n_gpus = max(1, int(getattr(args, "n_gpus", 1))) if args else 1

    while active:
        still_running = []
        for w in active:
            retcode = w["proc"].poll()
            if retcode is not None:
                w["log_file"].close()

                # Check if this is a recyclable exit (code 0 + remaining eps)
                if retcode == 0 and max_eps > 0 and args is not None:
                    completed = _get_completed_episodes(w["worker_dir"])
                    remaining = [e for e in w["episodes"] if e not in completed]

                    if remaining:
                        # Re-launch with remaining episodes
                        print(f"  Worker {w['worker_id']}: recycling "
                              f"({len(completed)} done, "
                              f"{len(remaining)} remaining)")
                        cmd = build_worker_cmd(
                            args, w["worker_id"], remaining,
                            w["worker_dir"],
                            base_env.get("AGENTIC_MQA_PAUSE_FILE", ""))
                        log_path = w["log_path"]
                        log_file = open(log_path, "a")  # append to same log

                        worker_env = dict(base_env)
                        if n_gpus > 1:
                            gpu_id = w["worker_id"] % n_gpus
                            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                        proc = subprocess.Popen(
                            cmd, stdout=log_file,
                            stderr=subprocess.STDOUT,
                            env=worker_env, cwd=os.getcwd())
                        w["proc"] = proc
                        w["log_file"] = log_file
                        w["episodes"] = remaining
                        still_running.append(w)
                        continue

                # Terminal exit
                results[w["worker_id"]] = retcode
                status = "✓" if retcode == 0 else f"✗ (exit {retcode})"
                print(f"  Worker {w['worker_id']} finished: {status}  "
                      f"(episodes {w['episodes']})")
            else:
                still_running.append(w)
        active = still_running
        if active:
            time.sleep(5)

    return results


# ---------------------------------------------------------------------------
# Merge results
# ---------------------------------------------------------------------------

def merge_results(save_dir: str, n_workers: int) -> None:
    """Merge per-worker results.jsonl into the top-level file."""
    merged_path = os.path.join(save_dir, "results.jsonl")
    all_records = []

    for wid in range(n_workers):
        worker_results = os.path.join(save_dir, f"worker_{wid}", "results.jsonl")
        if os.path.exists(worker_results):
            with open(worker_results) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))

    # Sort by episode number
    all_records.sort(key=lambda r: r.get("episode", 0))

    with open(merged_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"\n  Merged {len(all_records)} episode results → {merged_path}")
    return all_records


from mgvqa.orchestrator.summary import generate_summary  # noqa: E402

def symlink_episode_dirs(save_dir: str, n_workers: int) -> None:
    """Create top-level symlinks to episode dirs inside worker subdirs.

    This makes the rollout dir look like a normal sequential run:
        rollout_dir/ep_000_counting → worker_0/ep_000_counting
    """
    for wid in range(n_workers):
        worker_dir = os.path.join(save_dir, f"worker_{wid}")
        if not os.path.isdir(worker_dir):
            continue
        for entry in os.listdir(worker_dir):
            if entry.startswith("ep_"):
                src = os.path.join(worker_dir, entry)
                dst = os.path.join(save_dir, entry)
                if not os.path.exists(dst):
                    os.symlink(os.path.relpath(src, save_dir), dst)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel episode orchestrator for Agentic MQA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 6 workers, 30 episodes, Gemini 3 Flash
  python demo_parallel.py --n_workers 6 --n_episodes 30 \\
      --model_name gcp/google/gemini-3-flash-preview --seed 42

  # 4 workers, 12 episodes, GPT-5.2
  python demo_parallel.py --n_workers 4 --n_episodes 12 \\
      --model_name azure/openai/gpt-5.2 --seed 100
""")

    # ── Primary args (what users typically set) ──
    p.add_argument("--mode", type=str, default="agentic",
                   choices=["direct", "perception", "agentic", "singlepass"],
                   help="Tool-access regime: direct (no tools, == singlepass) "
                        "| perception (perceive only, no manipulation) "
                        "| agentic (full loop). 'singlepass' aliases 'direct'.")
    p.add_argument("--model_name", type=str,
                   default="gcp/google/gemini-3-flash-preview",
                   help="VLM model for planning.")
    p.add_argument("--n_workers", type=int, default=4,
                   help="Number of parallel worker processes.")
    p.add_argument("--save_blend_anim", type=int, default=0,
                   help="Save .blend animation per episode.")
    p.add_argument("--save_snapshot", type=int, default=1,
                   help="Save a frozen scene snapshot per episode "
                        "(settled poses + table texture). Default: 1.")
    p.add_argument("--from_snapshot", type=str, default="",
                   help="Directory of pre-saved snapshots to restore from "
                        "instead of spawning. Default: '' (spawn fresh).")
    p.add_argument("--snapshots_only", type=int, default=0,
                   help="Snapshot-generation mode: spawn + settle + save "
                        "artifacts (question/occlusion/gt_scene/seg_overlay/"
                        "snapshot/review.json) then STOP. No VLM. Default: 0.")

    # ── Benchmark defaults (rarely need to override) ──
    p.add_argument("--n_episodes", type=int, default=None,
                   help="Total episode slots. Auto-inferred from "
                        "--episode_subset if not set.")
    p.add_argument("--seed", type=int, default=1000000,
                   help="Master seed. Default: 1000000 (frozen benchmark).")
    p.add_argument("--episode_subset", type=str,
                   default="",
                   help="Benchmark episode list JSON. Set to '' to disable.")
    p.add_argument("--annotations_full", type=str,
                   default="owg_robot/assets/object_annotations_eval.json",
                   help="Object annotations. Default: EVAL pool.")
    from models.catalog import DEFAULT_SCENE_MODEL
    p.add_argument("--scene_model", default=DEFAULT_SCENE_MODEL,
                   help="Fixed scene graph backend, independent of planner.")
    p.add_argument("--pointing_model", type=str, default="vlm",
                   choices=["vlm", "roborefer", "molmo"],
                   help="Pointing model. Default: vlm.")
    p.add_argument("--pointing_vlm_model", type=str,
                   default="gcp/google/gemini-3.1-pro-preview",
                   help="VLM for pointing calls (when --pointing_model=vlm).")
    p.add_argument("--max_eps_per_subprocess", type=int, default=8,
                   help="Worker recycle interval (0=unlimited). Default: 8.")

    # ── Advanced / rarely changed ──
    p.add_argument("--n_gpus", type=int, default=1,
                   help="GPUs to spread workers across.")
    p.add_argument("--exclude_qtype", type=str, default="",
                   help="Comma-separated question types to skip.")
    p.add_argument("--n_objects", type=int, default=15)
    p.add_argument("--renderer", type=str, default="blender",
                   choices=["blender", "pybullet"])
    p.add_argument("--lighting_mode", type=str, default="easy",
                   choices=["easy", "hard"])
    p.add_argument("--vis", type=int, default=1)
    p.add_argument("--save_spawn_screenshots", type=int, default=0)
    p.add_argument("--max_actions", type=int, default=10)
    p.add_argument("--rearrange_visible_thresh", type=float, default=0.10)
    p.add_argument("--sft_max_steps", type=int, default=0,
                   help="Max VLM steps before early exit (0=unlimited).")
    p.add_argument("--question_type", type=str, default="random")
    p.add_argument("--benchmark_version", type=str, default="v1",
                   choices=["v1", "v2"])
    p.add_argument("--question_phrasing", type=str, default="template",
                   choices=["template", "llm"])
    p.add_argument("--phrasing_model", type=str,
                   default="gcp/google/gemini-3.1-pro-preview")
    p.add_argument("--find_polarity", type=str, default="mixed",
                   choices=["mixed", "yes", "no"])
    p.add_argument("--random_table_texture", type=int, default=1)
    p.add_argument("--use_planes", type=int, default=1)
    p.add_argument("--use_basket", type=int, default=0)
    p.add_argument("--max_graspable_dim_cm", type=float, default=8.0)
    p.add_argument("--num_pc_points", type=int, default=4000)
    p.add_argument("--two_pass_spawn", type=int, default=0)
    p.add_argument("--sp_identify_stage", type=int, default=0)
    p.add_argument("--marked_condition", type=str, default="agentic",
                   choices=["direct", "perception", "perception_suggest",
                            "perception_report", "agentic", "agentic_instructed",
                            "agentic_ledger"])
    p.add_argument("--asset_path", type=str,
                   default="./owg_robot/assets/unified_objects")
    p.add_argument("--min_dim_cm", type=float, default=2.0)
    p.add_argument("--graspgen_config", type=str,
                   default="checkpoints/graspgen/graspgen_robotiq_2f_140.yml")
    p.add_argument("--few_shot", action="store_true",
                   help="Inject few-shot exemplars. Agentic only.")
    p.add_argument("--few_shot_rollout", type=str,
                   default="rollouts/rollout_vlm_20260516_063429")
    p.add_argument("--target_episodes", "--episodes", type=str, default=None,
                   help="Episode indices to run. Supports ranges (180-379).")

    args = p.parse_args()

    # ── Auto-infer n_episodes from episode_subset JSON ──
    if args.n_episodes is None:
        if args.episode_subset and os.path.isfile(args.episode_subset):
            try:
                import json as _json
                with open(args.episode_subset) as _f:
                    _doc = _json.load(_f)
                args.n_episodes = _doc.get("n_episodes_total", 380)
            except (OSError, ValueError):
                args.n_episodes = 380
        else:
            args.n_episodes = 380

    # Treat empty string as None
    if args.episode_subset == "":
        args.episode_subset = None

    return args


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def print_model_banner(args):
    """Describe enabled components, not inactive legacy configuration."""
    mode = "direct" if args.mode == "singlepass" else args.mode
    print("=" * 70)
    print(f"  Mode            : {mode}")
    print(f"  Planner VLM     : {args.model_name}")
    if mode == "direct":
        print("  Perception      : disabled (one image, no tools)")
        print("  Manipulation    : disabled")
    else:
        print(f"  Scene graph VLM : {args.scene_model}")
        print("  Grounding       : planner-selected point → SAM3 (no pointing VLM)")
        print("  Manipulation    : " + (
            "GraspGen + push" if mode == "agentic" else "disabled"))
    print("=" * 70)


def main():
    args = parse_args()
    print_model_banner(args)

    # Generate seed if not provided
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**31 - 1)
        print(f"  Generated random seed: {args.seed}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from mgvqa.output import create_run_directory
    save_dir = create_run_directory(args.model_name, args.mode, timestamp=timestamp)
    print(f"MGVQA_RUN_DIR={os.path.abspath(save_dir)}", flush=True)

    # Pause file for rate-limit coordination
    pause_file = f"/tmp/agentic_mqa_pause_{timestamp}"

    # Assign episodes to workers
    target_eps = None
    if getattr(args, "target_episodes", None):
        # Parse ranges (e.g. "180-379") and comma-separated (e.g. "0,3,6")
        target_eps = []
        for part in args.target_episodes.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part and not part.startswith("-"):
                lo, hi = part.split("-", 1)
                target_eps.extend(range(int(lo), int(hi) + 1))
            else:
                target_eps.append(int(part))
        target_eps = sorted(set(target_eps))
        # Cap n_workers to len(target_eps).
        n_workers = min(args.n_workers, len(target_eps))
    else:
        n_workers = min(args.n_workers, args.n_episodes)
    assignments = assign_episodes(args.n_episodes, n_workers,
                                   target_episodes=target_eps)

    print(f"\n{'='*60}")
    print(f"  Parallel MG-VQA-Bench evaluation")
    print(f"  Mode: {args.mode} | Model: {args.model_name}")
    n_gpus_eff = max(1, int(getattr(args, "n_gpus", 1)))
    print(f"  Episodes: {args.n_episodes} | Workers: {n_workers} | "
          f"GPUs: {n_gpus_eff} | Objects: {args.n_objects}")
    print(f"  Seed: {args.seed} | Renderer: {args.renderer}")
    if getattr(args, "exclude_qtype", ""):
        print(f"  Excluding qtypes: {args.exclude_qtype}")
    if getattr(args, "annotations_full", None):
        ann_path = args.annotations_full
        ann_base = os.path.basename(ann_path).lower()
        if "_eval" in ann_base:
            pool_label = "EVAL pool (held-out 20% — for benchmark evaluation)"
        elif "_train" in ann_base:
            pool_label = "TRAIN pool (80% — for SFT data generation)"
        elif ann_base in ("object_annotations.json",):
            pool_label = "FULL pool (NOT split — use only for ad-hoc runs)"
        else:
            pool_label = "custom pool"
        print(f"  Object pool: {pool_label}")
        print(f"    → {ann_path}  (min_dim={args.min_dim_cm}cm)")
        # Loud warning if accidentally using the FULL pool for eval/SFT.
        if ann_base == "object_annotations.json":
            print(f"    ⚠ WARNING: this file mixes train and eval objects. "
                  f"For benchmark runs use object_annotations_eval.json; "
                  f"for SFT data use object_annotations_train.json.")
    else:
        print(f"  Object pool: LEGACY annotations.json "
              f"(max_graspable_dim={args.max_graspable_dim_cm}cm, "
              f"no target/distractor distinction)")
    print(f"  Output: {save_dir}")
    print(f"  Pause file: {pause_file}")
    print(f"{'='*60}")
    print(f"\n  Episode assignments:")

    # Write run_config.json
    run_config = {
        "n_episodes": args.n_episodes,
        "n_workers": n_workers,
        "n_gpus": int(getattr(args, "n_gpus", 1)),
        "n_objects": args.n_objects,
        "model_name": args.model_name,
        "scene_model": args.scene_model,
        "mode": args.mode,
        "seed": args.seed,
        "base_seed": args.seed,
        "pointing_model": args.pointing_model,
        # Always record the pointer-VLM string so we never have to
        # guess which model produced the points. When pointing_model
        # is roborefer/molmo this field is informational only.
        "pointing_vlm_model": getattr(
            args, "pointing_vlm_model", "gcp/google/gemini-3.1-pro-preview"),
        "renderer": args.renderer,
        "lighting_mode": args.lighting_mode,
        "question_type": args.question_type,
        "marked_condition": getattr(args, "marked_condition", "agentic"),
        "exclude_qtype": getattr(args, "exclude_qtype", "") or None,
        # ── V2 / frozen-replay provenance (essential to reproduce a run) ──
        "benchmark_version": getattr(args, "benchmark_version", "v1"),
        "from_snapshot": getattr(args, "from_snapshot", "") or None,
        "scene_bank": getattr(args, "scene_bank", "") or None,
        "snapshots_only": int(getattr(args, "snapshots_only", 0) or 0),
        "question_phrasing": getattr(args, "question_phrasing", "template"),
        "timestamp": timestamp,
        "parallel": True,
        # Object-pool flags — essential for reproducing / understanding a run
        "annotations_full": getattr(args, "annotations_full", None),
        "min_dim_cm": getattr(args, "min_dim_cm", None),
        "asset_path": getattr(args, "asset_path", None),
        "max_graspable_dim_cm": getattr(args, "max_graspable_dim_cm", None),
        "sft_max_steps": int(getattr(args, "sft_max_steps", 0) or 0),
        "max_eps_per_subprocess": int(
            getattr(args, "max_eps_per_subprocess", 0) or 0),
        # Few-shot in-context-learning experiment flags. When enabled,
        # the 6 reserved exemplar episode IDs (one per qtype) are
        # excluded from evaluation; their identities are recorded here
        # so the run can be reproduced without re-deriving them.
        "few_shot": bool(getattr(args, "few_shot", False)),
        "few_shot_rollout": (
            args.few_shot_rollout
            if getattr(args, "few_shot", False) else None),
        "episode_subset": getattr(args, "episode_subset", None),
        "assignments": {str(k): v for k, v in enumerate(assignments)},
    }

    # If --few_shot is on, also record which episode IDs were used as
    # exemplars (and therefore skipped) and how many episodes are
    # actually evaluated, for at-a-glance summary later.
    if run_config["few_shot"]:
        try:
            from few_shot import DEFAULT_EXEMPLARS, exemplar_episode_ids
            run_config["few_shot_exemplars"] = {
                qt: int(ep) for qt, ep in DEFAULT_EXEMPLARS.items()}
            run_config["few_shot_exemplar_episode_ids"] = sorted(
                exemplar_episode_ids())
        except Exception as _e:
            print(f"  ⚠ Could not record few_shot exemplar metadata: {_e}")

    # If --episode_subset is provided, record the size before / after
    # exemplar exclusion so the eval-task count is unambiguous.
    if run_config.get("episode_subset"):
        try:
            with open(args.episode_subset, "r", encoding="utf-8") as f:
                _subset_doc = json.load(f)
            _subset_eps = {int(e) for e in _subset_doc.get("episodes", [])}
            run_config["episode_subset_size"] = len(_subset_eps)
            if run_config["few_shot"]:
                _ex = set(run_config.get(
                    "few_shot_exemplar_episode_ids") or [])
                run_config["n_episodes_evaluated"] = len(_subset_eps - _ex)
            else:
                run_config["n_episodes_evaluated"] = len(_subset_eps)
        except Exception as _e:
            print(f"  ⚠ Could not record episode_subset metadata: {_e}")

    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Up-front wall-time estimate (calibrated against Gemini-3.1-Pro
    # 6-worker baseline; see MIN_PER_EP at top of this file).
    print_time_estimate(args, n_workers)

    # Launch workers
    t0 = time.time()
    workers = launch_workers(args, assignments, save_dir, pause_file)
    print(f"\n  All {len(workers)} workers launched. Monitoring...\n")

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n  [INTERRUPTED] Terminating all workers...")
        for w in workers:
            try:
                w["proc"].terminate()
            except Exception:
                pass
        time.sleep(2)
        for w in workers:
            try:
                w["proc"].kill()
            except Exception:
                pass
            w["log_file"].close()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for all workers (with recycling if --max_eps_per_subprocess is set)
    results = monitor_workers(workers, args=args)
    elapsed = time.time() - t0

    # Check for failures
    failed = {wid: rc for wid, rc in results.items() if rc != 0}
    if failed:
        print(f"\n  ⚠️  {len(failed)} worker(s) exited with errors:")
        for wid, rc in failed.items():
            log_path = os.path.join(save_dir, f"worker_{wid}", "worker.log")
            print(f"    Worker {wid}: exit code {rc} — see {log_path}")

    # Merge results
    print(f"\n  Merging results...")
    records = merge_results(save_dir, n_workers)
    generate_summary(save_dir, records)
    symlink_episode_dirs(save_dir, n_workers)

    # Cleanup pause file
    if os.path.exists(pause_file):
        os.remove(pause_file)

    # Final report
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Wall-clock time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Episodes completed: {len(records)}/{args.n_episodes}")
    if args.n_episodes > 0:
        seq_estimate = elapsed * args.n_episodes / max(len(records), 1)
        print(f"  Estimated sequential time: {seq_estimate/n_workers:.0f}s "
              f"({seq_estimate/n_workers/60:.1f} min)")
        print(f"  Parallel speedup: ~{n_workers:.0f}x")
    print(f"  Output: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
