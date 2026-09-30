#!/usr/bin/env python3
"""
run_benchmark.py — MG-VQA-Bench V2 **evaluation** entry point.

This is the frozen-benchmark REPLAY + SCORING harness.  It runs one or more
VLMs against a pre-built snapshot bank (e.g. ``approved-questions/benchmark_600``)
and scores each answer against the ground truth stored in every episode's
``scene_snapshot/scene_snapshot.json`` annotation.

It does NOT generate scenes.  Benchmark/question creation lives elsewhere
(``run_v2.py --snapshots_only`` + ``scripts/finalize_bank.py``).  A
``--from_snapshot <bank>`` argument is therefore REQUIRED.

The 4 V2 tasks (all replayed from the bank, never regenerated here):

    count    → free-form INTEGER answer
    find     → existence + 2D point
    beneath  → under_object, 4-option MCQ (NOTA always present)
    compare  → 3-way marked A/B/C size comparison

Usage:

    # Agentic eval on the frozen 600Q benchmark:
    python run_benchmark.py --mode agentic \
        --model_name gcp/google/gemini-3.1-pro-preview \
        --from_snapshot approved-questions/benchmark_600 --n_workers 4

    # Singlepass baseline:
    python run_benchmark.py --mode singlepass \
        --model_name gcp/google/gemini-3.1-pro-preview \
        --from_snapshot approved-questions/benchmark_600 --n_workers 4

    # Debug a slice:
    python run_benchmark.py --mode agentic --model_name <MODEL> \
        --from_snapshot approved-questions/benchmark_600 --episodes 0-3

All other CLI flags are the shared worker flags (see core/cli.py).
"""

import sys
import os

# ── Frozen configuration ───────────────────────────────────────────────
V2_SEED = 1000000  # frozen eval seed (scenes come from the bank, but the seed
                   # still governs any per-run RNG the worker touches)

# This file lives at mgvqa/run_benchmark.py, so the repo root (where the
# SHARED simulator/model modules live — tools.py, questions.py, models/,
# owg_robot/, robo_blender/, scene_gen/, experiments/) is one level up.
_mgvqa_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_mgvqa_dir)

# Auto-load .env (API keys, etc.)
_env_path = os.path.join(_proj_root, ".env")
if os.path.isfile(_env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=False)
    except ImportError:
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    _k, _v = _k.strip(), _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "third_party", "graspgen"))


def _has(argv, flag):
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _get(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _replace(argv, flag, value):
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == flag:
            out.extend([flag, value]); i += 2
        elif a.startswith(flag + "="):
            out.append(f"{flag}={value}"); i += 1
        else:
            out.append(a); i += 1
    return out


def _pin_eval_defaults(argv):
    """Configure a frozen-benchmark REPLAY run.

    Requires ``--from_snapshot <bank>``.  n_episodes is sized to the bank
    (every scene is replayed; per-episode types come from the bank, not from
    a schedule shuffle).  benchmark_version is pinned to v2.  The seed is
    fixed.  No scene generation happens.
    """
    # ── --from_snapshot is mandatory ──
    _from_snapshot = _get(argv, "--from_snapshot")
    if not _from_snapshot:
        sys.exit(
            "run_benchmark.py: --from_snapshot <bank> is REQUIRED.\n"
            "  This is the eval/replay harness — it scores a pre-built bank.\n"
            "  e.g. --from_snapshot approved-questions/benchmark_600\n"
            "  (To CREATE a bank, use run_v2.py --snapshots_only + "
            "scripts/finalize_bank.py.)"
        )

    # Guard against accidentally passing generation-only flags here.
    if _get(argv, "--snapshots_only") not in (None, "0", "false", "False"):
        sys.exit("run_benchmark.py: --snapshots_only is not valid here "
                 "(this harness only REPLAYS a bank; use run_v2.py to "
                 "generate scenes).")

    # ── size n_episodes to the bank ──
    _bank_n = None
    try:
        from mgvqa.scene_snapshot import discover_snapshot_episodes
        _bank_n = len(discover_snapshot_episodes(_from_snapshot))
    except Exception as _e:
        print(f"  ⚠ run_benchmark.py: could not size --from_snapshot bank "
              f"({_e}); relying on --n_episodes.")
    if _bank_n == 0:
        sys.exit(f"run_benchmark.py: no restorable episodes found under "
                 f"{os.path.abspath(_from_snapshot)} — is this a bank dir "
                 f"with ep_*/scene_snapshot/ (or a bank_manifest.json)?")
    if _bank_n:
        val = _get(argv, "--n_episodes")
        if val is not None and val != str(_bank_n):
            print(f"  ⚠ run_benchmark.py: bank has {_bank_n} scenes — "
                  f"overriding --n_episodes {val} → {_bank_n}.")
        print(f"  ▸ run_benchmark.py: replaying all {_bank_n} frozen scenes "
              f"from {os.path.basename(os.path.abspath(_from_snapshot))}.")
        argv = (_replace(argv, "--n_episodes", str(_bank_n))
                if _has(argv, "--n_episodes")
                else argv + ["--n_episodes", str(_bank_n)])

    # ── benchmark_version (pinned to v2) ──
    val = _get(argv, "--benchmark_version")
    if val is not None and val != "v2":
        print(f"  ⚠ run_benchmark.py: overriding --benchmark_version '{val}' "
              f"→ 'v2'.")
    argv = (_replace(argv, "--benchmark_version", "v2")
            if _has(argv, "--benchmark_version")
            else argv + ["--benchmark_version", "v2"])

    # ── seed ──
    if not _has(argv, "--seed"):
        argv += ["--seed", str(V2_SEED)]

    # ── no curated subset filter (the bank IS the curated set) ──
    if not _has(argv, "--episode_subset"):
        argv += ["--episode_subset", ""]

    # ── compare / marked_comparison wiring (needed at REPLAY time) ──
    # The compare task's reasoning loop is controlled by --marked_condition
    # (see mgvqa/reasoning_loop.py).  By default it tracks --mode so the
    # compare arm matches the global tool-access regime:
    #   direct/singlepass → direct,  perception → perception,  agentic → agentic.
    # An explicit --marked_condition (compare sub-study) still overrides.
    # NOTE: --two_pass_spawn is a SCENE-SPAWNING flag only — it has no effect
    # when replaying from a snapshot, so it is intentionally NOT set here.
    if not _has(argv, "--marked_condition"):
        _mode = _get(argv, "--mode")
        _cond = {"singlepass": "direct", "direct": "direct",
                 "perception": "perception",
                 "agentic": "agentic"}.get(_mode, "agentic")
        argv += ["--marked_condition", _cond]

    return argv


def main():
    import argparse
    sys.argv = _pin_eval_defaults(sys.argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--workers", type=int, default=None)
    pre.add_argument("--n_workers", type=int, default=None)
    pre_args, _ = pre.parse_known_args()

    n_workers = pre_args.workers or pre_args.n_workers or 0
    if n_workers > 0:
        _run_parallel()
    else:
        _run_single()


def _run_parallel():
    sys.argv = _remap_workers_arg(sys.argv)
    # Subprocess target for the orchestrator: mgvqa's own worker entry shim
    # (absolute path so it resolves regardless of the worker's cwd).
    os.environ["MGVQA_WORKER_SCRIPT"] = os.path.join(_mgvqa_dir,
                                                     "worker_entry.py")
    from mgvqa.orchestrator.parallel import main as orchestrator_main
    orchestrator_main()


def _run_single():
    from mgvqa.worker import main as worker_main
    worker_main()


def _remap_workers_arg(argv):
    new_argv, i = [], 0
    while i < len(argv):
        if argv[i] == "--workers":
            new_argv.append("--n_workers"); i += 1
            if i < len(argv):
                new_argv.append(argv[i])
        elif argv[i].startswith("--workers="):
            new_argv.append("--n_workers=" + argv[i].split("=", 1)[1])
        else:
            new_argv.append(argv[i])
        i += 1
    return new_argv


if __name__ == "__main__":
    main()
