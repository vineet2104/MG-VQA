#!/usr/bin/env python3
"""
mgvqa/worker_entry.py — subprocess target for the parallel orchestrator.

The orchestrator (mgvqa.orchestrator.parallel) spawns this file as a worker
subprocess (one per --n_workers).  It sets up the import path so that both
the ``mgvqa`` package AND the shared repo-root modules (tools.py, questions.py,
models/, owg_robot/, robo_blender/, …) resolve, then hands off to
``mgvqa.worker.main``.

It also re-exports the key symbols the orchestrator imports by name.
"""

import sys
import os

# This file is mgvqa/worker_entry.py → repo root is two levels up.
_mgvqa_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_mgvqa_dir)

# Repo root FIRST (shared tools.py/questions.py/models/…), then graspgen.
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "third_party", "graspgen"))

# Re-export key symbols for the orchestrator and any code that does
# `from mgvqa.worker_entry import ...`.
from mgvqa.worker import main, parse_args, parse_fail_reason  # noqa: F401,E402
from mgvqa.reasoning_loop import run_vlm_reasoning_loop  # noqa: F401,E402
from mgvqa.singlepass import (  # noqa: F401,E402
    build_singlepass_prompt, run_singlepass, SINGLEPASS_SYSTEM_PROMPT,
)
from mgvqa.gt_helpers import (  # noqa: F401,E402
    _annotate_gt_scene, _capture_sim_gt_state, _classify_target_ids,
    _project_aabb_to_image,
)
from mgvqa.prompts import build_type_specific_constraint as _build_type_specific_constraint  # noqa: F401,E402
from mgvqa.verification import pick_forced_answer as _pick_forced_answer  # noqa: F401,E402


if __name__ == "__main__":
    main()
