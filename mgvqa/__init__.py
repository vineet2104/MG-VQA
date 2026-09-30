"""
mgvqa — MG-VQA-Bench evaluation package.

This package owns the LOGIC layer of the benchmark:
  - the multi-turn agentic reasoning loop  (mgvqa.reasoning_loop)
  - the single-pass baseline                (mgvqa.singlepass)
  - the episode/worker driver               (mgvqa.worker)
  - CLI + scheduling + scoring + snapshots  (mgvqa.cli / episode / verification / scene_snapshot)
  - the parallel orchestrator               (mgvqa.orchestrator)
  - prompt templates                        (mgvqa/prompt_templates/)

It IMPORTS the shared simulator/rendering/model code from the repo root
unchanged (tools.py, questions.py, models/, owg_robot/, robo_blender/,
scene_gen/, experiments/marked_size/, gt_visibility_filter.py). Those are
NOT copied here — only the tool-call definitions and planner setup live in
this package.

Entry point: ``python -m mgvqa.run_benchmark`` (or ``mgvqa/run_benchmark.py``).
"""
