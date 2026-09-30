# Runtime boundaries

`run_mgvqa.sh` → `python -m mgvqa` → `mgvqa/run_benchmark.py` → optional subprocess
orchestrator → `mgvqa/worker.py` restores a frozen scene, invokes a mode, and scores
the answer. `python -m mgvqa --help` is the public Python CLI; the shell wrapper
adds comma-separated mode loops and launch logs. The package and launcher have
been renamed to `mgvqa`/`run_mgvqa.sh`; historical runtime import paths are retired.

## Environment and rendering

- `owg_robot/env.py::Environment`: headless PyBullet connection, bodies,
  dynamics, arm/gripper movement, contacts and placement checks.
- `owg_robot/robot.py`: UR5/Robotiq joint configuration and kinematics helpers.
- `owg_robot/ui.py::RobotEnvUI`: environment assembly and scene-level helpers;
  the historical name does not mean evaluation needs a GUI.
- `owg_robot/camera.py::Camera`: aligned RGB/depth/segmentation access and camera
  transforms. Depth/segmentation come from PyBullet; planner RGB uses Blender.
- `robo_blender/renderer.py`: PyBullet-to-Blender visual synchronization.
  `server/scene_manager.py` owns Blender meshes/materials/lights;
  `server_launcher.py` owns a background Blender process.
- `mgvqa/scene_snapshot.py`: pose, texture and annotation restoration.

Do not change camera coordinates, settling, friction or object dynamics merely
as a cleanup: those are experimental settings. No mobile navigation or cuRobo
motion planner is part of this release.

## Five tools

`mgvqa/tool_registry.py` defines names, aliases, allowed modes, required planner
fields, mutation flags, terminal flags and backend entry points. The reasoning
loop invokes these backends but retains episode lifecycle bookkeeping:

| Name | Backend | Lifecycle |
|---|---|---|
| `perceive_scene` | `perception/scene_graph.py` | Fixed VLM → YAML map/points |
| `perceive_find_object` (`find_object`) | `perception/grounding.py` | Point → SAM3 mask → marker 1 |
| `grasp_and_remove` | `manipulation/grasp.py` | Mask/depth → GraspGen → contact execution |
| `push` | `manipulation/push.py` | Mask-based descent planning → physical sweep |
| `answer_question` (`answer_questions`) | `mgvqa/answer.py` | Submission only; loop terminates |

The grasp/push implementations and shared geometry live in
`manipulation/primitives.py`. `tools.py` retains utilities and compatibility
exports, not a second copy of these implementations. SAM and overlay utilities
are still shared with the legacy module; further extraction should be tested
rather than replaced with new algorithms.

To add a tool:
1. Implement a backend in a concrete module; accept only needed inputs.
2. Register its ToolSpec and supported modes.
3. Add a lifecycle branch in `mgvqa/reasoning_loop.py`: input validation,
   budget accounting, marker invalidation, structured feedback, and logging.
4. Advertise it in the appropriate mode prompt and add mocked tests.

Registration does **not** automatically implement lifecycle handling. The
registry's required-fields metadata documents the contract; existing loop-side
validation remains authoritative during this staged refactor.

## Prompts and privileged state

Each mode loads separate system and user `.txt` files under
`mgvqa/prompt_templates/`, shared across planners. `build_user_prompt` fills
explicit public fields; the release path no longer injects the old task-specific
strategy constraints or Compare badge-point requests. `format_previous_tool_response`
serializes perception results and physical outcomes without emitting simulator
object names. Search-guidance ablations are not implemented.

`normalize_planner_call` accepts `find_object`, `grasp`, and `push(direction)`:
legacy names map to internal dispatch, absent manipulation IDs select marker 1
(subject to the existing freshness checks), and planner pushes are fixed at
10 cm. The low-level push API remains configurable. Full conversation history
is retained equally across models; the former Nemotron-family truncation is
removed. See `mgvqa/prompt_templates/README.md` for unresolved paper/runtime
termination discrepancies. Termination is unchanged. Overall and per-task
accuracy now consistently exclude all non-answers (`mgvqa/metrics.py`), with
separate status/reason counts. The old merged `accuracy_attempted` is removed.

Immediately after successful `find_object`, the next user turn attaches clean
RGB first and the SAM overlay second, with explicit image descriptions. Both
attachments are saved and retained in history. Failed/rejected grounding clears
the selection and never reuses an earlier overlay. Textual find feedback says
only “Object detected.”; backend labels and identity metadata are not forwarded.
The overlay has mask tint/contours and numeric marker only, never GT names.

The scene model is configured by `--scene-model`, recorded in run configuration,
and forwarded to subprocesses. It is distinct from the planner and the legacy
name-based pointing backend. Changing it invalidates direct comparison to old
perception/agentic runs unless clearly reported as a new protocol.

GT answers belong to post-answer scoring. Simulator segmentation is used below
the planner interface for selected-mask/body association and retry centroid
prompting; masks themselves are predicted by SAM3. This is not an oracle-free
robot controller. Grasp friction is tuned; placement uses the existing tray
check and push success means execution, not guaranteed intended displacement.
These semantics have been preserved, not redesigned during cleanup.
