# Published planner prompts

| CLI mode | System | User |
|---|---|---|
| direct | direct.txt | direct_user.txt |
| perception | perception.txt | perception_user.txt |
| agentic (paper: Manipulation) | agentic.txt | agentic_user.txt |

These files are the release prompts, identical across planner models. System
text contains role, mode access, tools, answer schemas and JSON examples. User
text repeats the original question, task, optional options and current RGB;
interactive modes prepend status and the previous tool response. The first
turn omits the previous-response block. RGB is always image 1. Immediately after
successful find_object, image 2 is the SAM mask overlaid on the tool's input RGB;
the user prompt explicitly describes both attachments. Failed/rejected grounding
produces no second image, and later turns do not reattach an old mask. Both images
are retained in conversation history. Tool feedback says only "Object detected."
for a valid mask, without object names or identity metadata. Scene-graph descriptions
remain available in context for the planner to interpret the selection.

`mgvqa/prompts.py::build_user_prompt` substitutes `{question}`, `{task}`,
`{answer_options_block}`, `{step}`, `{max_steps}`, `{last_tool}`,
`{previous_tool_response_block}`, `{remaining_budget}`, `{max_actions}` and
`{perception_calls}`, plus the conditional `{segmentation_image_block}`. Task `marked_comparison` is displayed as `compare`.
Substitution is single-pass; braces inside questions/results remain literal.
Literal JSON examples in user templates would need doubled braces (`{{...}}`).
System files are loaded verbatim and do not require escaping JSON braces.

No search guidance is appended. Historical `build_type_specific_constraint`
and old prompt files remain for compatibility, but are not used to construct
release Bench user messages. Removed from the release path: mandatory plan/
belief, Count target-removal prohibition, Beneath must-remove-before-answer,
Compare search suggestions and nullable badge_points request, repeated grasp
blacklist instructions and model-family-specific conversation truncation.
Optional extra response fields may still be recorded as diagnostics.

## Details retained beyond the paper draft

- Top-left [row,col] convention for both grounding and Find answers; yes/no
  letters follow each question's shuffled options.
- Compare identity/physical-volume semantics and Beneath NOTA meaning.
- Segmentation validity before manipulation, finite turn limits, and distinct
  placed/slipped/failed grasp feedback. Push execution is not guaranteed object
  displacement. Failed executions consume budget; validation rejections do not.
- The overlay contains predicted SAM mask tint/contour and a numbered marker,
  not simulator segmentation or object names. `step_N_img1.png` and optional
  `step_N_img2.png` are the actual planner attachments at turn N. Debug scene
  graph points use `step_N_scene_graph_overlay.png` instead.

## Runtime differences that still need a paper/protocol decision

Termination/scoring were **not** changed as part of this prompt update:

1. Exhausting the action budget does not itself terminate or make an answer
   incorrect. Perception and answering remain available. Thus the system-box
   caption (“episodes ... by exhausting the manipulation budget ... incorrect”)
   conflicts with both its body and the current executor.
2. The fifth consecutive perception request ends an agentic episode before that
   tool executes; Perception mode disables this guard and uses the total-turn
   cap instead. The counter currently resets only when manipulation changes
   the scene, not after every unsuccessful/no-change manipulation.
3. There is a total-turn cap (`max_actions * 3 + 5`, normally 35) and a separate
   three-over-budget-rejections guard. Unanswered episodes have failure status.
   Per the confirmed scoring policy, overall AND per-task accuracy are correct /
   answered. Non-answer failures are excluded and listed separately by status and
   reason in worker/merged summaries. The former per-task total/max-step
   denominators and merged accuracy_attempted metric have been removed. Update
   the paper caption accordingly; no runtime termination rule was changed.

Published grasp/push syntax now maps implicitly to the latest marker, and the
planner's push distance is fixed at 10 cm as requested. Low-level physics,
segmentation freshness rules and scoring are unchanged. These prompt/interface
changes define a new evaluation protocol relative to older saved runs.
