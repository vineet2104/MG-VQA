# Question Types

This document describes each of the 6 question types in the Agentic MQA benchmark.

All 6 question types are available in **both agentic and single-pass modes**.
In single-pass mode the VLM sees a single image and must answer directly;
in agentic mode it can use perception tools and physically remove objects
before answering.

---

## Object Pool & Graspability Filter

Before any episodes run, the object pool is filtered by graspability.
An object is kept only if **at least 2 of its 3 dimensions** (from
`size_cm` in `object_annotations_*.json`) are ≤ `--max_graspable_dim_cm`
(default **8 cm**). This ensures the Robotiq 140 parallel-jaw gripper
(max opening 10 cm, default 8.5 cm) can close around the object's
cross-section.

Two disjoint pools (built by an internal train/eval split):
- **EVAL pool** — `owg_robot/assets/object_annotations_eval.json` (395 objects)
  → used by the 150-task benchmark.
- **TRAIN pool** — `owg_robot/assets/object_annotations_train.json` (1,375 objects)
  → used only for SFT data generation.

---

## 1. Counting (COUNT)

**Goal**: Count how many instances of a named object are on the table.

**Scene construction**: `_prepare_counting_scene()` builds the scene specifically
for counting episodes:
1. Pick a target name from the filtered object pool.
2. Choose count k ∈ {0, 1, 2, 3, 4, 5} uniformly at random
   (clamped to `n_objects`). k = 0 means the target is **absent**.
3. Place k target instances **first** in the spawn list — they are dropped first
   onto the table and settle at the bottom, then distractors pile on top. This
   makes targets more likely to be occluded, requiring manipulation to count.
4. Fill remaining slots with distractors (different names). Spawn order is **not
   shuffled** — it is intentional for occlusion.

**Question**: `"How many objects that resemble a {name} do you see on the table?"`

**Answer options**: 5-choice MCQ — 4 concrete numeric candidates + `"none of the
above"` (NOTA), which is **always** present in the option set (so its presence
carries no signal about whether the GT is concrete or NOTA). Distractors are
"tight", drawn near the GT (prioritising `gt ± 1, 2, 3`, then wider values up to
`N`). With 80% probability the GT count is one of the 4 concrete options (gt + 3
distractors); with 20% probability all 4 concretes are distractors and NOTA is
the correct answer. All 5 options are shuffled so NOTA's position is uniform.

**GT**: The number of target instances placed in the scene (may be 0).

---

## 2. Interactive Counting (REDUCE)

**Goal**: The VLM must physically remove x objects from the scene, then count the
remaining objects.

**Available in both modes**: In agentic mode the VLM can use `grasp_and_remove` to
remove objects before counting. In single-pass mode the VLM sees the full scene and
must reason about the answer from the MCQ options without manipulating.

**Question**: `"Remove exactly {x} object(s) from the table, then count how many
objects remain on the table."` where x ∈ {1, 2, 3, 4} (capped at `n_total − 1`).

**GT**: `n_total − x` (assumes x successful removals).

**Answer options**: 5-choice MCQ (4 concrete numeric candidates + always-present
NOTA) with the same tight-distractor + 20% NOTA logic as counting. Distractors
explicitly exclude both `n_total` and `x` so the VLM cannot trivially shortcut
the arithmetic from the question text.

---

## 3. Existence (FIND)

**Goal**: Determine whether a specific named object is present on the table. If
yes, the VLM must also localise it with a point.

**Target selection**: 50% chance the target exists in the scene (chosen as the
most-occluded object), 50% chance the target is an object name **not** in the
scene (drawn from the full object pool).

**Question**: `"Is there a {name} on the table? If yes, also point to it by
providing the (row, col) coordinates normalised to a [0, 1000] grid."`

**Answer format**: Shuffled A/B MCQ. Options are presented as `"A: yes, B: no"` or
`"A: no, B: yes"` (randomised per episode to prevent position bias). The VLM
answers with the letter.

**Point requirement**: If the VLM's chosen option resolves to "yes", it must also
provide `"point": [row, col]` in the JSON response. Coordinates are normalised to
`[0, 1000]`: row=0 is the top edge, row=1000 is the bottom, col=0 is left,
col=1000 is right.

**Evaluation**:
- "no" GT → letter match only.
- "yes" GT → letter must match **AND** the predicted point must land on the target
  object's segmentation mask. The mask comes from a fresh PyBullet render at answer
  time, reflecting the current scene state after any manipulations.

**Debug image**: `{stem}_point_check.png` — the predicted point drawn on the final
RGB. Green circle + "HIT" if correct, red circle + "MISS" if wrong.

**Answer example**:
```json
{"tool": "answer_question", "answer": "A", "point": [450, 620], "reasoning": "..."}
```

---

## 4. Same Attribute (COLOR)

**Goal**: Compare the color of two objects in the scene without revealing their
names (to prevent text-based color bias).

**Target selection**: The most-occluded object is one of the pair. The second
object is chosen from the remaining scene objects — 50/50 chance of same or
different color. If no same-color match exists, a different-color object is used
(and vice versa).

**Anonymous targets — text only, no image markers**: The two objects are designated
**A** and **B** in the prompt's `== A/B TARGETS ==` block. Images sent to the VLM
are always clean (no overlays). Each target line shows the object's normalised
`[row, col]` centroid on a `[0, 1000]` grid (row=0 top, col=0 left), e.g.

```
== A/B TARGETS ==
  A: point=[345, 520]
  B: point=[ ,  ]
```

Object names never appear anywhere in the prompt — the VLM has to look at the
image at each listed point to inspect colour.

**Question**: `"Two objects in the scene are marked A and B. Do they have the same
color?"`

**Answer format**: Shuffled A/B MCQ for yes/no — `"A: yes, B: no"` or
`"A: no, B: yes"` (randomised). The VLM answers with the letter.

**Visibility threshold — blank points**: A point is only published when the
object is **≥ 30 % visible** (`visible_area / full_area ≥ 0.30`, set by
`visibility_threshold=0.30` at every call site). Below that, the
entry is rendered as `point=[ ,  ]` — the VLM must remove obstructors before it
can answer. The block is recomputed every agentic turn from a fresh PyBullet
segmentation render, so the point automatically reappears once the object
becomes visible again (e.g. after a successful grasp). The VLM is instructed not
to answer until both A and B have non-blank points.

**A/B assignment**: Which physical object gets designated A vs B is randomised
per episode.

**Helpers**: `tools.py :: compute_ab_marker_info()` computes the visibility +
normalised centroid; `format_ab_targets_block()` renders it as text.
`draw_ab_markers()` is retained as a debug/GT helper but is **no longer called
on VLM-facing images**.

---

## 5. Relative Size (SIZE)

**Goal**: Determine which of two objects is physically larger, and estimate the
ratio of their volumes.

**Target selection**: The most-occluded object is one of the pair. The second
object is chosen from the remaining scene objects. Pairs whose volumes are within
10% of each other are skipped as ambiguous.

**Anonymous targets**: Same text-only A/B TARGETS block as Same Attribute (see
above) — clean image plus normalised `[row, col]` points, blank if the object
is < 30 % visible.

**Question**: `"Two objects in the scene are marked A and B. (1) Which one is
larger? (2) What is the approximate ratio of the volume of A to the volume of B?"`

**VLM must answer two things**:
1. **Which is larger** — letter A or B
2. **Volume ratio A/B** — a numeric value (e.g. `2.5` means A is 2.5× the volume
   of B; `0.4` means A is smaller)

**Answer options**: `["A", "B"]` (for the letter part).

**GT**:
- **Letter**: The letter of the larger object, determined by **volume** computed
  from `size_cm` annotations: `length × width × height`. Pairs within 10% volume
  ratio are rejected and a new comparison object is tried.

**Evaluation**:
- **Letter correctness** (`is_correct`): exact match of A/B letter.

**Marker assignment**: Randomised per episode.

**Answer example**:
```json
{"tool": "answer_question", "answer": "A", "ratio": 2.5, "reasoning": "Object A appears much larger..."}
```

---

## 6. Under Object (BENEATH)

**Goal**: Identify which object is hidden underneath a covering object.

**Pair selection**: `compute_cover_hidden_pairs()` uses alpha-transparency
rendering — for each candidate cover, it hides the cover in PyBullet and checks
which objects gain visible area. Produces geometrically valid (cover, hidden) pairs.
The top 5 pairs (by reveal area) are candidates; one is chosen at random.

**Question**: `"What object is under the {cover}?"`

**Answer options**: 5-choice MCQ — 4 concrete candidates + always-present
`"none of the above"` (NOTA). The 4 concrete slots are built from a deliberate
mix of present-and-absent distractors to prevent the VLM from using "which
option is NOT on the table?" as a shortcut:

1. Sample **2 scene-present** object names (excluding the cover and the GT).
2. Sample **1 scene-absent** name from `gt_lookup` (a real object that happens
   not to be in this episode).
3. The GT answer itself is always a scene-present, hidden object, so the first 3
   concrete options never contain it; with **20 %** probability the 4th concrete
   slot is another distractor and NOTA becomes the correct answer, otherwise the
   real GT fills the 4th concrete slot.
4. NOTA is then added as the 5th option and all 5 are shuffled. Ensure uniqueness.

**Tolerant GT** (`also_valid`): If the chosen cover hides multiple objects, any
sibling whose reveal area is ≥ 25 % of the primary answer's reveal area is
recorded in `also_valid` and the evaluator accepts it as correct.

**Protected objects**: None — any object can be removed.

---

## Task & Option Formulation Rules

Rules applied consistently across all 6 types to keep the benchmark fair and
leak-free:

1. **The target name never enters the prompt.** For same_attribute /
   relative_size, targets are referred to only as anonymous **A** / **B** with
   normalised centroid points. For existence / counting / under_object the name
   in the *question* is the queried name, never the GT answer identity.
2. **MCQ options are always shuffled** per episode, so the correct letter is
   uniformly distributed (no position bias).
3. **A/B assignment is randomised** per episode (which physical object is A vs B).
4. **NOTA ("none of the above")** is always present as the 5th option for the
   three 5-choice types (counting, interactive_counting, under_object) and is the
   correct answer with **~20 %** probability. Otherwise the true answer is one of
   the 4 concrete options. NOTA being always present means its presence carries
   no signal about whether the GT is concrete or NOTA.
5. **Distractor construction avoids shortcuts**:
   - counting/interactive_counting: numeric distractors are "tight" (near GT)
     and explicitly exclude values trivially derivable from the question text
     (e.g. `n_total`, `x`).
   - under_object: options mix scene-present and scene-absent names so
     "which option is NOT on the table?" is not a valid shortcut.
6. **Ground truth is invariant to manipulation.** Removing/pushing objects never
   changes the correct answer — it only changes what the VLM can *see*. This is
   what makes strategic de-occlusion the core skill under test.
7. **VLM-facing images are always clean** — no A/B markers, no GT overlays, no
   perception overlays drawn on the primary RGB. Any labelled overlay is sent as
   a *separate* second image.

## Known Issues & Caveats

Observations accumulated over the project — useful context for task re-design.


- **Occlusion gate can reject scenes.** Some seeded scenes fail the per-type
  occlusion requirement (e.g. relative_size needs ≥ 60 % max occlusion) and are
  recorded as `skipped / insufficient_occlusion`. These are excluded from the
  accuracy denominator. Expect a handful of skips per full benchmark run.
- **Pointing quality confounds cross-model comparison.** In agentic mode all
  grounding/pointing calls are pinned to a single fixed VLM
  (`--pointing_vlm_model`, default Gemini 3.1 Pro) so that differences reflect
  *planning* ability, not per-model pointing accuracy.
- **`same_attribute` colour GT comes from annotations**, not pixels. If an
  object's annotated colour disagrees with its rendered appearance under a given
  texture/lighting, the GT can look wrong to a human. Annotation quality is the
  bottleneck here.
- **Benchmark episode set was revised once.** The original CoRL benchmark
  (`gold-trajectories/final_benchmark_episodes.json`, run via `run_corl.py`,
  n_episodes=180 schedule) had 31 ill-formed episodes due to ambiguity in GT/solvable without manipulation. These were replaced to
  produce `probe_benchmark_episodes.json` (run via `run.py`, n_episodes=380
  schedule). The two are **not** interchangeable — the question-type schedule
  depends on `n_episodes`, so use the matching entry point for each.

## Summary

| Question Type | Format | GT Source | Key Anti-bias Feature |
|---|---|---|---|
| **Counting** | 5-choice MCQ (4 concrete + NOTA) | Instance count (0–5) in scene | Targets spawned first → buried under distractors |
| **Interactive Counting** | 5-choice MCQ (4 concrete + NOTA) | `n_total − x` | Requires physical manipulation (agentic); pure reasoning (singlepass) |
| **Existence** | A/B MCQ (shuffled yes/no) + point | Object presence + mask localisation | 50% absent targets; must localise precisely |
| **Same Attribute** | A/B MCQ (shuffled yes/no) | Color comparison from annotations | Anonymous A/B text points; names hidden from VLM |
| **Relative Size** | A/B MCQ + numeric ratio | Volume from `size_cm` annotations | Anonymous A/B text points; names hidden from VLM |
| **Under Object** | 5-choice MCQ (4 concrete + NOTA) | Alpha-transparency cover-hidden rendering | Geometrically validated pairs; no GT name leak |

**NOTA** = "none of the above" option. GT is NOTA with **~20 % probability** for
counting, interactive_counting, and under_object (lowered from 30 % / 50 % after
rollouts showed an over-representation of NOTA ground truths).

## Answer Format (VLM JSON)

Standard:
```json
{"tool": "answer_question", "answer": "A", "reasoning": "..."}
```

Existence "yes" (with point):
```json
{"tool": "answer_question", "answer": "A", "point": [450, 620], "reasoning": "..."}
```

Relative size (with ratio):
```json
{"tool": "answer_question", "answer": "A", "ratio": 2.5, "reasoning": "Object A is about 2.5 times larger..."}
```

## Answer Evaluation

| Question Type | How VLM answer is compared to GT |
|---|---|
| **Existence** | Letter match (A/B) directly |
| **Same Attribute** | Letter match (A/B) directly |
| **Relative Size** | Letter match (A/B) for `is_correct`; ratio within ±10% of GT for `ratio_correct` (tracked separately) |
| **Counting** | VLM may answer letter (A/B/C/D) or text; letters resolved to option text, then case-insensitive comparison |
| **Interactive Counting** | Same as counting |
| **Under Object** | Same as counting |

## Debug Artifacts

| File | When | Contents |
|---|---|---|
| `{stem}_gt_scene.png` | Every episode | Colour-coded bboxes: red=TARGET, orange=COMPARE, blue=COVER |
| `{stem}_point_check.png` | Existence (VLM answered "yes") | Predicted point on final RGB: green=HIT, red=MISS |
| `{stem}_occlusion.json` | Every episode | Per-object occlusion stats + cover-hidden pairs |
| `{stem}_question.json` | Every episode | Full question descriptor with answer_options, gt_answer, marker info |
| `vlm_steps/step_N_img1.png` | Every VLM turn | **Always clean RGB** sent to VLM (no overlays; A/B targets are text) |
| `vlm_steps/step_N_img2.png` | After perception tool | V2: scene-graph overlay (colored dots + "id: name" labels). V1: labelled perception overlay from RoboRefer/Molmo + SAM3. |
| `vlm_steps/step_N_scene_graph.yaml` | V2 only: after `perceive_scene` | Parsed YAML scene graph (objects, descriptions, relations) for offline analysis |
| `vlm_steps/step_N_point.png` | After `perceive_find_object` | Raw pointing model output (RoboRefer/Molmo/VLM) |
| `vlm_steps/step_N_seg.png` | After `perceive_find_object` | SAM3 segmentation mask |
| `vlm_steps/step_N_prompt.txt` | Every VLM turn | Full user prompt text (includes `== A/B TARGETS ==` block for same_attribute / relative_size) |
| `vlm_steps/step_N_response.json` | Every VLM turn | Raw VLM JSON response |
| `vlm_steps/system_prompt.txt` | Once per episode | System prompt sent to VLM |

**Naming convention**: `step_N_img1.png` is always saved (the first/only image the VLM
sees). `step_N_img2.png` is only saved when the VLM receives a second image — i.e., when
a perception tool has been run and produced a labelled overlay. This makes it easy to see
exactly what the VLM saw at each decision point.

## Manipulation Tools

The agentic loop exposes two manipulation tools to the VLM, both counted against
the `--max_actions` budget:

- **`grasp_and_remove(object_id)`** — 3 × 6-DOF GraspGen attempts per call.
  Re-perceives with SAM3 between attempts; on success the object is placed in
  the tray and disappears from the scene.
- **`push(object_id, direction, distance_cm)`** — closed-gripper lateral sweep.
  `direction` is one of `"left" | "right" | "up" | "down"` (image-relative);
  `distance_cm` is optional (clamped to [3, 20] cm, default 10). The object
  stays on the table but at a new position; all marker IDs go stale afterwards
  (VLM is instructed to re-perceive).
