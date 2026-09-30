# Marked Three-Way Scene Bank: Data Format and Replay Protocol

This document specifies the historical frozen-data format for the MG-VQA marked-size benchmark:

> Three objects carry texture-baked badges **A** (circle), **B** (square), and **C** (triangle). Which labelled object has the largest volume?

The purpose of the format is to decouple **scene generation and quality control** from **VLM benchmarking**. A scene is generated and accepted once, saved as a portable package, and then replayed unchanged by every baseline or ablation arm.

---

## 1. Design goals

A frozen scene package must:

1. replay in a fresh PyBullet process without regenerating random drops or physics;
2. render in both PyBullet and Blender;
3. preserve the texture-baked A/B/C markers exactly;
4. support live manipulation after replay (push, grasp, and removal);
5. avoid stale PyBullet body IDs, absolute local paths, and opaque `.bullet` snapshots;
6. keep answer-bearing and label-to-object metadata out of an evaluated model's prompt;
7. be inspectable and suitable for a later public-data release.

### What is frozen

For every accepted scene, the package freezes:

- the post-settle, post-containment-wall-removal physical state;
- the exact assets, poses, velocities, and dynamics for all remaining objects;
- the accepted three-object A/B/C assignment;
- the exact baked marker texture PNGs;
- the question text and A/B/C options;
- the answer, volumes, badge visibility diagnostics, and realised occlusion tier;
- the exact CCO table material copied into the package, plus per-file checksums;
- the render nuisance seed as an additional audit field (not the source of truth for replay).

### What is *not* replayed

A package does not contain a PyBullet process snapshot. On replay, it creates a new simulation and loads each object from its URDF. This is deliberate: PyBullet `.bullet` snapshots depend on ephemeral body IDs and local absolute asset paths, making them poor release artifacts and unreliable across fresh processes.

---

## 2. Current generation run

The current large generation root is:

```text
rollouts/marked_threeway_bankgen_20260820_1535/
```

It contains five independent 60-attempt shards (300 generation attempts total):

| Shard | Attempts | Base seed |
|---|---:|---:|
| root generation | 60 | 77,000,000 |
| `shard1` | 60 | 78,000,000 |
| `shard2` | 60 | 79,000,000 |
| `shard3` | 60 | 80,000,000 |
| `shard4` | 60 | 81,000,000 |

Each shard writes:

```text
<shard>/
  generation.log                 # append-only generator log
  generation.exit_code           # written when the process exits
  generation/
    sites.json                   # accepted-scene records + aggregate counters
    index.html                   # human-inspection gallery
    sceneXX_before.png           # pre-badge diagnostic image
    sceneXX_scene.png            # marked PyBullet scene image
    sceneXX_revealed.png         # non-target clutter parked oracle image
    sceneXX_badge_{A,B,C}_*.png  # marker close-ups
  packages/
    <unique-scene-id>/           # one portable package per accepted scene
```

Package IDs include the full generation seed (for example, `scene02_seed77002000`) so package directories cannot collide when shard outputs are combined.

---

## 3. Acceptance rule used during generation

An accepted scene has all of the following properties:

1. at least three objects have valid texture-space badge sites;
2. a selected triple has a non-trivial volume relationship:

   ```text
   1.20 <= largest-volume / runner-up-volume <= 3.00
   ```

3. at least one selected **marker disc itself** is hidden initially:

   ```text
   projected marker visibility fraction <= 0.15
   ```

4. the generator tests every cheap-valid triple, not only the first random one;
5. after moving **non-selected clutter** away in an oracle simulation, all three selected marker discs are readable:

   ```text
   projected marker visibility fraction >= 0.85
   ```

6. every badge survives UV-island clipping during baking (`badge_paint_frac >= 0.60`).

The true largest object's realised object-occlusion rank is recorded as one of:

```text
gt_least_occluded
gt_middle_occluded
gt_most_occluded
```

Generation does **not** target a rank. The final bank is balanced by stratified sampling after generation.

---

## 4. Per-scene package layout

```text
<scene-id>/
  scene.json
  preview.png                    # optional generated initial marked view
  textures/
    marker_00.png
    marker_01.png
    marker_02.png
  table_texture/                 # exact copied CCO material maps
  runtime.json
  annotation.json
```

`scene.json` is the public physical scene and task-input representation. `runtime.json` and `annotation.json` are evaluator-private before a benchmark closes.

The generic texture names intentionally do not encode `A`, `B`, or `C`.

### 4.1 `scene.json` — physical/public scene specification

Schema version: `2`.

Representative structure:

```json
{
  "schema_version": 1,
  "scene_id": "scene02_seed77002000",
  "generation_seed": 77002000,
  "replay_seed": 77002000,
  "objects": [
    {
      "object_key": "object_000",
      "name": "...",
      "folder": "unified_objects/...",
      "base_position": [x, y, z],
      "base_orientation": [qx, qy, qz, qw],
      "linear_velocity": [vx, vy, vz],
      "angular_velocity": [wx, wy, wz],
      "dynamics": {
        "mass": 0.3,
        "lateral_friction": 0.8,
        "local_inertia_diagonal": [ixx, iyy, izz],
        "restitution": 0.01,
        "rolling_friction": 0.005,
        "spinning_friction": 0.005,
        "contact_damping": -1,
        "contact_stiffness": -1
      }
    }
  ],
  "textures": [
    {"path": "textures/marker_00.png", "sha256": "..."}
  ],
  "table_texture": {
    "mode": "packaged_cc0",
    "path": "table_texture",
    "source_basename": "WoodFloor039_2K-JPG",
    "files": [
      {"path": "table_texture/WoodFloor039_2K_Color.jpg", "sha256": "..."}
    ]
  },
  "question": {
    "question_type": "marked_comparison",
    "question_subtype": "size_three_way",
    "question": "Which labelled object has the largest volume: A, B, or C?",
    "badge_description": "...",
    "answer_options": ["A", "B", "C"]
  },
  "preview": {"path": "preview.png", "sha256": "..."},
  "render_contract": {
    "camera": "MG-VQA default top-down camera",
    "state_capture": "post-settle, post-containment-wall-removal",
    "badge_storage": "texture-baked PNG copied into this package"
  }
}
```

Important semantics:

- `folder` is relative to the installed MG-VQA asset root supplied at replay time (`--asset_path`); it is not an absolute local path.
- `base_position` and `base_orientation` are PyBullet base/inertial-frame values accepted by `loadURDF` / `resetBasePositionAndOrientation`.
- `object_key` is a stable package identifier. It replaces process-local PyBullet body IDs.
- `replay_seed` is retained as a reproducibility/audit field. It does **not** cause object re-spawning and is **not** used to choose a table material during replay.
- `table_texture` is the render source of truth. `mode: "packaged_cc0"` points to a package-local copy of every PBR map Blender needs (color, roughness, normal, etc.) and lists SHA-256 hashes. `mode: "plain_urdf"` explicitly requests the original URDF material; it does not fall back to a random cache texture.
- `preview.png` is a convenient initial marked-image audit artifact. It is not used as the interactive simulator state.

### 4.2 `textures/marker_*.png` — baked marker textures

Each texture is a copy of an accepted texture map after the badge was composited and clipped to the correct UV island. The badge is therefore physically attached to its object's surface and moves correctly under later manipulation.

The package includes SHA-256 checksums for texture integrity auditing.

### 4.3 `runtime.json` — evaluator-only render binding

Representative structure:

```json
{
  "schema_version": 1,
  "scene_id": "scene02_seed77002000",
  "markers": [
    {
      "label": "A",
      "object_key": "object_005",
      "badge_texture": "textures/marker_00.png"
    }
  ]
}
```

On replay, fresh PyBullet body IDs are created. This file maps the generic stored texture files to those fresh bodies and pushes each texture to both:

```text
PyBullet visual shape
Blender renderer object texture
```

This file is evaluator-private during a blind benchmark because it directly encodes the A/B/C-to-object association. It is **not** supplied to the evaluated agent or inserted into prompts.

### 4.4 `annotation.json` — evaluator-only labels and diagnostics

Representative contents:

```json
{
  "schema_version": 1,
  "scene_id": "scene02_seed77002000",
  "ground_truth": {
    "gt_letter": "B",
    "triple_tier": "gt_middle_occluded",
    "largest_occlusion_rank": 1,
    "largest_runnerup_ratio": 2.84,
    "largest_smallest_ratio": 4.66,
    "n_labels_hidden_t0": 1,
    "occlusions": {"A": 0.68, "B": 0.08, "C": 0.00}
  },
  "question_descriptor": {"...": "full evaluator descriptor"},
  "marker_diagnostics": [
    {
      "label": "A",
      "object_key": "object_005",
      "badge_frac": 0.12,
      "badge_frac_revealed": 1.0,
      "volume_cm3": 1907.8,
      "world_point": [x, y, z]
    }
  ]
}
```

It contains:

- the correct A/B/C answer;
- true object volumes and volume-ratio diagnostics;
- true largest-object occlusion tier and object occlusions;
- initially-hidden and clutter-cleared badge visibility fractions;
- saved badge site metadata for evaluator auditing;
- the complete question descriptor, with object keys instead of stale body IDs.

The `world_point` fields are evaluator diagnostics only. They are never sent to the agent.

---

## 5. Frozen bank manifest

After generation, `select_scene_bank.py` combines one or more `sites.json` files and performs deterministic stratified sampling.

Example:

```bash
python -m experiments.marked_size.select_scene_bank \
  shard0/generation/sites.json \
  shard1/generation/sites.json \
  shard2/generation/sites.json \
  shard3/generation/sites.json \
  shard4/generation/sites.json \
  --per_tier 20 \
  --seed 88000000 \
  --out frozen_marked_bank.json
```

The output manifest contains:

```json
{
  "schema_version": 1,
  "description": "Frozen, balanced marked-size replay bank ...",
  "per_tier_requested": 20,
  "seed": 88000000,
  "available_by_tier": {
    "gt_least_occluded": 0,
    "gt_middle_occluded": 0,
    "gt_most_occluded": 0
  },
  "shortage": {},
  "selected": [
    {
      "scene_file": "relative/path/to/scene.json",
      "gt": {"triple_tier": "gt_middle_occluded", "...": "..."},
      "scene_package": {"scene_id": "...", "replay_seed": 77002000},
      "...": "original generation diagnostics"
    }
  ],
  "selected_by_tier": {
    "gt_least_occluded": 20,
    "gt_middle_occluded": 20,
    "gt_most_occluded": 20
  }
}
```

`scene_file` paths are made relative to the final manifest, so the manifest and packages can be moved together as a release bundle.

If any tier has fewer than the requested number of scenes, the tool writes the partial manifest, reports `shortage`, and exits non-zero. That is a signal to generate more scenes rather than silently construct an unbalanced evaluation set.

---

## 6. Replay and benchmark execution

Run a fixed bank with any marked-condition arm:

```bash
python run_marked.py \
  --mode agentic \
  --marked_condition agentic \
  --scene_bank /path/to/frozen_marked_bank.json \
  --renderer blender \
  --model_name <model> \
  --output_dir /path/to/results
```

Supported marked-condition arms:

| Condition | Mode | Capabilities |
|---|---|---|
| `direct` | singlepass | one initial image; no tools |
| `perception` | agentic | perception tools; no scene changes |
| `perception_suggest` | agentic | perception tools plus a soft badge-search suggestion |
| `perception_report` | agentic | perception tools plus optional nullable badge-point reporting |
| `agentic` | agentic | same reporting interface plus push/grasp/remove |

With `--scene_bank`:

1. the worker loads the frozen manifest once at startup;
2. it creates an empty fresh `RobotEnvUI` for each episode;
3. it loads all package URDFs and restores their frozen poses and dynamics;
4. it starts Blender when requested;
5. it applies the exact package-local CCO table material in Blender (or explicitly retains plain URDF material);
6. it reapplies all packaged badge textures to PyBullet and Blender;
7. it hydrates evaluator body IDs into the saved question descriptor;
8. it executes the requested arm;
9. it never replaces a rejected slot with a random retry scene.

Thus all arms receive exactly the same accepted physical scenes and question labels.

---

## 7. Validation before benchmarking

Use the replay validator before freezing or distributing a bank:

```bash
# Fast state/segmentation validation
python -m experiments.marked_size.validate_scene_bank \
  frozen_marked_bank.json \
  --renderer pybullet \
  --out validation_pybullet.json

# Confirm live Blender synchronization and baked-texture application
python -m experiments.marked_size.validate_scene_bank \
  frozen_marked_bank.json \
  --renderer blender \
  --blender_port 9888 \
  --out validation_blender.json
```

For each package, the validator checks:

- package/schema presence and texture files;
- asset-relative URDF availability;
- restored object count;
- restored A/B/C marker bindings;
- hydrated correct answer and marked object names;
- segmentation-based initial marker visibility fractions against stored diagnostics (default tolerance `0.03`);
- deterministic PyBullet preview consistency when applicable;
- successful live Blender synchronization, package-local table-material application, and marker texture application.

A one-scene round-trip validation has already passed for both PyBullet and Blender:

```text
A: 0.12 visible
B: 1.00 visible
C: 1.00 visible
```

The actual `run_marked.py --scene_bank ...` worker path was also smoke-tested through scene restoration, question construction, image generation, and the point immediately before an intentionally invalid test-model call.

---

## 8. Release policy and data leakage

The benchmark must distinguish between **renderable public data** and **private evaluator data**.

| File | Pre-evaluation release | Post-evaluation archival release | Reason |
|---|---|---|---|
| `scene.json` | may release | may release | physical scene and public task text |
| `preview.png` | may release for inspection sets | may release | visual scene audit artifact |
| `textures/marker_*.png` | may release with public images | may release | marker identity is visually present in the scene anyway |
| `table_texture/` | may release | may release | package-local CC0 Blender material required for render reproducibility |
| `runtime.json` | keep private | may release after benchmark closes | explicitly maps A/B/C texture to object key |
| `annotation.json` | keep private | may release after benchmark closes | correct answer, volumes, label/object mapping, diagnostic coordinates |
| frozen manifest with `gt` fields | keep private | may release after benchmark closes | contains answer/tier metadata |

For a blind test set, public users can inspect a rendered image but must not receive machine-readable label-to-object bindings or answer-bearing metadata. The evaluator uses `runtime.json` and `annotation.json` locally to replay and score episodes. After the evaluation period, all files can be released as a fully reproducible archive, subject to asset-license terms.

---

## 9. Implementation locations

| File | Responsibility |
|---|---|
| `experiments/marked_size/scene_bank.py` | package write/load, package-local table material capture, fresh-scene instantiation, texture replay, descriptor hydration |
| `experiments/marked_size/migrate_table_textures.py` | deterministic v1→v2 upgrade that copies historical scene materials from their saved replay seeds |
| `experiments/marked_size/make_scenes.py` | scene generation and package emission via `--scene_bank_dir` |
| `experiments/marked_size/select_scene_bank.py` | deterministic tier-balanced manifest construction |
| `experiments/marked_size/validate_scene_bank.py` | fresh-process PyBullet/Blender replay validation |
| `core/cli.py` | `--scene_bank` command-line option |
| `core/worker.py` | frozen-package episode replay in the actual benchmark worker |
| `experiments/marked_size/integrate.py` | marked triple selection, marker baking, and saved descriptor construction |

---

## 10. Reproducibility checklist

Before starting a multi-arm baseline experiment:

- [ ] all generation shards completed successfully;
- [ ] inspect the combined HTML galleries and selected marker crops;
- [ ] create a balanced manifest with `select_scene_bank.py`;
- [ ] validate every selected package in PyBullet;
- [ ] validate every selected package in Blender;
- [ ] archive the frozen manifest, packages, source `sites.json`, generator logs, and validator reports;
- [ ] run every ablation with the **same** `--scene_bank` manifest;
- [ ] record model, renderer, lighting, action budget, and code revision alongside results.
