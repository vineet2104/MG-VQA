"""
questions.py — Question generation, GT computation, and scene preparation
==========================================================================

Contains:
  - Annotation/object pool parsing
  - Scene construction helpers (counting duplicate injection)
  - All 6 question type generators
  - Question descriptor orchestrator
  - Protected-object ID resolution
"""

import os
import json
import random
import numpy as np
import cv2
from collections import Counter
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Extra, opt-in question types. Deliberately NOT part of QUESTION_TYPES:
# build_episode_schedule() shuffles a list whose LENGTH is len(QUESTION_TYPES),
# so appending here would reassign question types to episode indices and
# invalidate both frozen benchmark JSONs.  Selectable only via
# --question_type <name>, which takes the non-random schedule branch.
EXTRA_QUESTION_TYPES = [
    "marked_comparison",
]

# Types whose two targets are designated A/B.  Previously these were spelled
# out as ("same_attribute", "relative_size") tuples in ~8 places in
# core/worker.py; the marked type joins the family but must NOT get the A/B
# coordinate block.
AB_TYPES = ("same_attribute", "relative_size", "marked_comparison")
AB_COORD_TYPES = ("same_attribute", "relative_size")   # get the point block

QUESTION_TYPES = [
    "counting",
    "interactive_counting",
    "existence",
    "same_attribute",
    "relative_size",
    "under_object",
]

# ── Benchmark V2 (branch: benchmark-v2) ────────────────────────────────
# A 4-task suite. These use their OWN names (never colliding with the V1
# types above) so schedule / scoring / prompting dispatch stays fully
# separated. See docs/BENCHMARK_V2_DESIGN.md.
#   count         → integer answer (no MCQ)
#   find          → existence + point (30% visibility gate removed)
#   beneath       → under_object, 4-option MCQ (NOTA always present)
#   compare       → marked_comparison (3 texture-baked A/B/C badges; which
#                   labelled object has the largest volume?).  ``compare`` is
#                   a V2-vocabulary ALIAS for the runtime type
#                   ``marked_comparison`` (Mika's marker pipeline); the worker
#                   maps it via ``_V2_TO_V1_SCENE`` and the descriptor / result
#                   still carry ``question_type = "marked_comparison"``.
#
# NOTE: `rearrangement` was part of the original V2 draft (world-state-
# conditioned removal, F1 scored) but has been DROPPED from the default
# benchmark.  The full rearrangement implementation (generator, FoV-visibility
# scoring, reasoning-loop branch) is left intact so it can still be run
# explicitly via `--question_type rearrangement`; it is simply no longer in
# the default "random" schedule pool.
QUESTION_TYPES_V2 = [
    "count",
    "find",
    "beneath",
    "compare",
]


# ---------------------------------------------------------------------------
# Display-name construction
# ---------------------------------------------------------------------------
def _dedup_color_name(color: str, name: str) -> str:
    """Build a display name ``"{color} {name}"`` WITHOUT repeating colour
    words already present in ``name``.

    Many annotations bake a colour into the ``name`` field (e.g.
    color="pink", name="pink crayon"), so a naive ``f"{color} {name}"`` yields
    "pink pink crayon".  This drops colour words from the prefix that already
    appear in the name (case-insensitive, whole-word), and if that removes
    the entire prefix (name already fully describes the colour) it returns the
    name unchanged.

    Examples:
        ("pink", "pink crayon")            → "pink crayon"
        ("white and tan", "white boat shoe") → "tan white boat shoe"
        ("navy and grey", "navy boat shoe")  → "grey navy boat shoe"
        ("orange and grey", "baby spoon")    → "orange and grey baby spoon"
    """
    color = (color or "").strip()
    name = (name or "").strip()
    if not color:
        return name
    if not name:
        return color
    name_words = set(w.lower() for w in name.split())
    # Keep colour tokens not already in the name; drop the connective "and"
    # if it becomes dangling.
    kept = []
    for tok in color.split():
        if tok.lower() == "and":
            kept.append(tok)
            continue
        if tok.lower() in name_words:
            continue
        kept.append(tok)
    # Clean up dangling / leading / trailing / doubled "and".
    while kept and kept[0].lower() == "and":
        kept.pop(0)
    while kept and kept[-1].lower() == "and":
        kept.pop()
    dedup = []
    for tok in kept:
        if tok.lower() == "and" and dedup and dedup[-1].lower() == "and":
            continue
        dedup.append(tok)
    prefix = " ".join(dedup).strip()
    return f"{prefix} {name}".strip() if prefix else name


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_annotations_gt_lookup(
    annotations_path: str,
    asset_dir: str,
    max_graspable_dim_cm: float = 8.0,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Parse ``annotations.json`` into (gt_lookup, obj_pool).

    gt_lookup: {name → {color, shape, size_cm, ...}}  (one per unique name)
    obj_pool:  [{name, obj_folder, color, shape, ...}]  (may have duplicate names)

    Args:
        max_graspable_dim_cm: Maximum smallest dimension (cm) for an object
            to be included.  Objects whose ``min(length, width, height)`` from
            ``size_cm`` exceeds this value are filtered out because the
            Robotiq 140 gripper cannot close around them.
            Set to ``None`` or ``0`` to disable the filter.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        all_annotations = json.load(f)

    gt_lookup: Dict[str, Dict] = {}
    obj_pool: List[Dict] = []
    n_accepted = 0
    n_filtered_size = 0

    for ann in all_annotations:
        if ann.get("status") != "ACCEPT":
            continue
        obj_folder = ann["obj_name"]
        if not os.path.isfile(os.path.join(asset_dir, obj_folder, "model.urdf")):
            continue
        n_accepted += 1

        # --- Graspability filter: at least 2 of 3 dims must be <= threshold ---
        # This ensures the gripper can close around the object's cross-section.
        if max_graspable_dim_cm:
            sz = ann.get("size_cm") or {}
            dims = sorted([
                sz.get("length", 0),
                sz.get("width", 0),
                sz.get("height", 0),
            ])
            # dims[1] is the second-smallest; if it exceeds the threshold,
            # at most one dimension fits in the gripper — skip.
            if dims[1] > max_graspable_dim_cm:
                n_filtered_size += 1
                continue

        name = ann.get("name", obj_folder.replace("_", " "))
        obj_pool.append({
            "name": name,
            "obj_folder": obj_folder,
            "color": ann.get("color", ""),
            "shape": ann.get("shape", ""),
            "size_cm": ann.get("size_cm"),
            "graspable": ann.get("graspable", ""),
            "ambiguous": ann.get("ambiguous", ""),
        })
        if name not in gt_lookup:
            gt_lookup[name] = {
                "color": ann.get("color", ""),
                "shape": ann.get("shape", ""),
                "display_name": name,
                "size_cm": ann.get("size_cm"),
            }

    if n_filtered_size:
        print(f"[questions] Graspability filter: kept {len(obj_pool)}/{n_accepted} "
              f"objects (removed {n_filtered_size} with 2nd-smallest dim > {max_graspable_dim_cm} cm)")

    return gt_lookup, obj_pool


def parse_annotations_full_jsonl(
    jsonl_path: str,
    asset_dirs: List[str],
    min_dim_cm: float = 2.0,
) -> Tuple[Dict[str, Dict], List[Dict], List[Dict]]:
    """Parse ``annotations_full.jsonl`` into (gt_lookup, target_pool, distractor_pool).

    Each object's display name is ``final_color + " " + final_name``
    (e.g. "brown plastic sauce bottle").

    Objects with ``role`` == '' or 'remove' are skipped entirely.
    Objects whose smallest dimension (from the source annotations) is
    below ``min_dim_cm`` are filtered out (too flat to grasp reliably).

    Returns:
        gt_lookup:       {display_name → {color, shape, size_cm, obj_folder, source, ...}}
        target_pool:     [{name, obj_folder, color, ...}]  (role == 'target')
        distractor_pool: [{name, obj_folder, color, ...}]  (role == 'distractor')
    """
    # ── Load size data from source annotation files ──
    size_lookup: Dict[str, Dict] = {}
    for adir in asset_dirs:
        ann_file = os.path.join(adir, "annotations.json")
        if os.path.isfile(ann_file):
            with open(ann_file, "r", encoding="utf-8") as f:
                for obj in json.load(f):
                    oname = obj.get("obj_name", "")
                    if oname and obj.get("size_cm"):
                        size_lookup[oname] = obj["size_cm"]

    # ── Parse the JSONL ──
    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    gt_lookup: Dict[str, Dict] = {}
    target_pool: List[Dict] = []
    distractor_pool: List[Dict] = []
    n_skipped_role = 0
    n_skipped_size = 0
    n_skipped_nomodel = 0

    for entry in entries:
        role = entry.get("role", "")
        if role not in ("target", "distractor"):
            n_skipped_role += 1
            continue

        obj_folder = entry["obj_name"]
        source = entry.get("source", "")

        # Find model directory and build relative obj_folder path
        # so that asset_path/obj_folder/model.urdf resolves correctly.
        model_found = False
        resolved_folder = obj_folder
        for adir in asset_dirs:
            if os.path.isfile(os.path.join(adir, obj_folder, "model.urdf")):
                # Store as "subdir/obj_name" relative to parent of asset_dirs
                # e.g. "unified_objects/hope_obj_002"
                subdir_name = os.path.basename(adir)
                resolved_folder = os.path.join(subdir_name, obj_folder)
                model_found = True
                break
        if not model_found:
            n_skipped_nomodel += 1
            continue

        # Min-dimension filter
        sz = size_lookup.get(obj_folder, {})
        dims = sorted([
            sz.get("length", 0),
            sz.get("width", 0),
            sz.get("height", 0),
        ])
        if min_dim_cm and dims[0] < min_dim_cm:
            n_skipped_size += 1
            continue

        # Build display name: "final_color final_name" (dedup repeated colour)
        final_color = (entry.get("final_color") or "").strip()
        final_name = (entry.get("final_name") or "").strip()
        display_name = _dedup_color_name(final_color, final_name)

        obj_entry = {
            "name": display_name,
            "obj_folder": resolved_folder,
            "color": final_color,
            "shape": entry.get("category", ""),
            "size_cm": sz if sz else None,
            "source": source,
            "role": role,
        }

        if role == "target":
            target_pool.append(obj_entry)
        else:
            distractor_pool.append(obj_entry)

        if display_name not in gt_lookup:
            gt_lookup[display_name] = {
                "color": final_color,
                "shape": entry.get("category", ""),
                "display_name": display_name,
                "size_cm": sz if sz else None,
                "obj_folder": resolved_folder,
                "source": source,
                "role": role,
            }

    total_kept = len(target_pool) + len(distractor_pool)
    print(f"[annotations_full] Loaded {len(entries)} entries: "
          f"{len(target_pool)} targets, {len(distractor_pool)} distractors "
          f"({total_kept} total)")
    if n_skipped_role:
        print(f"  Skipped {n_skipped_role} (no role / role='remove')")
    if n_skipped_size:
        print(f"  Skipped {n_skipped_size} (min dim < {min_dim_cm} cm)")
    if n_skipped_nomodel:
        print(f"  Skipped {n_skipped_nomodel} (no model.urdf found)")

    return gt_lookup, target_pool, distractor_pool


def parse_object_annotations_json(
    json_path: str,
    asset_dirs: List[str],
    min_dim_cm: float = 2.0,
) -> Tuple[Dict[str, Dict], List[Dict], List[Dict]]:
    """Parse the new ``object_annotations.json`` format (single JSON list).

    Schema per entry (see ``docs/OBJECT_ANNOTATIONS.md``):
        obj_name, source, category, role, role_source, graspable,
        name, color, size_cm{length,width,height}

    Display name is ``color + " " + name`` (e.g. "tan and brown lion figurine").
    Returns the same triple as ``parse_annotations_full_jsonl`` so the rest
    of the pipeline (target_pool, distractor_pool, gt_lookup) is unchanged.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"{json_path}: expected JSON list, got {type(entries).__name__}")

    gt_lookup: Dict[str, Dict] = {}
    target_pool: List[Dict] = []
    distractor_pool: List[Dict] = []
    n_skipped_role = 0
    n_skipped_size = 0
    n_skipped_nomodel = 0

    for entry in entries:
        role = entry.get("role", "")
        if role not in ("target", "distractor"):
            n_skipped_role += 1
            continue

        obj_folder = entry["obj_name"]
        source = entry.get("source", "")

        # Resolve the URDF path (same logic as the JSONL variant).
        model_found = False
        resolved_folder = obj_folder
        for adir in asset_dirs:
            if os.path.isfile(os.path.join(adir, obj_folder, "model.urdf")):
                subdir_name = os.path.basename(adir)
                resolved_folder = os.path.join(subdir_name, obj_folder)
                model_found = True
                break
        if not model_found:
            n_skipped_nomodel += 1
            continue

        # Min-dim filter using inline size_cm.
        sz = entry.get("size_cm") or {}
        dims = sorted([
            float(sz.get("length", 0) or 0),
            float(sz.get("width", 0) or 0),
            float(sz.get("height", 0) or 0),
        ])
        if min_dim_cm and dims[0] < min_dim_cm:
            n_skipped_size += 1
            continue

        color = (entry.get("color") or "").strip()
        name = (entry.get("name") or "").strip()
        display_name = _dedup_color_name(color, name)

        obj_entry = {
            "name": display_name,
            "obj_folder": resolved_folder,
            "color": color,
            "shape": entry.get("category", ""),
            "size_cm": sz if sz else None,
            "source": source,
            "role": role,
        }

        if role == "target":
            target_pool.append(obj_entry)
        else:
            distractor_pool.append(obj_entry)

        if display_name not in gt_lookup:
            gt_lookup[display_name] = {
                "color": color,
                "shape": entry.get("category", ""),
                "display_name": display_name,
                "size_cm": sz if sz else None,
                "obj_folder": resolved_folder,
                "source": source,
                "role": role,
            }

    total_kept = len(target_pool) + len(distractor_pool)
    print(f"[object_annotations] Loaded {len(entries)} entries: "
          f"{len(target_pool)} targets, {len(distractor_pool)} distractors "
          f"({total_kept} total)")
    if n_skipped_role:
        print(f"  Skipped {n_skipped_role} (no role / role='remove')")
    if n_skipped_size:
        print(f"  Skipped {n_skipped_size} (min dim < {min_dim_cm} cm)")
    if n_skipped_nomodel:
        print(f"  Skipped {n_skipped_nomodel} (no model.urdf found)")

    return gt_lookup, target_pool, distractor_pool


def parse_obj_gt_lookup(obj_list_path: str) -> Dict[str, Dict[str, str]]:
    """Legacy parser for ``obj_list.txt`` (sungsu format)."""
    gt_lookup: Dict[str, Dict[str, str]] = {}
    with open(obj_list_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                name, color, shape = parts[0], parts[1], parts[2]
                gt_lookup[name] = {"color": color, "shape": shape, "display_name": name}
    return gt_lookup


# ---------------------------------------------------------------------------
# Scene construction helpers
# ---------------------------------------------------------------------------

def _bbox_volume(entry: Dict) -> float:
    """Bounding-box volume (cm³) of a pool entry from its ``size_cm`` dims."""
    s = entry.get("size_cm") or {}
    return ((s.get("length", 0) or 0) * (s.get("width", 0) or 0)
            * (s.get("height", 0) or 0))


def _prepare_counting_scene(
    eligible_pool: List[Dict],
    n_objects: int,
    rng: random.Random,
    target_pool: List[Dict] = None,
    distractor_pool: List[Dict] = None,
) -> Tuple[List[str], List[str], str, int]:
    """Build a scene for counting: target instances first, then distractors.

    Args:
        eligible_pool:   Full eligible pool (legacy path).  Used only when
                         ``target_pool`` / ``distractor_pool`` are None.
        target_pool:     Objects with ``role='target'`` in annotations_full.
                         When provided, the counting target is drawn ONLY
                         from this pool.
        distractor_pool: Objects with ``role='distractor'``.  When provided,
                         filler objects are drawn ONLY from this pool
                         (never from targets) so at most one target name
                         appears in the scene.

    Returns (scene_names, scene_folders, target_name, target_count).
    Spawn order is intentional — do NOT shuffle afterward.
    """
    # Select the source pool for the TARGET NAME.  When annotations_full
    # is in use, the target must come from the designated ``target_pool``
    # (role='target') — a distractor must never become the thing being
    # counted.  Falling back to ``eligible_pool`` keeps the legacy path
    # working.
    if target_pool is not None and len(target_pool) > 0:
        target_source = target_pool
    else:
        target_source = eligible_pool

    # Deterministic dedup preserving insertion order (dict.fromkeys).
    # Previously this used ``list({e["name"] for e in eligible_pool})`` —
    # a set comprehension whose iteration order depends on Python's
    # hash randomisation (PYTHONHASHSEED).  That caused the scene chosen
    # here to differ between identically-seeded runs (e.g. counting
    # episodes ep_009 / ep_010 picking different targets across
    # otherwise-identical rollouts).
    target_names = list(dict.fromkeys(e["name"] for e in target_source))
    target_name = rng.choice(target_names)
    variants = [e for e in target_source if e["name"] == target_name]
    target_count = rng.randint(0, 5)
    target_count = min(target_count, n_objects)

    t_names, t_folders = [], []
    for _ in range(target_count):
        v = rng.choice(variants)
        t_names.append(v["name"])
        t_folders.append(v["obj_folder"])

    # Filler pool: distractors ONLY (annotations_full) or rest of eligible
    # pool (legacy).  Always exclude the target name so the final count
    # matches ``target_count`` exactly.
    if distractor_pool is not None:
        filler_source = [e for e in distractor_pool if e["name"] != target_name]
    else:
        filler_source = [e for e in eligible_pool if e["name"] != target_name]

    n_dist = n_objects - target_count
    d_names, d_folders = [], []
    if n_dist > 0 and filler_source:
        # Reserve the LAST filler slot for a deliberately LARGE cover object
        # (one of the top-25 by bounding-box volume).  Spawned last, it lands
        # on top of the pile and adds occlusion over the buried targets.
        big_pool = sorted(filler_source, key=_bbox_volume, reverse=True)[:25]
        big_cover = rng.choice(big_pool) if (n_dist >= 1 and big_pool) else None
        rest_source = [e for e in filler_source if e is not big_cover]
        n_rest = n_dist - (1 if big_cover is not None else 0)
        entries = rng.sample(rest_source, k=min(n_rest, len(rest_source)))
        if big_cover is not None:
            entries = entries + [big_cover]   # cover LAST → spawned on top
        d_names = [e["name"] for e in entries]
        d_folders = [e["obj_folder"] for e in entries]

    return t_names + d_names, t_folders + d_folders, target_name, target_count


# ---------------------------------------------------------------------------
# Protected-object helper
# ---------------------------------------------------------------------------

def _get_protected_ids(occ_stats: Dict, names: List[str]) -> List[int]:
    """Get env object IDs for named objects (for removal protection)."""
    name_to_id = occ_stats.get("name_to_id", {})
    ids = []
    for n in names:
        oid = name_to_id.get(n)
        if oid is not None:
            ids.append(int(oid))
    return ids


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------

def _tight_numeric_distractors(
    gt: int, n_scene: int, rng: random.Random, k: int = 3,
    exclude: set = None,
) -> List[int]:
    """Pick *k* numeric distractors clustered near *gt*.

    Prefers gt±1, gt±2, gt±3 (in that order), then falls back to wider
    values within [0, n_scene].  Never returns *gt* or anything in *exclude*.
    See QUESTION_UPDATES.MD §4.1.
    """
    exclude = set(exclude or ()) | {gt}
    candidates: List[int] = []
    # Near band: ±1, ±2, ±3 (prioritised)
    for delta in (1, 2, 3):
        for sign in (-1, +1):
            v = gt + sign * delta
            if 0 <= v <= n_scene and v not in exclude and v not in candidates:
                candidates.append(v)
    # Fall back to wider values
    far = [v for v in range(0, n_scene + 1)
           if v not in exclude and v not in candidates]
    rng.shuffle(far)
    candidates.extend(far)
    return candidates[:k]


def _generate_counting_question(
    scene_obj_names: List[str],
    gt_lookup: Dict,
    rng: random.Random,
    counting_target_name: str = None,
    counting_target_count: int = None,
    integer_answer: bool = False,
) -> dict:
    """Counting: "How many X are on the table?"

    V1 (default): 5-option MCQ (4 numeric + NOTA).
    V2 (``integer_answer=True``): free-form integer — no options, no NOTA,
        no range hint.  The VLM must predict the exact count.
    """
    name_counts = Counter(scene_obj_names)

    if counting_target_name is not None and counting_target_count is not None:
        queried = counting_target_name
        gt_count = counting_target_count
    else:
        multi = [n for n, c in name_counts.items() if c >= 2]
        queried = rng.choice(multi) if multi else rng.choice(scene_obj_names)
        gt_count = name_counts[queried]

    question = (f"How many objects that resemble a {queried} do you see on the table?")
    gt_answer = str(gt_count)

    # ── V2: free-form integer answer ──────────────────────────────────
    if integer_answer:
        return {
            "question": question,
            "question_type": "count",
            "question_subtype": "integer",
            "gt_answer": gt_answer,           # str(int) — compared as int
            "gt_count": int(gt_count),
            "answer_options": [],             # no MCQ
            "answer_format": "integer",
            "target_name": queried,
            "counting_target_count": gt_count,
            # Asset folder of the counted object — persisted so the review
            # "Referred object guide" renders it even when count==0 (object
            # absent from the scene) or the name was LLM-rephrased.
            "target_ref_folder": (gt_lookup.get(queried, {}) or {}).get("obj_folder"),
            "protected_obj_ids": [],
            "require_target_visible": False,
        }

    NONE_OPTION = "none of the above"
    max_possible = len(scene_obj_names)
    distractors = _tight_numeric_distractors(
        gt=gt_count, n_scene=max_possible, rng=rng, k=3)

    # 5-option format: 4 concrete numerical candidates + NOTA always present.
    # NOTA is always in the option set (so its presence carries no info
    # about whether the GT is concrete vs NOTA), and the 5 options are
    # shuffled together so NOTA's position is uniformly random.
    # 20% NOTA rate (lowered from 30% — too many NOTA GTs skewed results).
    if rng.random() < 0.2:
        # GT is NOTA — need 4 concrete wrong answers.
        extra = _tight_numeric_distractors(
            gt=gt_count, n_scene=max_possible, rng=rng, k=4)
        numeric_opts = [str(v) for v in extra[:4]]
        gt_answer = NONE_OPTION
    else:
        # GT is concrete — 4 concretes = gt + 3 distractors.
        numeric_opts = [str(gt_count)] + [str(d) for d in distractors]
    answer_options = numeric_opts + [NONE_OPTION]
    rng.shuffle(answer_options)

    return {
        "question": question,
        "question_type": "counting",
        "question_subtype": "by_name",
        "gt_answer": gt_answer,
        "answer_options": answer_options,
        "target_name": queried,
        "counting_target_count": gt_count,
        "target_ref_folder": (gt_lookup.get(queried, {}) or {}).get("obj_folder"),
        "protected_obj_ids": [],
        "require_target_visible": False,
    }


def _generate_interactive_counting_question(
    scene_obj_names: List[str],
    rng: random.Random,
) -> dict:
    """Interactive counting: 'Remove x objects then count the remaining.'"""
    n_total = len(scene_obj_names)
    x = rng.randint(1, min(4, n_total - 1))
    gt_remaining = n_total - x

    question = (
        f"Remove exactly {x} object(s) "
        f"from the table, then count how many objects remain on the table."
    )

    NONE_OPTION = "none of the above"
    # Exclude n_total and x from distractors — both appear in the question
    # text, making arithmetic trivial.  See QUESTION_UPDATES.MD §4.2.
    distractors = _tight_numeric_distractors(
        gt=gt_remaining, n_scene=n_total, rng=rng, k=3,
        exclude={n_total, x})

    # 5-option format: 4 concrete numerical candidates + NOTA always present.
    # NOTA is always in the option set so its presence carries no info;
    # all 5 options are shuffled so NOTA's position is uniformly random.
    # 20% NOTA rate (lowered from 30% — too many NOTA GTs skewed results).
    if rng.random() < 0.2:
        # GT is NOTA — 4 concrete wrong answers.
        extra = _tight_numeric_distractors(
            gt=gt_remaining, n_scene=n_total, rng=rng, k=4,
            exclude={n_total, x})
        numeric_opts = [str(v) for v in extra[:4]]
        gt_answer = NONE_OPTION
    else:
        # GT is concrete — 4 concretes = gt + 3 distractors.
        numeric_opts = [str(gt_remaining)] + [str(d) for d in distractors]
        gt_answer = str(gt_remaining)
    answer_options = numeric_opts + [NONE_OPTION]
    rng.shuffle(answer_options)

    return {
        "question": question,
        "question_type": "interactive_counting",
        "question_subtype": "remove_then_count",
        "gt_answer": gt_answer,
        "answer_options": answer_options,
        "target_name": None,
        "n_to_remove": x,
        "gt_remaining": gt_remaining,
        "protected_obj_ids": [],
        "require_target_visible": False,
    }


def _generate_existence_question(
    scene_obj_names: List[str],
    gt_lookup: Dict,
    occ_stats: Dict,
    rng: random.Random,
    designated_targets: List[str] = None,
    qtype_name: str = "existence",
    forced_is_yes: Optional[bool] = None,
    forced_target: Optional[str] = None,
) -> dict:
    """Existence with A/B MCQ and point localisation for 'yes' answers.

    If ``designated_targets`` is provided (from annotations_full), the
    queried object is always drawn from that list — either a designated
    target present in the scene (yes) or one that is absent (no).

    ``qtype_name`` sets the ``question_type`` field — "existence" for V1,
    "find" for the V2 benchmark (identical GT logic; only the 30%
    visibility gate in worker.py differs, and that is keyed on the type
    name).

    V2 find-yes control (``forced_is_yes`` / ``forced_target``): the caller
    pre-buries a specific target and forces a "yes" question about it.  The
    target is intentionally occluded, so it won't appear in the visible-name
    list — ``forced_target`` bypasses name-based selection and queries it
    directly.  When both are ``None`` the V1 behaviour is byte-identical.
    """
    is_yes = forced_is_yes if forced_is_yes is not None else (rng.random() < 0.5)

    if is_yes and forced_target is not None:
        # V2 find-yes: query the pre-buried target directly.
        queried = forced_target
        gt_text = "yes"
        protected = [occ_stats.get("name_to_id", {}).get(queried)]
        protected = [p for p in protected if p is not None]
    elif is_yes:
        occlusion = occ_stats.get("occlusion", {})
        scene_occ = [(n, occlusion.get(occ_stats.get("name_to_id", {}).get(n, -1), 0.0))
                     for n in scene_obj_names]
        scene_occ.sort(key=lambda x: -x[1])

        if designated_targets is not None:
            # Only pick from designated targets that are in the scene
            scene_set = set(scene_obj_names)
            candidates = [(n, occ) for n, occ in scene_occ
                          if n in designated_targets]
            if not candidates:
                # No designated targets in scene — fall back to absent
                is_yes = False
            else:
                queried = rng.choice(candidates)[0]
        else:
            top = scene_occ[:max(1, len(scene_occ) // 2)]
            queried = rng.choice(top)[0]

        if is_yes:
            gt_text = "yes"
            idx = list(scene_obj_names).index(queried) if queried in scene_obj_names else -1
            protected = [occ_stats.get("name_to_id", {}).get(queried)] if idx >= 0 else []
            protected = [p for p in protected if p is not None]

    if not is_yes:
        if designated_targets is not None:
            # Pick an absent designated target
            scene_set = set(scene_obj_names)
            absent = [n for n in designated_targets if n not in scene_set]
        else:
            absent = [n for n in gt_lookup if n not in set(scene_obj_names)]

        if not absent:
            # All targets are present — force a "yes" answer
            if designated_targets is not None:
                present = [n for n in scene_obj_names if n in designated_targets]
            else:
                present = list(scene_obj_names)
            queried = rng.choice(present) if present else rng.choice(scene_obj_names)
            gt_text = "yes"
            protected = [occ_stats.get("name_to_id", {}).get(queried)]
            protected = [p for p in protected if p is not None]
        else:
            queried = rng.choice(absent)
            gt_text = "no"
            protected = []

    yes_no = ["yes", "no"]
    rng.shuffle(yes_no)
    option_map = {"A": yes_no[0], "B": yes_no[1]}
    gt_letter = "A" if option_map["A"] == gt_text else "B"

    return {
        "question": (
            f"Is there a {queried} on the table? "
            f"If yes, also point to it by providing the (row, col) coordinates "
            f"normalised to a [0, 1000] grid."
        ),
        "question_type": qtype_name,
        "question_subtype": "yes" if gt_text == "yes" else "no",
        "gt_answer": gt_letter,
        "gt_answer_text": gt_text,
        "answer_options": [f"A: {option_map['A']}", f"B: {option_map['B']}"],
        "option_map": option_map,
        "target_name": queried if gt_text == "yes" else None,
        "queried_name": queried,  # always set (needed for prompt variants)
        "target_obj_id": occ_stats.get("name_to_id", {}).get(queried) if gt_text == "yes" else None,
        # Asset folder of the queried object — persisted so the human-review
        # "Referred object guide" can render it even when the object is ABSENT
        # from the scene (find=no) or was LLM-renamed (breaks name lookup).
        "queried_ref_folder": (gt_lookup.get(queried, {}) or {}).get("obj_folder"),
        "protected_obj_ids": protected,
        "require_target_visible": False,
    }


def _generate_same_attribute_question(
    target_name: str,
    scene_obj_names: List[str],
    gt_lookup: Dict,
    occ_stats: Dict,
    rng: random.Random,
) -> Optional[dict]:
    """Same-attribute (colour) using anonymous A/B markers."""
    target_color = gt_lookup.get(target_name, {}).get("color", "")
    others = [n for n in scene_obj_names if n != target_name]
    if not others or not target_color:
        return None

    same = [n for n in others if gt_lookup.get(n, {}).get("color") == target_color]
    diff = [n for n in others if gt_lookup.get(n, {}).get("color") != target_color]

    if rng.random() < 0.5 and same:
        comp, gt_text = rng.choice(same), "yes"
    elif diff:
        comp, gt_text = rng.choice(diff), "no"
    elif same:
        comp, gt_text = rng.choice(same), "yes"
    else:
        return None

    yes_no = ["yes", "no"]
    rng.shuffle(yes_no)
    option_map = {"A": yes_no[0], "B": yes_no[1]}
    gt_letter = "A" if option_map["A"] == gt_text else "B"

    obj_a, obj_b = (target_name, comp) if rng.random() < 0.5 else (comp, target_name)
    n2id = occ_stats.get("name_to_id", {})
    fa = occ_stats.get("full_area", {})

    return {
        "question": "Two objects in the scene are marked A and B. Do they have the same color?",
        "question_type": "same_attribute",
        "question_subtype": "same_color" if gt_text == "yes" else "diff_color",
        "gt_answer": gt_letter,
        "gt_answer_text": gt_text,
        "answer_options": [f"A: {option_map['A']}", f"B: {option_map['B']}"],
        "option_map": option_map,
        "target_name": target_name,
        "comparison_object": comp,
        "marker_obj_names": {"A": obj_a, "B": obj_b},
        "marker_obj_ids": {"A": n2id.get(obj_a), "B": n2id.get(obj_b)},
        "marker_full_area": {"A": fa.get(n2id.get(obj_a), 0), "B": fa.get(n2id.get(obj_b), 0)},
        "protected_obj_ids": _get_protected_ids(occ_stats, [target_name, comp]),
        "require_target_visible": True,
    }


def _generate_relative_size_question(
    target_name: str,
    scene_obj_names: List[str],
    gt_lookup: Dict,
    occ_stats: Dict,
    rng: random.Random,
) -> Optional[dict]:
    """Relative size using anonymous A/B markers.

    The VLM must answer two things:
      1. Which object is larger? (letter A or B)
      2. What is the approximate volume ratio of A to B? (numeric)
    """
    ti = gt_lookup.get(target_name, {})
    ts = ti.get("size_cm")
    if not ts:
        return None
    t_vol = ts.get("length", 1) * ts.get("width", 1) * ts.get("height", 1)

    others = [n for n in scene_obj_names if n != target_name]
    rng.shuffle(others)

    for comp in others:
        ci = gt_lookup.get(comp, {})
        cs = ci.get("size_cm")
        if not cs:
            continue
        c_vol = cs.get("length", 1) * cs.get("width", 1) * cs.get("height", 1)
        ratio = max(t_vol, c_vol) / max(min(t_vol, c_vol), 1e-9)
        if ratio < 1.1:
            continue

        obj_a, obj_b = (target_name, comp) if rng.random() < 0.5 else (comp, target_name)
        bigger = target_name if t_vol > c_vol else comp
        gt_letter = "A" if obj_a == bigger else "B"

        # Volume of A and B, and the GT ratio A/B
        vol_a = t_vol if obj_a == target_name else c_vol
        vol_b = c_vol if obj_a == target_name else t_vol
        gt_ratio = round(vol_a / max(vol_b, 1e-9), 2)

        n2id = occ_stats.get("name_to_id", {})
        fa = occ_stats.get("full_area", {})

        return {
            "question": (
                "Two objects in the scene are marked A and B. "
                "(1) Which one is larger? "
                "(2) What is the approximate ratio of the volume of A to the volume of B?"
            ),
            "question_type": "relative_size",
            "question_subtype": "volume_comparison",
            "gt_answer": gt_letter,
            "gt_answer_text": bigger,
            "gt_ratio": gt_ratio,
            "gt_volume_a": round(vol_a, 2),
            "gt_volume_b": round(vol_b, 2),
            "answer_options": ["A", "B"],
            "target_name": target_name,
            "comparison_object": comp,
            "marker_obj_names": {"A": obj_a, "B": obj_b},
            "marker_obj_ids": {"A": n2id.get(obj_a), "B": n2id.get(obj_b)},
            "marker_full_area": {"A": fa.get(n2id.get(obj_a), 0), "B": fa.get(n2id.get(obj_b), 0)},
            "protected_obj_ids": _get_protected_ids(occ_stats, [target_name, comp]),
            "require_target_visible": True,
        }
    return None


def _volume_cc(name: str, gt_lookup: Dict) -> float:
    """Approximate object volume (cm³) from its ``size_cm`` annotation.

    Returns 0.0 when size is unknown.
    """
    sz = (gt_lookup.get(name, {}) or {}).get("size_cm") or {}
    return float(sz.get("length", 0)) * float(sz.get("width", 0)) * float(sz.get("height", 0))


def _generate_under_object_question(
    scene_obj_names: List[str],
    occ_stats: Dict,
    gt_lookup: Dict,
    rng: random.Random,
    cover_hidden_pairs: List[Dict] = None,
    benchmark_v2: bool = False,
) -> Optional[dict]:
    """Under-object: MCQ naming the object hidden beneath a cover.

    V1 (default): 5-option MCQ (4 concrete names + NOTA always present),
        concretes = 2 scene-present + 1 absent + [GT | extra distractor].

    V2 (``benchmark_v2=True``, task name "beneath"): exactly **4** options:
        - "none of the above" (NOTA) — always present
        - 1 object ABSENT from the scene (easy distractor)
        - 1 object IN the scene (the 80/20 slot)
        - 1 object with size ≈ the hidden target (hard distractor; drawn
          preferentially from absent objects so it is never a 2nd valid
          answer)
        80% → the in-scene slot IS the hidden object (it is correct);
        20% → the in-scene slot is a DIFFERENT in-scene object, the true
              target is absent from the options, and NOTA is correct.

    When a cover hides multiple objects, the one with the **largest**
    ``reveal_area`` is chosen as the GT answer.  EVERY other object that
    ``compute_cover_hidden_pairs`` detected as hidden by the same cover
    (i.e. ``reveal_area >= min_reveal_area``, default 200 px) is recorded
    in ``also_valid`` so evaluation can accept it too.  This guarantees a
    VLM is never penalised for naming a different object that genuinely
    is under the cover (e.g. ep_014 in rollout_vlm_20260512_160211 where
    the alphabet block was visually under the planter pot but had only
    1049 px reveal — well below the old 25 %-of-primary cutoff).
    """
    if not cover_hidden_pairs:
        return None

    # ── Group pairs by cover and pick the best (largest reveal) ──
    from collections import defaultdict
    by_cover: dict = defaultdict(list)
    for p in cover_hidden_pairs:
        by_cover[p["cover_name"]].append(p)

    # Pick a random cover (weighted towards covers with large reveal)
    cover_candidates = list(by_cover.keys())
    rng.shuffle(cover_candidates)
    chosen_cover = cover_candidates[0]

    # Among all objects hidden by this cover, pick the one most revealed
    siblings = sorted(by_cover[chosen_cover],
                      key=lambda p: p["reveal_area"], reverse=True)
    pair = siblings[0]  # largest reveal_area

    cover_name = pair["cover_name"]
    hidden_name = pair["hidden_name"]
    cover_id = pair["cover_id"]
    hidden_id = pair["hidden_id"]

    # Track alternative valid answers: every other object that
    # compute_cover_hidden_pairs flagged as hidden by THIS cover.  We
    # used to require reveal_area ≥ 25 % of the primary, but that
    # incorrectly rejected genuine occluders with small visible-area
    # gains (e.g. an object peeking out from a corner of the cover).
    # The detection threshold inside compute_cover_hidden_pairs
    # (default min_reveal_area = 200 px) already filters out edge-
    # bleed artifacts, so any sibling that survives that filter is
    # truly hidden under the cover.
    also_valid = [sib["hidden_name"] for sib in siblings[1:]]

    question = f"What object is under the {cover_name}?"
    gt_answer = hidden_name

    NONE_OPTION = "none of the above"

    # ── V2 "beneath": exactly 4 options with structured distractors ────
    if benchmark_v2:
        scene_set = set(scene_obj_names)
        excluded = {cover_name, hidden_name, *also_valid}

        # Slot: 1 object ABSENT from the scene (easy distractor).
        absent_pool = [n for n in gt_lookup.keys()
                       if n not in scene_set and n not in excluded]
        rng.shuffle(absent_pool)
        absent_distractor = absent_pool[0] if absent_pool else None

        # Slot: 1 object with size ≈ the hidden target (hard distractor).
        # Drawn PREFERENTIALLY from absent objects so it can never be a
        # second valid answer; fall back to any pool object if needed.
        hidden_vol = _volume_cc(hidden_name, gt_lookup)
        size_excluded = set(excluded) | ({absent_distractor} if absent_distractor else set())

        def _size_match(pool):
            cands = []
            for n in pool:
                if n in size_excluded:
                    continue
                v = _volume_cc(n, gt_lookup)
                if v <= 0 or hidden_vol <= 0:
                    continue
                ratio = max(v, hidden_vol) / min(v, hidden_vol)
                if ratio <= 1.4:
                    cands.append(n)
            return cands

        size_cands = _size_match([n for n in gt_lookup.keys() if n not in scene_set])
        if not size_cands:
            size_cands = _size_match(list(gt_lookup.keys()))
        rng.shuffle(size_cands)
        size_distractor = size_cands[0] if size_cands else None

        # Slot: 1 object IN the scene (the 80/20 slot).
        #   80% → it is the hidden object (correct).
        #   20% → a DIFFERENT in-scene object; true target absent → NOTA correct.
        in_scene_pool = [n for n in scene_set
                         if n not in excluded and n != absent_distractor
                         and n != size_distractor]
        rng.shuffle(in_scene_pool)
        if rng.random() < 0.8:
            in_scene_slot = hidden_name
            gt_answer = hidden_name
        else:
            in_scene_slot = in_scene_pool[0] if in_scene_pool else None
            gt_answer = NONE_OPTION

        # Assemble 3 concrete slots + NOTA. Top up if any slot was empty
        # so we always have exactly 4 options.
        concrete = [s for s in (absent_distractor, size_distractor, in_scene_slot) if s]
        if len(concrete) < 3:
            topup_pool = [n for n in list(gt_lookup.keys())
                          if n not in concrete and n not in excluded]
            rng.shuffle(topup_pool)
            for n in topup_pool:
                if len(concrete) >= 3:
                    break
                concrete.append(n)
        # Dedup preserving order, keep 3 concretes, append NOTA.
        seen_v2: set = set()
        concrete = [o for o in concrete if not (o in seen_v2 or seen_v2.add(o))][:3]
        options = concrete + [NONE_OPTION]
        rng.shuffle(options)

        return {
            "question": question,
            "question_type": "beneath",
            "question_subtype": "identity",
            "gt_answer": gt_answer,
            "answer_options": options,
            "target_name": hidden_name,
            "cover_object": cover_name,
            # In the 20% branch the true target is not an option, so
            # also_valid siblings must NOT be accepted (answer is NOTA).
            "also_valid": also_valid if gt_answer == hidden_name else [],
            "protected_obj_ids": [hidden_id],
            "require_target_visible": False,
            "reveal_area": pair["reveal_area"],
        }

    # Build distractors: 2 scene-present + 1 scene-absent.
    # This prevents a VLM from using "which option is NOT visible on
    # the table?" as a shortcut.  See QUESTION_UPDATES.MD §4.3.
    #
    # IMPORTANT: distractors must NOT include any object that is itself
    # hidden under the same cover (``also_valid``) — otherwise we'd
    # have *two* correct answers in the option set (GT and the
    # also-valid distractor), and the VLM would have to guess which
    # one the evaluator considers primary.  By excluding ``also_valid``
    # from both pools we guarantee exactly one correct option among
    # the four concrete choices.
    scene_set = set(scene_obj_names)
    excluded = {cover_name, gt_answer, *also_valid}
    scene_pool = [n for n in set(scene_obj_names) if n not in excluded]
    rng.shuffle(scene_pool)
    scene_distractors = scene_pool[:2]

    absent_pool = [n for n in gt_lookup.keys()
                   if n not in scene_set and n not in excluded]
    rng.shuffle(absent_pool)
    absent_distractors = absent_pool[:1]

    first3 = scene_distractors + absent_distractors
    # Top up from whichever pool has leftovers if either was too small
    if len(first3) < 3:
        leftover = [n for n in (scene_pool + absent_pool)
                    if n not in first3][:3 - len(first3)]
        first3.extend(leftover)
    first3 = first3[:3]

    # 5-option format: 4 concrete name candidates + NOTA always present.
    # The four concrete slots are: 2 scene-present + 1 absent + either
    # the GT (80%) or one more distractor (20% — NOTA is correct).
    # NOTA is always included and all 5 options are shuffled together
    # so its position is uniformly random.
    if gt_answer in first3:
        # Defensive — scene_pool/absent_pool exclude gt_answer by
        # construction, but if we got here anyway add a different 4th.
        remaining = [n for n in (scene_pool + absent_pool)
                     if n not in first3 and n != gt_answer]
        fourth = rng.choice(remaining) if remaining else None
        four_concrete = first3 + ([fourth] if fourth else [])
    else:
        # 20% NOTA rate.  When NOTA is the correct answer the 4th
        # concrete slot is another distractor (NOT the GT).
        if rng.random() < 0.2:
            remaining = [n for n in (scene_pool + absent_pool)
                         if n not in first3 and n != gt_answer]
            fourth = rng.choice(remaining) if remaining else None
            four_concrete = first3 + ([fourth] if fourth else [])
            gt_answer = NONE_OPTION
        else:
            four_concrete = first3 + [gt_answer]

    # Dedup concretes, append NOTA, and shuffle all 5 together.
    seen: set = set()
    four_concrete = [o for o in four_concrete if not (o in seen or seen.add(o))]
    unique = four_concrete + [NONE_OPTION]
    rng.shuffle(unique)

    return {
        "question": question,
        "question_type": "under_object",
        "question_subtype": "identity",
        "gt_answer": gt_answer,
        "answer_options": unique,
        "target_name": hidden_name,
        "cover_object": cover_name,
        "also_valid": also_valid,           # other objects substantially revealed
        "protected_obj_ids": [hidden_id],   # kept for reference (not enforced)
        "require_target_visible": False,
        "reveal_area": pair["reveal_area"],
    }


def _generate_rearrangement_question(
    scene_obj_names: List[str],
    occ_stats: Dict,
    gt_lookup: Dict,
    rng: random.Random,
    pixel_centroids: Dict[int, Tuple[float, float]] = None,
    margin_px: int = 40,
) -> Optional[dict]:
    """Rearrangement (V2 NEW): "Remove all objects to the {dir} of the {ref}."

    Success is conditioned on the FINAL simulator world state (F1 over the
    removed set), not a text answer.  See docs/BENCHMARK_V2_DESIGN.md.

    Direction is defined in the TOP-DOWN PERCEPTION IMAGE (NOT the world
    frame):
        left  : col < ref_col - margin
        right : col > ref_col + margin
        above : row < ref_row - margin   (row 0 = top)
        below : row > ref_row + margin

    ``pixel_centroids`` maps obj_id → (row, col) image centroid, computed
    by the caller from the settled seg buffer.  A dead-band ``margin_px``
    around the reference on the predicate axis must be EMPTY — any object
    inside it makes the scene ambiguous and the function returns None so
    the caller regenerates with a new seed.

    Constraints (else return None → caller retries):
        1. reference object is unique by name (nameable, unambiguous);
        2. GT removal-set size ∈ [1, 4];
        3. no scene object lies inside the dead-band on the predicate axis.
    """
    if not pixel_centroids:
        return None

    id_to_name = occ_stats.get("id_to_name", {})
    name_to_id = occ_stats.get("name_to_id", {})

    # Only consider objects that have a valid centroid AND a known name.
    valid_ids = [oid for oid in pixel_centroids
                 if oid in id_to_name or str(oid) in id_to_name]
    if len(valid_ids) < 3:
        return None

    # Candidate references = objects whose NAME is unique in the scene
    # (so "the {name}" is unambiguous).
    name_counts = Counter(scene_obj_names)
    unique_ref_ids = [
        oid for oid in valid_ids
        if name_counts.get(id_to_name.get(oid, id_to_name.get(str(oid))), 0) == 1
    ]
    rng.shuffle(unique_ref_ids)

    directions = ["left", "right", "above", "below"]

    for ref_id in unique_ref_ids:
        ref_name = id_to_name.get(ref_id, id_to_name.get(str(ref_id)))
        ref_row, ref_col = pixel_centroids[ref_id]

        rng.shuffle(directions)
        for direction in directions:
            removal_ids = []
            in_band = False
            for oid in valid_ids:
                if oid == ref_id:
                    continue
                row, col = pixel_centroids[oid]
                if direction in ("left", "right"):
                    delta = col - ref_col
                else:  # above / below
                    delta = row - ref_row
                # Predicate sign: left/above = negative delta; right/below = positive
                positive_side = direction in ("right", "below")
                signed = delta if positive_side else -delta

                if abs(delta) < margin_px:
                    in_band = True
                    break
                if signed > 0:
                    removal_ids.append(oid)

            if in_band:
                continue  # ambiguous scene for this ref/dir — try next
            if not (1 <= len(removal_ids) <= 4):
                continue  # need 1–4 objects on the predicate side

            removal_names = [id_to_name.get(o, id_to_name.get(str(o)))
                             for o in removal_ids]
            # Natural phrasing: "to the left/right of" vs "above/below".
            if direction in ("left", "right"):
                _phrase = f"to the {direction} of the {ref_name}"
            else:
                _phrase = f"{direction} the {ref_name}"
            question = (f"Remove all objects {_phrase} (as seen in the top-down "
                        f"image). When you are done, call answer_question.")

            return {
                "question": question,
                "question_type": "rearrangement",
                "question_subtype": direction,
                # No text GT answer — scored on final world state.
                "gt_answer": None,
                "answer_options": [],
                "answer_format": "world_state",
                "target_name": ref_name,          # the reference (must NOT be removed)
                "reference_obj_id": int(ref_id),
                "direction": direction,
                "gt_removal_ids": [int(o) for o in removal_ids],
                "gt_removal_names": removal_names,
                "margin_px": int(margin_px),
                # Reference must survive; treat as protected for reporting
                # (NOT enforced — removing it just fails via F1).
                "protected_obj_ids": [int(ref_id)],
                "require_target_visible": False,
            }

    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_question_descriptor(
    question_type: str,
    scene_obj_names: List[str],
    gt_lookup: Dict,
    occ_stats: Dict,
    rng: random.Random,
    cover_hidden_pairs: List[Dict] = None,
    counting_target_name: str = None,
    counting_target_count: int = None,
    designated_targets: List[str] = None,
    benchmark_version: str = "v1",
    pixel_centroids: Dict = None,
    question_phrasing: str = "template",
    phrasing_model: str = None,
    find_forced_is_yes: Optional[bool] = None,
    find_forced_target: Optional[str] = None,
) -> Optional[dict]:
    """Generate a question descriptor for the given type (or 'random').

    Args:
        designated_targets:  Display names of objects with role='target'
            (from annotations_full).  When provided, existence questions
            will only query objects from this list.
        benchmark_version:   "v1" (6-task) or "v2" (4-task: count / find /
            beneath / rearrangement).  V2 dispatches to the same generators
            with V2 flags so V1 output is byte-identical.
        pixel_centroids:     {obj_id: (row, col)} image centroids — required
            by the V2 rearrangement generator (computed by the caller from
            the settled seg buffer).
        question_phrasing:   "template" (default, hand-written variant pool)
            or "llm" (natural phrasing from ``phrasing_model``).  The "llm"
            path only rewrites the question STEM; the deterministic answer-
            format suffix and all ground truth are unchanged.  Applies to
            the V2 tasks count / find / beneath only.
        phrasing_model:      model name for the "llm" phrasing path.
    """
    is_v2 = (benchmark_version == "v2")
    _type_pool = QUESTION_TYPES_V2 if is_v2 else QUESTION_TYPES
    if question_type == "random":
        question_type = rng.choice(_type_pool)

    best_id = occ_stats.get("best_id")
    id_to_name = occ_stats.get("id_to_name", {})
    target_name = (
        occ_stats.get("best_name")
        or id_to_name.get(best_id)
        or id_to_name.get(str(best_id))
    )

    generators = {
        # ── V1 (6-task) ──
        "counting": lambda: _generate_counting_question(
            scene_obj_names, gt_lookup, rng,
            counting_target_name=counting_target_name,
            counting_target_count=counting_target_count),
        "interactive_counting": lambda: _generate_interactive_counting_question(
            scene_obj_names, rng),
        "existence": lambda: _generate_existence_question(
            scene_obj_names, gt_lookup, occ_stats, rng,
            designated_targets=designated_targets),
        "same_attribute": lambda: _generate_same_attribute_question(
            target_name, scene_obj_names, gt_lookup, occ_stats, rng),
        "relative_size": lambda: _generate_relative_size_question(
            target_name, scene_obj_names, gt_lookup, occ_stats, rng),
        "under_object": lambda: _generate_under_object_question(
            scene_obj_names, occ_stats, gt_lookup, rng,
            cover_hidden_pairs=cover_hidden_pairs),
        # ── V2 (4-task) ──
        "count": lambda: _generate_counting_question(
            scene_obj_names, gt_lookup, rng,
            counting_target_name=counting_target_name,
            counting_target_count=counting_target_count,
            integer_answer=True),
        "find": lambda: _generate_existence_question(
            scene_obj_names, gt_lookup, occ_stats, rng,
            designated_targets=designated_targets,
            qtype_name="find",
            forced_is_yes=find_forced_is_yes,
            forced_target=find_forced_target),
        "beneath": lambda: _generate_under_object_question(
            scene_obj_names, occ_stats, gt_lookup, rng,
            cover_hidden_pairs=cover_hidden_pairs,
            benchmark_v2=True),
        "rearrangement": lambda: _generate_rearrangement_question(
            scene_obj_names, occ_stats, gt_lookup, rng,
            pixel_centroids=pixel_centroids),
    }

    gen = generators.get(question_type)
    if gen is None:
        return None
    desc = gen()
    if desc is None:
        # Fall back only within the SAME benchmark version's type pool.
        for fb in [qt for qt in _type_pool if qt != question_type]:
            desc = generators.get(fb, lambda: None)()
            if desc is not None:
                break

    # ── Question phrasing ──────────────────────────────────────────────
    # "llm"      → natural phrasing from an LLM for the V2 tasks
    #              (count / find / beneath).  Falls back to the template
    #              path for any type it does not handle or on API failure.
    # "template" → the hand-written variant pool (default; V1 behaviour is
    #              byte-identical).
    if desc is not None and question_phrasing == "llm" and desc.get(
            "question_type") in ("count", "find", "beneath"):
        desc = _apply_llm_phrasing(desc, gt_lookup, phrasing_model)
    elif desc is not None:
        # Apply a random prompt variant for linguistic diversity.
        # Skip same_attribute and relative_size — those use anonymous A/B
        # markers and the question text intentionally hides object names.
        desc = _apply_random_variant(desc, rng)

    return desc


def _apply_llm_phrasing(descriptor: dict, gt_lookup: Dict,
                        phrasing_model: str) -> dict:
    """Replace the V2 question STEM with a natural LLM phrasing.

    Ground truth and the deterministic answer-format suffix are preserved;
    only the human-readable question text changes.
    """
    try:
        from scene_gen.llm_phrasing import phrase_question
    except ImportError:
        return descriptor

    qtype = descriptor.get("question_type", "")
    slots: dict = {}
    if qtype in ("count", "find"):
        item = descriptor.get("target_name") or descriptor.get("queried_name")
        if not item:
            return descriptor
        slots["item"] = item
        slots["item_details"] = gt_lookup.get(item)
    elif qtype == "beneath":
        cover = descriptor.get("cover_object")
        if not cover:
            return descriptor
        slots["cover"] = cover
        slots["cover_details"] = gt_lookup.get(cover)
    else:
        return descriptor

    result = phrase_question(
        qtype, slots, model_name=phrasing_model, use_llm=True)
    descriptor["question"] = result["question"]
    descriptor["question_stem"] = result["stem"]
    descriptor["phrasing_source"] = result["source"]      # llm|cache|template
    descriptor["template_variant"] = f"llm_{result['source']}"
    descriptor["template_category"] = "llm"
    return descriptor


def _apply_random_variant(descriptor: dict, rng: random.Random) -> dict:
    """Replace the question text with a random prompt variant.

    Picks a non-canonical variant from scene_gen.prompt_variants for
    question types that support it (counting, existence, under_object).
    Same_attribute and relative_size are skipped because they use anonymous
    A/B markers — their question text must not reveal object names.
    Interactive_counting has no variants defined.
    """
    try:
        from scene_gen.prompt_variants import VARIANTS
    except ImportError:
        return descriptor

    qtype = descriptor.get("question_type", "")
    variants = VARIANTS.get(qtype, [])
    if len(variants) <= 1:
        return descriptor

    # Pick a random non-canonical variant
    non_canonical = [v for v in variants if v["cat"] != "canonical"]
    if not non_canonical:
        return descriptor
    chosen = rng.choice(non_canonical)
    template = chosen["template"]

    # Build format kwargs from descriptor fields
    kwargs = {}
    if qtype == "under_object":
        kwargs["cover"] = descriptor.get("cover_object", "")
        if "{COVER}" in template:
            kwargs["COVER"] = kwargs["cover"].upper()
    elif qtype == "existence":
        kwargs["queried"] = descriptor.get("queried_name", "") or descriptor.get("target_name", "") or ""
        if "{QUERIED}" in template:
            kwargs["QUERIED"] = kwargs["queried"].upper()
    elif qtype == "counting":
        # Extract the queried name from the original question
        # Our counting questions use: "How many objects that resemble a {name} ..."
        import re
        m = re.search(r'resemble a (.+?) do you', descriptor.get("question", ""))
        kwargs["queried"] = m.group(1) if m else ""
        if "{QUERIED}" in template:
            kwargs["QUERIED"] = kwargs["queried"].upper()
    elif qtype == "interactive_counting":
        # Extract x from: "Remove exactly {x} object(s) ..."
        import re
        m = re.search(r'Remove exactly (\d+)', descriptor.get("question", ""))
        kwargs["x"] = m.group(1) if m else "1"
        if "{X}" in template:
            kwargs["X"] = kwargs["x"]
    elif qtype in ("same_attribute", "relative_size"):
        # A/B marker questions — templates use no object names
        pass

    try:
        new_question = template.format(**kwargs)
        # For existence, preserve the point instruction suffix
        if qtype == "existence" and "point" not in new_question.lower():
            new_question += (
                " If yes, also point to it by providing the (row, col) "
                "coordinates normalised to a [0, 1000] grid."
            )
        # For relative_size, append the ratio sub-question
        if qtype == "relative_size":
            new_question += (
                " Also estimate the approximate ratio of the volume of A "
                "to the volume of B."
            )
        descriptor["question"] = new_question
        descriptor["template_variant"] = chosen["id"]
        descriptor["template_category"] = chosen["cat"]

        # ── Inversion fix for same_attribute ──
        # The canonical question is "Do they have the same color?" where
        # gt_text = "yes" means same, "no" means different.
        # Inversion variants ("Do they differ?") flip the semantics:
        # now "yes" means different and "no" means same.
        # We must swap the GT answer and option_map accordingly.
        if qtype == "same_attribute" and chosen["cat"] == "inversion":
            om = descriptor.get("option_map", {})
            if om:
                # Flip the yes/no mapping: A↔B swap their meanings
                old_gt = descriptor["gt_answer"]
                descriptor["gt_answer"] = "B" if old_gt == "A" else "A"
                # Also flip gt_answer_text
                old_text = descriptor.get("gt_answer_text", "")
                descriptor["gt_answer_text"] = "no" if old_text == "yes" else "yes"
                # Flip subtype
                old_sub = descriptor.get("question_subtype", "")
                if old_sub == "same_color":
                    descriptor["question_subtype"] = "diff_color"
                elif old_sub == "diff_color":
                    descriptor["question_subtype"] = "same_color"

    except (KeyError, IndexError):
        pass  # keep original question on format failure

    return descriptor
