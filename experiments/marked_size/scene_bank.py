"""Portable frozen-scene packages for the marked three-way benchmark.

A generated scene package is intentionally more durable than a PyBullet
``.bullet`` snapshot.  It stores asset-relative URDF references, settled rigid
body states, dynamic parameters, and the already-baked badge texture PNGs.
That makes it replayable in a fresh simulator process (and therefore by every
ablation arm) without relying on fragile PyBullet body IDs or a local absolute
asset path.

Package layout::

    scenes/<scene_id>/
      scene.json          public physical scene + question input
      runtime.json        evaluator-only badge-to-body texture bindings
      annotation.json     evaluation-only labels/diagnostics
      textures/marker_00.png
      textures/marker_01.png
      textures/marker_02.png

``scene.json`` contains the physical scene and task text, while ``runtime.json``
contains the hidden evaluator bindings needed to paint textures onto fresh body
IDs. ``annotation.json`` contains the answer and other evaluation metadata.
Neither private file is passed to an evaluated agent, so label locations are
never a privileged prompt input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from typing import Any, Dict, Iterable, Tuple

import pybullet as p

from experiments.marked_size import marker_lib as ML

SCHEMA_VERSION = 2

# The table material is a compact CCO/ambientCG directory containing color,
# roughness, normal, and sometimes AO maps. Copying it into the scene package
# makes Blender renders stable even when the evaluator's user cache differs.
TABLE_TEXTURE_DIRNAME = "table_texture"


def _as_list(xs: Iterable) -> list:
    return [float(x) for x in xs]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump(path: str, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_table_texture(texture_dir: str | None, package_dir: str) -> dict:
    """Copy the exact applied Blender table material into a scene package.

    A missing cache texture is a legitimate generation condition: it means the
    renderer used the URDF's plain material. Store that explicitly instead of
    silently sampling a different table during replay.
    """
    if not texture_dir:
        return {"mode": "plain_urdf"}
    texture_dir = os.path.abspath(texture_dir)
    if not os.path.isdir(texture_dir):
        raise FileNotFoundError(f"Applied table texture directory vanished: {texture_dir}")
    dest = os.path.join(package_dir, TABLE_TEXTURE_DIRNAME)
    shutil.copytree(texture_dir, dest)
    files = []
    for root, _, names in os.walk(dest):
        for name in sorted(names):
            path = os.path.join(root, name)
            files.append({"path": os.path.relpath(path, package_dir),
                          "sha256": _sha256(path)})
    return {"mode": "packaged_cc0", "path": TABLE_TEXTURE_DIRNAME,
            "source_basename": os.path.basename(texture_dir), "files": files}


def upgrade_v1_package_with_table_texture(scene_file: str) -> bool:
    """In-place migration from package schema v1 to v2.

    V1 packages predate table-material capture, but they retain the exact
    ``replay_seed`` used by the renderer's deterministic texture picker. While
    the original CCO cache is available, we can therefore copy the *same*
    material into each package without re-running physics or re-baking badges.
    This returns ``True`` when an upgrade was performed and ``False`` when the
    package was already v2.
    """
    scene_file = os.path.abspath(scene_file)
    package_dir = os.path.dirname(scene_file)
    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)
    version = scene.get("schema_version")
    if version == SCHEMA_VERSION:
        return False
    if version != 1:
        raise ValueError(f"Cannot migrate scene schema {version!r}: {scene_file}")
    if "table_texture" in scene:
        raise ValueError(f"Ambiguous v1 package already has table_texture: {scene_file}")
    try:
        from robo_blender.renderer import get_random_table_texture
        source = get_random_table_texture(seed=int(scene["replay_seed"]))
    except Exception as exc:
        raise RuntimeError(f"Could not recover table material for {scene_file}") from exc
    scene["table_texture"] = _copy_table_texture(source, package_dir)
    scene["schema_version"] = SCHEMA_VERSION
    _json_dump(scene_file, scene)
    for name in ("runtime.json", "annotation.json"):
        path = os.path.join(package_dir, name)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("schema_version") != 1:
            raise ValueError(f"Expected v1 {name} beside {scene_file}")
        doc["schema_version"] = SCHEMA_VERSION
        _json_dump(path, doc)
    return True


def _folder_for_object(folder_by_name: Dict[str, str], name: str) -> str:
    folder = folder_by_name.get(name)
    if not folder:
        raise ValueError(
            f"Cannot package object {name!r}: no asset folder mapping. "
            "Scene packages require an explicit name→folder mapping.")
    return str(folder)


def capture_scene_state(demo, folder_by_name: Dict[str, str]) -> Tuple[list, Dict[int, str]]:
    """Capture every remaining dynamic object in stable package order.

    Body IDs are process-local and intentionally never persisted as identifiers.
    ``object_key`` is the stable link between the rigid-body record and the
    marker/answer annotations.
    """
    env = demo.env
    objects = []
    key_by_id = {}
    for index, (obj_id, name) in enumerate(zip(env.obj_ids, env.obj_names)):
        key = f"object_{index:03d}"
        key_by_id[int(obj_id)] = key
        pos, orn = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=env.physicsClient)
        lin_vel, ang_vel = p.getBaseVelocity(obj_id, physicsClientId=env.physicsClient)
        dyn = p.getDynamicsInfo(obj_id, -1, physicsClientId=env.physicsClient)
        objects.append({
            "object_key": key,
            "name": str(name),
            "folder": _folder_for_object(folder_by_name, name),
            # These are base/inertial-frame transforms, exactly the quantities
            # accepted by loadURDF/resetBasePositionAndOrientation.
            "base_position": _as_list(pos),
            "base_orientation": _as_list(orn),
            "linear_velocity": _as_list(lin_vel),
            "angular_velocity": _as_list(ang_vel),
            "dynamics": {
                "mass": float(dyn[0]),
                "lateral_friction": float(dyn[1]),
                "local_inertia_diagonal": _as_list(dyn[2]),
                "restitution": float(dyn[5]),
                "rolling_friction": float(dyn[6]),
                "spinning_friction": float(dyn[7]),
                "contact_damping": float(dyn[8]),
                "contact_stiffness": float(dyn[9]),
            },
        })
    return objects, key_by_id


def _sanitise_descriptor(q_desc: dict, key_by_id: Dict[int, str]) -> dict:
    """Convert process-local body IDs in a descriptor to stable object keys."""
    out = copy.deepcopy(q_desc)
    ids = out.get("marker_obj_ids", {})
    out["marker_object_keys"] = {
        label: key_by_id[int(obj_id)] for label, obj_id in ids.items()
    }
    # IDs are regenerated during replay; retaining stale numbers is a subtle
    # source of incorrect removal bookkeeping, so remove them from disk.
    out.pop("marker_obj_ids", None)
    out.pop("protected_obj_ids", None)
    for site in out.get("marked_sites", []):
        oid = site.pop("obj_id", None)
        if oid is not None:
            site["object_key"] = key_by_id[int(oid)]
        # The source scratch texture is outside the package. The portable
        # texture reference is inserted below once its PNG is copied.
        site.pop("badge_texture", None)
    return out


def write_scene_package(scene_root: str, scene_id: str, demo,
                        folder_by_name: Dict[str, str], q_desc: dict,
                        gt: dict, triple: list, baked_textures: list,
                        generation_seed: int, variant: str = "shape",
                        preview_rgb=None, table_texture_dir: str | None = None) -> dict:
    """Write one self-contained scene package and return its manifest record.

    Call this only *after* the non-target-clutter reveal oracle has passed and
    before the temporary scratch badge textures are deleted.
    """
    scene_root = os.path.abspath(scene_root)
    package_dir = os.path.join(scene_root, scene_id)
    texture_dir = os.path.join(package_dir, "textures")
    os.makedirs(texture_dir, exist_ok=False)

    objects, key_by_id = capture_scene_state(demo, folder_by_name)
    texture_by_label = {}
    copied_textures = []
    for marker_index, (site, source) in enumerate(zip(triple, baked_textures)):
        label = str(site["label"])
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Baked texture for {label} disappeared: {source}")
        # Generic filenames keep the public package from supplying an easy
        # machine-readable label→object lookup. The evaluator-only runtime
        # manifest supplies that binding while replaying a scene.
        rel = os.path.join("textures", f"marker_{marker_index:02d}.png")
        dest = os.path.join(package_dir, rel)
        shutil.copy2(source, dest)
        texture_by_label[label] = rel
        copied_textures.append({"path": rel, "sha256": _sha256(dest)})

    runtime_markers = []
    for site in triple:
        label = str(site["label"])
        runtime_markers.append({
            "label": label,
            "object_key": key_by_id[int(site["obj_id"])],
            "badge_texture": texture_by_label[label],
        })

    table_texture = _copy_table_texture(table_texture_dir, package_dir)

    # The initial image can be regenerated exactly from this state + texture;
    # the question text and badge legend are public task inputs, while the
    # answer-bearing descriptor is placed in annotation.json.
    scene_doc = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": str(scene_id),
        "generation_seed": int(generation_seed),
        # Reusing this seed pins per-episode visual nuisance choices such as
        # the Blender table texture across all baseline arms.
        "replay_seed": int(generation_seed),
        "objects": objects,
        "textures": copied_textures,
        "table_texture": table_texture,
        "question": {
            key: copy.deepcopy(q_desc[key]) for key in (
                "question_type", "question_subtype", "question",
                "badge_description", "answer_options",
            ) if key in q_desc
        },
        "render_contract": {
            "camera": "MG-VQA default top-down camera",
            "state_capture": "post-settle, post-containment-wall-removal",
            "badge_storage": "texture-baked PNG copied into this package",
        },
    }
    if preview_rgb is not None:
        from PIL import Image
        preview_rel = "preview.png"
        Image.fromarray(preview_rgb).save(os.path.join(package_dir, preview_rel))
        scene_doc["preview"] = {"path": preview_rel,
                                "sha256": _sha256(os.path.join(package_dir, preview_rel))}
    runtime = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": str(scene_id),
        "markers": runtime_markers,
    }
    annotation = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": str(scene_id),
        "ground_truth": copy.deepcopy(gt),
        "question_descriptor": _sanitise_descriptor(q_desc, key_by_id),
        "marker_diagnostics": [{
            key: copy.deepcopy(site[key]) for key in (
                "label", "name", "folder", "badge_cm_actual", "badge_frac",
                "badge_frac_revealed", "badge_hidden_t0", "badge_visible_t0",
                "occlusion", "volume_cm3", "painted_texels",
                "badge_paint_frac", "badge_rendered_frac", "world_point",
            ) if key in site
        } | {"object_key": key_by_id[int(site["obj_id"])]}
        for site in triple],
    }
    _json_dump(os.path.join(package_dir, "scene.json"), scene_doc)
    _json_dump(os.path.join(package_dir, "runtime.json"), runtime)
    _json_dump(os.path.join(package_dir, "annotation.json"), annotation)

    return {
        "scene_id": str(scene_id),
        # Stored relative to scene_bank_dir's parent so an individual package
        # record remains understandable, while select_scene_bank rebases it
        # again relative to the final frozen-manifest location.
        "scene_file": os.path.relpath(os.path.join(package_dir, "scene.json"),
                                        os.path.dirname(scene_root)),
        "scene_bank_dir": scene_root,
        "replay_seed": int(generation_seed),
        "schema_version": SCHEMA_VERSION,
    }


def load_scene_package(scene_file: str, require_annotation: bool = True) -> dict:
    """Load and validate a portable package from its public scene JSON."""
    scene_file = os.path.abspath(scene_file)
    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)
    if scene.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported marked scene schema in {scene_file}: "
                         f"{scene.get('schema_version')!r}; regenerate the "
                         "bank so table materials are package-local")
    package_dir = os.path.dirname(scene_file)
    runtime_path = os.path.join(package_dir, "runtime.json")
    if not os.path.isfile(runtime_path):
        raise FileNotFoundError(f"Missing evaluator runtime bindings: {runtime_path}")
    with open(runtime_path, encoding="utf-8") as f:
        runtime = json.load(f)
    if runtime.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Runtime schema mismatch in {runtime_path}")
    for marker in runtime.get("markers", []):
        texture = os.path.join(package_dir, marker["badge_texture"])
        if not os.path.isfile(texture):
            raise FileNotFoundError(f"Missing packaged badge texture: {texture}")
    annotation = None
    annotation_path = os.path.join(package_dir, "annotation.json")
    if os.path.isfile(annotation_path):
        with open(annotation_path, encoding="utf-8") as f:
            annotation = json.load(f)
        if annotation.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Annotation schema mismatch in {annotation_path}")
    elif require_annotation:
        raise FileNotFoundError(
            f"Evaluation needs annotation.json beside {scene_file}. "
            "A public-only package can be rendered but not scored.")
    table = scene.get("table_texture", {})
    if table.get("mode") == "packaged_cc0":
        table_path = os.path.join(package_dir, table.get("path", ""))
        if not os.path.isdir(table_path):
            raise FileNotFoundError(f"Missing packaged table material: {table_path}")
    elif table.get("mode") != "plain_urdf":
        raise ValueError(f"Unknown/missing table_texture contract in {scene_file}")
    return {"scene_file": scene_file, "package_dir": package_dir,
            "scene": scene, "runtime": runtime, "annotation": annotation}


def resolve_bank_entry(entry: dict, manifest_path: str) -> str:
    """Resolve a ``scene_file`` record relative to the frozen bank manifest."""
    path = entry.get("scene_file") or entry.get("scene_package")
    if not path:
        raise ValueError("Scene-bank entry has no scene_file")
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), path)
    return os.path.abspath(path)


def load_scene_bank_manifest(manifest_path: str) -> list:
    """Load a frozen selected-bank manifest and resolve every package path."""
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        doc = json.load(f)
    rows = doc.get("selected", doc.get("scenes", []))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No selected scenes in scene bank: {manifest_path}")
    out = []
    seen = set()
    for index, row in enumerate(rows):
        scene_file = resolve_bank_entry(row, manifest_path)
        pkg = load_scene_package(scene_file, require_annotation=True)
        scene_id = pkg["scene"].get("scene_id")
        if scene_id in seen:
            raise ValueError(f"Duplicate scene_id {scene_id!r} in {manifest_path}")
        seen.add(scene_id)
        out.append({"bank_index": index, "entry": row, "package": pkg})
    return out


def packaged_table_texture_path(package: dict) -> str | None:
    """Return the exact copied table material path, or ``None`` for plain URDF."""
    table = package["scene"]["table_texture"]
    return (os.path.join(package["package_dir"], table["path"])
            if table["mode"] == "packaged_cc0" else None)


def instantiate_scene_package(demo, package: dict, asset_root: str) -> Dict[str, int]:
    """Load a frozen object state into a freshly initialized empty ``demo``.

    ``demo`` must have been constructed with ``cfg.n_objects=0`` and without
    temporary containment walls. We deliberately use ``loadURDF`` plus the
    stored base transforms rather than ``saveBullet/loadBullet``: the latter
    bakes in ephemeral body IDs and absolute asset paths, which makes a release
    archive non-portable and breaks cross-process replay.
    """
    env = demo.env
    if env.obj_ids:
        raise RuntimeError("Scene replay requires an empty RobotEnvUI instance")
    asset_root = os.path.abspath(asset_root)
    key_to_id = {}
    for obj in package["scene"]["objects"]:
        folder = obj["folder"]
        urdf = os.path.join(asset_root, folder, "model.urdf")
        if not os.path.isfile(urdf):
            raise FileNotFoundError(
                f"Frozen scene asset is unavailable: {urdf}. "
                "Install the benchmark asset release at --asset_path.")
        obj_id = p.loadURDF(urdf, obj["base_position"], obj["base_orientation"],
                            physicsClientId=env.physicsClient)
        dyn = obj.get("dynamics", {})
        kwargs = {
            "mass": dyn.get("mass", 0.3),
            "lateralFriction": dyn.get("lateral_friction", 0.8),
            "localInertiaDiagonal": dyn.get("local_inertia_diagonal"),
            "restitution": dyn.get("restitution", 0.01),
            "rollingFriction": dyn.get("rolling_friction", 0.005),
            "spinningFriction": dyn.get("spinning_friction", 0.005),
        }
        # PyBullet rejects a contact-stiffness-only update; set the pair only
        # when the original body actually carried valid positive values.
        stiff, damp = dyn.get("contact_stiffness", -1), dyn.get("contact_damping", -1)
        if stiff is not None and damp is not None and stiff > 0 and damp >= 0:
            kwargs["contactStiffness"] = stiff
            kwargs["contactDamping"] = damp
        p.changeDynamics(obj_id, -1, physicsClientId=env.physicsClient, **kwargs)
        p.resetBaseVelocity(obj_id, obj.get("linear_velocity", [0, 0, 0]),
                            obj.get("angular_velocity", [0, 0, 0]),
                            physicsClientId=env.physicsClient)
        env.obj_ids.append(obj_id)
        env.obj_names.append(obj["name"])
        env.obj_positions.append(tuple(obj["base_position"]))
        env.obj_orientations.append(tuple(obj["base_orientation"]))
        env.obj_grasps.append(None)
        env.obj_grasp_rects.append(None)
        key_to_id[obj["object_key"]] = int(obj_id)
    env.update_obj_states()
    demo.init_obj_state = env.get_obj_states()
    demo.scene_key_to_obj_id = key_to_id
    return key_to_id


def apply_packaged_badges(demo, package: dict) -> Dict[str, int]:
    """Apply stored badge PNGs to fresh PyBullet and Blender body IDs."""
    key_to_id = getattr(demo, "scene_key_to_obj_id", None)
    if not key_to_id:
        raise RuntimeError("Replay scene has no object_key→body-ID mapping")
    out = {}
    for marker in package["runtime"]["markers"]:
        key = marker["object_key"]
        if key not in key_to_id:
            raise KeyError(f"Marker references absent replay object {key!r}")
        obj_id = int(key_to_id[key])
        texture = os.path.join(package["package_dir"], marker["badge_texture"])
        ML.apply_texture(obj_id, texture)
        demo.env.camera.set_object_texture(obj_id, texture)
        out[str(marker["label"])] = obj_id
    return out


def hydrate_marked_descriptor(demo, package: dict) -> dict:
    """Regenerate process-local IDs in the saved marked question descriptor."""
    annotation = package.get("annotation")
    if annotation is None:
        raise ValueError("Cannot hydrate an unannotated public-only scene package")
    q_desc = copy.deepcopy(annotation["question_descriptor"])
    key_to_id = getattr(demo, "scene_key_to_obj_id", None)
    if not key_to_id:
        raise RuntimeError("Replay scene has no object_key→body-ID mapping")
    by_label = q_desc.pop("marker_object_keys", {})
    marker_obj_ids = {label: int(key_to_id[key]) for label, key in by_label.items()}
    q_desc["marker_obj_ids"] = marker_obj_ids
    q_desc["protected_obj_ids"] = list(marker_obj_ids.values())
    for site in q_desc.get("marked_sites", []):
        key = site.pop("object_key", None)
        if key is not None:
            site["obj_id"] = int(key_to_id[key])
        label = site.get("label")
        marker = next((m for m in package["runtime"]["markers"]
                       if m["label"] == label), None)
        if marker is not None:
            site["badge_texture"] = os.path.join(
                package["package_dir"], marker["badge_texture"])
    return q_desc
