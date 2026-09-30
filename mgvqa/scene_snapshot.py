"""Frozen-scene snapshots — save & restore the exact settled tabletop.

Motivation
----------
A ``.blend`` captures the *Blender* scene (meshes + applied textures + camera)
but **loses the PyBullet world**: body IDs, per-object settled pose, dynamics,
and the ability to recompute occlusion / segmentation ground truth.  All of the
benchmark's GT (counting-buried, beneath cover-fraction, find visibility) is
derived from PyBullet, so a ``.blend`` alone cannot restore an authoritative
scene.

Re-spawning from the seed is also fragile: spawning is a drop-and-settle
physics process whose final poses are sensitive to step counts and float
determinism (and therefore to PyBullet / torch versions and the machine).

The robust fix is a **JSON snapshot of the settled scene** plus a copy of the
table-texture directory.  Restoration is a pure *teleport*: load each URDF and
immediately ``resetBasePositionAndOrientation`` to its saved settled pose — no
drop, no settle loop — so the restored scene is byte-identical regardless of
physics reproducibility.

Snapshot layout (per episode)::

    <rollout_root>/_texture_store/<hash>/    # de-duplicated PBR texture dirs
    <ep_dir>/scene_snapshot/
        scene_snapshot.json      # object folders + settled poses + dynamics + meta

Table textures are content-hashed and stored ONCE per unique texture in a
shared ``_texture_store/`` at the rollout root; each episode's snapshot merely
references the hash.  (300 seeded episodes use only ~93 distinct textures, so a
per-episode copy would waste ~2.8 GB; the shared store cuts this to <100 MB.)
The legacy per-episode ``table_texture/`` copy is still understood on restore
for backward compatibility.

Schema (``scene_snapshot.json``) — aligned with the SCENE_BANK_FORMAT.md
``scene.json`` contract (documented field names) plus a few internal extras::

    {
      "schema_version": 1,
      "scene_id": "ep013_seed1013000",
      "episode": 13,
      "episode_seed": 1013000,
      "generation_seed": 1013000,
      "replay_seed": 1013000,
      "question_type": "count",
      "benchmark_version": "v2",
      "asset_path": "owg_robot/assets",
      "img_size": 1024,
      "table_texture": {                      # DOCUMENTED contract block
        "source_name": "WoodFloor036_2K-JPG",
        "store_rel": "../_texture_store/<hash>",  # ep-dir-relative (shared store)
        "hash": "a1b2c3d4e5f6a7b8",
        "texture_seed": 1013000,
        "files": [                            # PBR maps + per-file sha256
          {"path": "../_texture_store/<hash>/..._Color.jpg", "sha256": "..."},
          ...
        ]
      },
      "table": { ... },                       # legacy block (back-compat)
      "question": {                           # public, no GT / no answer
        "question_type": "count",
        "question": "...",
        "answer_options": [...]
      },
      "objects": [
        {
          "object_key": "object_000",          # stable id (documented)
          "spawn_index": 0,                     # internal extra
          "obj_id": 11,                         # PyBullet body id at save (extra)
          "name": "blue and grey toy food can",
          "folder": "dtc_objects/dtc_can_toy_...",   # -> <asset>/<folder>/model.urdf
          "base_position": [x, y, z],           # SETTLED base pose (world)
          "base_orientation": [qx, qy, qz, qw],
          "linear_velocity": [vx, vy, vz],
          "angular_velocity": [wx, wy, wz],
          "dynamics": {                         # nested (documented)
            "mass": 0.3, "lateral_friction": 0.8,
            "rolling_friction": 0.005, "spinning_friction": 0.005,
            "restitution": 0.01, "contact_damping": -1, "contact_stiffness": -1,
            "local_inertia_diagonal": [ixx, iyy, izz]
          }
        }, ...
      ]
    }

See ``save_scene_snapshot`` (called from ``core/worker.py`` right after the
scene settles, before the first VLM call) and ``restore_scene_into_env``
(teleport-restore into an already-built ``RobotEnvUI``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Optional

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_DIRNAME = "scene_snapshot"
SNAPSHOT_JSON = "scene_snapshot.json"
SNAPSHOT_TEXTURE_DIRNAME = "table_texture"       # legacy per-episode copy
TEXTURE_STORE_DIRNAME = "_texture_store"         # shared, content-hashed


# ---------------------------------------------------------------------------
# Question-descriptor persistence + obj-id remapping (for --from_snapshot eval)
# ---------------------------------------------------------------------------
#
# The full question descriptor (``q_desc``) is the ground truth the eval loop
# scores against.  A restored scene loads fresh URDFs → fresh PyBullet body
# IDs, so every id-keyed field in the descriptor must be translated from the
# *original* (generation-time) id to the *restored* id via the ``id_remap``
# that ``restore_scene_into_env`` returns.
#
# The descriptor also holds a few non-JSON-serialisable leaves (e.g. the
# trimesh ``mesh`` object stashed in ``marked_sites``), which are dropped
# before persisting.

# Descriptor keys whose VALUE is a single obj-id (int).
_DESC_SCALAR_ID_KEYS = ("target_obj_id", "reference_obj_id", "best_id")
# Descriptor keys whose VALUE is a list of obj-ids.
_DESC_LIST_ID_KEYS = ("protected_obj_ids", "gt_removal_ids", "target_obj_ids")
# Descriptor keys whose VALUE is a {label: obj-id} dict.
_DESC_DICT_ID_KEYS = ("marker_obj_ids",)
# Keys inside each ``marked_sites[]`` record that are non-serialisable and must
# be stripped before saving the annotation.
_MARKED_SITE_DROP_KEYS = ("mesh",)


def _json_safe(value):
    """Recursively convert a value into a JSON-serialisable form.

    numpy scalars/arrays → python; sets/tuples → lists; objects without a
    natural representation (e.g. trimesh meshes) → dropped (returns a
    sentinel ``_DROP`` the caller filters out).  Used to sanitise ``q_desc``
    before it is embedded in the snapshot annotation.
    """
    import numpy as _np
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, _np.generic):
        return value.item()
    if isinstance(value, _np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            sv = _json_safe(v)
            if sv is _DROP:
                continue
            # dict keys must be str/int/float/bool/None for json
            if not isinstance(k, (str, int, float, bool, type(None))):
                k = str(k)
            out[k] = sv
        return out
    if isinstance(value, (list, tuple, set)):
        return [sv for sv in (_json_safe(v) for v in value) if sv is not _DROP]
    # Unknown leaf (trimesh mesh, function, custom class, ...) — drop it.
    return _DROP


class _Drop:
    __slots__ = ()
_DROP = _Drop()


def sanitize_descriptor_for_snapshot(q_desc: dict) -> dict:
    """Return a JSON-safe deep copy of ``q_desc`` fit for the snapshot annotation.

    Strips non-serialisable leaves (trimesh meshes in ``marked_sites`` etc.)
    while keeping every id-keyed field so eval can re-hydrate + score.
    """
    if not q_desc:
        return {}
    # Pre-strip known heavy leaves inside marked_sites, then json-sanitise all.
    q = dict(q_desc)
    if isinstance(q.get("marked_sites"), (list, tuple)):
        clean_sites = []
        for site in q["marked_sites"]:
            if isinstance(site, dict):
                clean_sites.append({k: v for k, v in site.items()
                                    if k not in _MARKED_SITE_DROP_KEYS})
            else:
                clean_sites.append(site)
        q["marked_sites"] = clean_sites
    safe = _json_safe(q)
    return {} if safe is _DROP else safe


def remap_descriptor_ids(q_desc: dict, id_remap: dict) -> dict:
    """Translate every obj-id-keyed field in a descriptor via ``id_remap``.

    ``id_remap`` maps original (generation-time) PyBullet body id → restored
    body id.  Fields absent from the map are left unchanged (defensive: a
    dropped/duplicate object would otherwise crash scoring).  Returns a NEW
    descriptor dict (the input is not mutated).
    """
    if not q_desc:
        return q_desc
    import copy as _copy
    q = _copy.deepcopy(q_desc)

    def _rid(v):
        try:
            return id_remap.get(int(v), v)
        except (TypeError, ValueError):
            return v

    for k in _DESC_SCALAR_ID_KEYS:
        if q.get(k) is not None:
            q[k] = _rid(q[k])
    for k in _DESC_LIST_ID_KEYS:
        if isinstance(q.get(k), list):
            q[k] = [_rid(v) for v in q[k]]
    for k in _DESC_DICT_ID_KEYS:
        if isinstance(q.get(k), dict):
            q[k] = {lab: (_rid(v) if v is not None else v)
                    for lab, v in q[k].items()}
    # name_to_id: {name: obj_id} — remap the values.
    if isinstance(q.get("name_to_id"), dict):
        q["name_to_id"] = {nm: _rid(v) for nm, v in q["name_to_id"].items()}
    # marked_sites[].obj_id
    if isinstance(q.get("marked_sites"), list):
        for site in q["marked_sites"]:
            if isinstance(site, dict) and site.get("obj_id") is not None:
                site["obj_id"] = _rid(site["obj_id"])
    return q


def _hash_texture_dir(tex_dir: str) -> str:
    """Content hash of a texture directory (stable across machines).

    Hashes the sorted (relative-path, size, sha1-of-bytes) of every file so the
    same texture set always maps to the same hash regardless of absolute path
    or filesystem ordering.  Returns a 16-hex-char digest.
    """
    h = hashlib.sha1()
    files = []
    for root, _dirs, names in os.walk(tex_dir):
        for n in sorted(names):
            fp = os.path.join(root, n)
            rel = os.path.relpath(fp, tex_dir)
            files.append((rel, fp))
    for rel, fp in sorted(files):
        try:
            with open(fp, "rb") as f:
                data = f.read()
        except OSError:
            continue
        h.update(rel.encode("utf-8"))
        h.update(str(len(data)).encode("utf-8"))
        h.update(hashlib.sha1(data).digest())
    return h.hexdigest()[:16]


def _sha256_file(path: str) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rollout_root_of(ep_dir: str) -> str:
    """Return the rollout root that should hold the shared texture store.

    Layouts:
      parallel:  <root>/worker_N/ep_NNN_*/   → root is 2 levels up
      single:    <root>/ep_NNN_*/            → root is 1 level up
    Detect the parallel case by a ``worker_<n>`` parent directory name.
    """
    ep_dir = os.path.abspath(ep_dir)
    parent = os.path.dirname(ep_dir)              # worker_N  OR  rollout root
    base = os.path.basename(parent)
    if base.startswith("worker_") and base[len("worker_"):].isdigit():
        return os.path.dirname(parent)            # up past worker_N
    return parent


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_scene_snapshot(
    demo,
    ep_dir: str,
    *,
    episode: int,
    episode_seed: int,
    question_type: str,
    benchmark_version: str = "v1",
    asset_path: str = "owg_robot/assets",
    copy_texture: bool = True,
    question_descriptor: Optional[dict] = None,
    verbose: bool = True,
) -> Optional[str]:
    """Serialize the CURRENT settled scene to ``<ep_dir>/scene_snapshot/``.

    Captures, for every spawned object, the settled base pose + orientation
    and the dynamics parameters, plus the object's asset folder so the URDF
    can be re-loaded.  Optionally copies the applied table-texture directory
    into the snapshot so restoration does not depend on the seed or the
    texture cache.

    Must be called AFTER spawn + settle (walls removed, poses at rest) and
    BEFORE any manipulation, so the snapshot is the frozen initial scene.

    Returns the snapshot directory path, or None on failure.
    """
    try:
        import pybullet as p
    except ImportError:
        if verbose:
            print("[snapshot] pybullet unavailable — cannot save snapshot")
        return None

    env = demo.env

    # Map spawn folders by name.  ``selected_obj_folders`` is parallel to
    # ``selected_obj_names`` (the spawn order), which equals env.obj_names
    # order at spawn time.  But objects may have been removed before the
    # snapshot (should not happen pre-VLM, but be defensive), so we align by
    # position in the ORIGINAL spawn lists and fall back to name lookup.
    spawn_names = list(getattr(demo, "selected_obj_names", None)
                       or env.obj_names)
    spawn_folders = list(getattr(demo, "selected_obj_folders", None)
                         or spawn_names)
    # name -> first folder (for fallback when ordering diverges)
    name_to_folder = {}
    for nm, fol in zip(spawn_names, spawn_folders):
        name_to_folder.setdefault(nm, fol)

    objects = []
    for spawn_index, (oid, oname) in enumerate(zip(env.obj_ids, env.obj_names)):
        try:
            pos, orn = p.getBasePositionAndOrientation(oid)
        except Exception:
            pos = orn = None
        try:
            lin_vel, ang_vel = p.getBaseVelocity(oid)
        except Exception:
            lin_vel = ang_vel = None
        # Prefer the parallel folder list (exact), fall back to name map.
        if spawn_index < len(spawn_folders) and \
                spawn_index < len(spawn_names) and \
                spawn_names[spawn_index] == oname:
            folder = spawn_folders[spawn_index]
        else:
            folder = name_to_folder.get(oname, oname)

        # Documented scene.json field names (base_position / base_orientation /
        # object_key / dynamics{}), PLUS our internal obj_id + spawn_index
        # (harmless extra fields used by teleport-restore / debugging).
        entry = {
            "object_key": f"object_{spawn_index:03d}",
            "spawn_index": spawn_index,
            "obj_id": int(oid),
            "name": str(oname),
            "folder": str(folder),
            "base_position": list(pos) if pos is not None else None,
            "base_orientation": list(orn) if orn is not None else None,
            "linear_velocity": list(lin_vel) if lin_vel is not None else None,
            "angular_velocity": list(ang_vel) if ang_vel is not None else None,
        }
        # Dynamics — nested block, mirroring what load_obj/changeDynamics
        # applied so a restored object behaves identically if physics IS
        # later stepped.
        try:
            dyn = p.getDynamicsInfo(oid, -1)
            entry["dynamics"] = {
                "mass": float(dyn[0]),
                "lateral_friction": float(dyn[1]),
                "local_inertia_diagonal": list(dyn[2]),
                "restitution": float(dyn[5]),
                "rolling_friction": float(dyn[6]),
                "spinning_friction": float(dyn[7]),
                "contact_damping": float(dyn[8]),
                "contact_stiffness": float(dyn[9]),
            }
        except Exception:
            pass
        objects.append(entry)

    snap_dir = os.path.join(ep_dir, SNAPSHOT_DIRNAME)
    os.makedirs(snap_dir, exist_ok=True)

    # ── Table texture (content-hashed shared store) ───────────────────
    # Store each unique texture ONCE in <rollout_root>/_texture_store/<hash>/
    # and reference it by hash from every episode that uses it.  Avoids the
    # ~2.8 GB of redundant per-episode copies (300 eps → ~93 unique textures).
    tex_seed = getattr(env.camera, "_table_texture_seed", None)
    tex_source = _resolve_applied_texture_dir(env)
    tex_dir_name = os.path.basename(tex_source) if tex_source else None
    tex_hash = None
    tex_store_rel = None
    tex_files = []          # [{path (ep-dir-relative), sha256}, ...]
    store_dst = None
    if copy_texture and tex_source and os.path.isdir(tex_source):
        try:
            tex_hash = _hash_texture_dir(tex_source)
            store_root = os.path.join(_rollout_root_of(ep_dir),
                                      TEXTURE_STORE_DIRNAME)
            store_dst = os.path.join(store_root, tex_hash)
            # Copy only if this hash isn't already stored (idempotent, and
            # safe across parallel workers via a temp-dir + atomic rename).
            if not os.path.isdir(store_dst):
                os.makedirs(store_root, exist_ok=True)
                tmp_dst = os.path.join(
                    store_root, f".tmp_{tex_hash}_{os.getpid()}")
                if os.path.isdir(tmp_dst):
                    shutil.rmtree(tmp_dst)
                shutil.copytree(tex_source, tmp_dst)
                try:
                    os.rename(tmp_dst, store_dst)  # atomic if not present
                except OSError:
                    # Another worker won the race — discard our temp copy.
                    shutil.rmtree(tmp_dst, ignore_errors=True)
            # Relative path from the EPISODE dir to the shared store entry.
            tex_store_rel = os.path.relpath(store_dst, ep_dir)
            # Per-map SHA-256 (documented scene.json contract).  Paths are
            # ep-dir-relative into the shared store (shared-store variant of
            # the format, see SCENE_BANK_FORMAT.md §4.2b).  Only the PBR maps
            # the renderer binds by suffix are listed (skip tags.txt etc.).
            _MAP_SUFFIXES = ("_Color", "_Diffuse", "_Roughness", "_Normal",
                             "_norm", "_AmbientOcclusion", "_Displacement")
            for fn in sorted(os.listdir(store_dst)):
                stem = os.path.splitext(fn)[0]
                if not any(stem.endswith(s) for s in _MAP_SUFFIXES):
                    continue
                fp = os.path.join(store_dst, fn)
                if os.path.isfile(fp):
                    tex_files.append({
                        "path": os.path.join(tex_store_rel, fn),
                        "sha256": _sha256_file(fp),
                    })
            if verbose:
                print(f"[snapshot] table texture '{tex_dir_name}' "
                      f"→ store hash {tex_hash} ({len(tex_files)} maps)")
        except Exception as e:
            if verbose:
                print(f"[snapshot] WARNING: texture store failed: {e}")

    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "scene_id": f"ep{int(episode):03d}_seed{int(episode_seed)}",
        "episode": int(episode),
        "episode_seed": int(episode_seed),
        "generation_seed": int(episode_seed),
        "replay_seed": int(episode_seed),
        "question_type": str(question_type),
        "benchmark_version": str(benchmark_version),
        "asset_path": str(asset_path),
        "img_size": int(getattr(env.camera, "width", 0)) or None,
        "n_objects": len(objects),
        # Documented scene.json contract: top-level table_texture{} block that
        # binds the table surface to the copied PBR maps (shared-store variant).
        "table_texture": {
            "source_name": tex_dir_name,
            "store_rel": tex_store_rel,
            "hash": tex_hash,
            "texture_seed": (int(tex_seed) if tex_seed is not None else None),
            "files": tex_files,
        },
        # Legacy internal block (kept so older restore code / snapshots that
        # read snapshot['table'] still work).
        "table": {
            "texture_seed": (int(tex_seed) if tex_seed is not None else None),
            "texture_dir_name": tex_dir_name,
            "texture_hash": tex_hash,
            "texture_store_rel": tex_store_rel,
            "texture_copied": None,  # legacy per-ep copy (unused now)
            "texture_source": tex_source,
        },
        "objects": objects,
    }

    # Documented scene.json contract: public question block (no GT / no answer).
    if question_descriptor:
        qd = question_descriptor
        opts = qd.get("answer_options")
        snapshot["question"] = {
            "question_type": str(question_type),
            "question": qd.get("question", ""),
            "answer_options": list(opts) if opts else None,
        }

        # ── Evaluator-private annotation (GT + full descriptor) ────────
        # This is what a ``--from_snapshot`` eval run scores against.  The
        # descriptor's id-keyed fields reference the ORIGINAL (generation-time)
        # PyBullet body ids; the snapshot's ``objects[].obj_id`` use the same
        # ids, so on restore ``id_remap`` translates both consistently.
        # Kept SEPARATE from the public ``question`` block so a blind release
        # can strip ``annotation`` (mirrors Mika's scene_bank annotation.json).
        try:
            snapshot["annotation"] = {
                "schema_version": 1,
                "gt_answer": qd.get("gt_answer"),
                "gt_answer_text": qd.get("gt_answer_text"),
                "question_descriptor": sanitize_descriptor_for_snapshot(qd),
            }
        except Exception as _ae:
            if verbose:
                print(f"[snapshot] WARNING: annotation persist failed: {_ae}")

    # ── compare / marked_comparison: persist baked badge textures ──────
    # The 3 A/B/C badges are baked into scratch texture PNGs and applied to
    # the object bodies at generation time.  A teleport-restore reloads the
    # URDFs with their ORIGINAL textures, so we must copy the baked badge PNGs
    # into the snapshot dir and record (obj_id, label, badge_png) so
    # restore_scene_into_env can re-apply them.  Without this the frozen scene
    # would replay with unbadged objects.
    if (question_descriptor
            and question_descriptor.get("question_type") == "marked_comparison"):
        try:
            badge_dir = os.path.join(snap_dir, "badges")
            os.makedirs(badge_dir, exist_ok=True)
            badges = []
            for site in (question_descriptor.get("marked_sites") or []):
                src_png = site.get("badge_texture")
                if not src_png or not os.path.isfile(src_png):
                    continue
                dst_name = f"badge_{site.get('label')}_{os.path.basename(src_png)}"
                dst_png = os.path.join(badge_dir, dst_name)
                shutil.copy2(src_png, dst_png)
                badges.append({
                    "label": site.get("label"),
                    "obj_id": int(site["obj_id"]) if site.get("obj_id") is not None else None,
                    "name": site.get("name"),
                    "folder": site.get("folder"),
                    # ep-dir-relative path so the snapshot is portable
                    "badge_texture": os.path.relpath(dst_png, ep_dir),
                    "badge_texture_sha256": _sha256_file(dst_png),
                })
            if badges:
                snapshot["badges"] = badges
                if verbose:
                    print(f"[snapshot] persisted {len(badges)} badge textures "
                          f"→ {os.path.relpath(badge_dir, ep_dir)}")
        except Exception as _be:
            if verbose:
                print(f"[snapshot] WARNING: badge persist failed: {_be}")

    out_path = os.path.join(snap_dir, SNAPSHOT_JSON)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    if verbose:
        print(f"[snapshot] wrote {out_path} "
              f"({len(objects)} objects, texture="
              f"{'store:'+tex_hash if tex_hash else 'seed-only'})")
    return snap_dir


def _resolve_applied_texture_dir(env) -> Optional[str]:
    """Best-effort recovery of the table-texture directory currently applied.

    Prefers the renderer's remembered ``_last_table_texture``; falls back to
    recomputing from the seed via ``get_random_table_texture``.
    """
    renderer = getattr(env.camera, "_blender_renderer", None)
    tex = getattr(renderer, "_last_table_texture", None) if renderer else None
    if tex and os.path.isdir(tex):
        return tex
    # Fall back to the seed-based pick (deterministic).
    seed = getattr(env.camera, "_table_texture_seed", None)
    try:
        from robo_blender.renderer import get_random_table_texture
        cand = get_random_table_texture(seed=seed)
        if cand and os.path.isdir(cand):
            return cand
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_snapshot_json(snapshot_path: str) -> dict:
    """Load a ``scene_snapshot.json`` given the file OR its containing dir."""
    if os.path.isdir(snapshot_path):
        snapshot_path = os.path.join(snapshot_path, SNAPSHOT_JSON)
    with open(snapshot_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_snapshot_episodes(from_dir: str) -> list:
    """Discover restorable episodes under a rollout-shaped ``--from_snapshot`` dir.

    Accepts either layout (matching how rollouts are written):
      single-process:  <from_dir>/ep_NNN_<type>/scene_snapshot/scene_snapshot.json
      parallel:        <from_dir>/worker_N/ep_NNN_<type>/scene_snapshot/...
      finalized bank:  <from_dir>/ep_NNN_<type>/scene_snapshot/...   (renumbered)

    Episodes are DEDUPED by directory basename (``ep_NNN_<type>``) so the
    root-level symlinks that a parallel rollout creates don't double-count.
    Returns a list of dicts sorted by basename::

        [{"ep_dirname": "ep_000_beneath",
          "ep_dir": "<abs path to episode dir>",
          "snapshot_dir": "<abs path>/scene_snapshot",
          "question_type": "beneath"}, ...]

    A ``bank_manifest.json`` at the root, if present, pins the order/subset.
    """
    import glob as _glob

    from_dir = os.path.abspath(from_dir)

    # 1) If a bank manifest exists, honour its explicit episode list/order.
    manifest_path = os.path.join(from_dir, "bank_manifest.json")
    manifest = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = None

    def _snap_of(ep_dir):
        sd = os.path.join(ep_dir, SNAPSHOT_DIRNAME)
        return sd if os.path.isfile(os.path.join(sd, SNAPSHOT_JSON)) else None

    def _qtype_of(ep_dirname, snapshot_dir):
        # Prefer the snapshot's recorded type; fall back to the dir suffix.
        try:
            snap = load_snapshot_json(snapshot_dir)
            qt = snap.get("question_type")
            if qt:
                return qt
        except Exception:
            pass
        parts = ep_dirname.split("_", 2)
        return parts[2] if len(parts) >= 3 else (parts[-1] if parts else "")

    found = {}   # ep_dirname -> record (dedupe key)

    if manifest and isinstance(manifest.get("episodes"), list):
        for row in manifest["episodes"]:
            rel = row.get("ep_dir") or row.get("dir")
            if not rel:
                continue
            ep_dir = os.path.normpath(os.path.join(from_dir, rel))
            sd = _snap_of(ep_dir)
            if sd is None:
                continue
            name = os.path.basename(ep_dir)
            found[name] = {
                "ep_dirname": name, "ep_dir": ep_dir, "snapshot_dir": sd,
                "question_type": row.get("question_type")
                or _qtype_of(name, sd),
            }
        # Manifests are authoritative — return in listed order.
        return [found[os.path.basename(
                    os.path.normpath(os.path.join(
                        from_dir, r.get("ep_dir") or r.get("dir") or "")))]
                for r in manifest["episodes"]
                if os.path.basename(os.path.normpath(os.path.join(
                    from_dir, r.get("ep_dir") or r.get("dir") or ""))) in found]

    # 2) No manifest — scan root then worker_* dirs.
    search_globs = [os.path.join(from_dir, "ep_*")]
    search_globs += [os.path.join(from_dir, "worker_*", "ep_*")]
    for pat in search_globs:
        for ep_dir in _glob.glob(pat):
            if not os.path.isdir(ep_dir):
                continue
            name = os.path.basename(ep_dir)
            if name in found:
                continue  # first hit wins (root symlink or worker copy)
            sd = _snap_of(ep_dir)
            if sd is None:
                continue
            found[name] = {
                "ep_dirname": name,
                "ep_dir": os.path.abspath(ep_dir),
                "snapshot_dir": sd,
                "question_type": _qtype_of(name, sd),
            }
    return [found[k] for k in sorted(found)]


def restore_scene_into_env(
    demo,
    snapshot: dict,
    *,
    snapshot_dir: Optional[str] = None,
    asset_path: Optional[str] = None,
    settle_steps: int = 5,
    verbose: bool = True,
) -> dict:
    """Teleport-restore a saved scene into an already-built ``RobotEnvUI``.

    The env must already have the table / robot / tray loaded but NO spawned
    objects (any present are cleared first).  For each object in the snapshot
    we ``loadURDF`` then immediately ``resetBasePositionAndOrientation`` to the
    saved settled pose — no drop, no settle loop — so poses are exact.

    A few no-force settle steps run at the end only so contacts register; the
    objects are already at rest so they should not move.

    Args:
        demo:         RobotEnvUI whose env will receive the objects.
        snapshot:     Parsed scene_snapshot.json dict.
        snapshot_dir: Dir containing the snapshot (for the copied texture).
        asset_path:   Override the asset root; defaults to snapshot['asset_path'].
        settle_steps: No-force sim steps after teleport (0 = none).

    Returns a dict with ``restored_obj_ids`` (parallel to snapshot objects)
    and ``id_remap`` (snapshot obj_id -> restored obj_id).
    """
    import pybullet as p
    import numpy as np  # noqa: F401  (kept for parity / future use)

    env = demo.env
    asset_root = asset_path or snapshot.get("asset_path") or demo.asset_path

    # Clear any pre-existing spawned objects.
    if getattr(env, "obj_ids", None):
        for oid in list(env.obj_ids):
            try:
                p.removeBody(oid)
            except Exception:
                pass
        env.obj_ids.clear()
        env.obj_names.clear()
        for lst in ("obj_positions", "obj_orientations",
                    "obj_grasps", "obj_grasp_rects"):
            if hasattr(env, lst):
                getattr(env, lst).clear()

    restored_ids = []
    id_remap = {}
    for obj in snapshot.get("objects", []):
        folder = obj["folder"]
        urdf = os.path.join(asset_root, folder, "model.urdf")
        if not os.path.isfile(urdf):
            if verbose:
                print(f"[snapshot] WARNING: URDF not found: {urdf} — skipping "
                      f"'{obj.get('name')}'")
            restored_ids.append(None)
            continue

        # Accept both documented (base_position/base_orientation) and legacy
        # (position/orientation_xyzw) field names.
        pos = obj.get("base_position", obj.get("position"))
        orn = obj.get("base_orientation", obj.get("orientation_xyzw"))
        oid = p.loadURDF(urdf, pos, orn)
        # Teleport again explicitly (loadURDF may re-centre inertially).
        p.resetBasePositionAndOrientation(oid, pos, orn)
        p.resetBaseVelocity(oid, [0, 0, 0], [0, 0, 0])

        # Re-apply saved dynamics so behaviour matches if physics is stepped.
        # Dynamics may be a nested block (documented) or flat (legacy).
        dyn = obj.get("dynamics", obj)
        kw = {}
        if "mass" in dyn:
            kw["mass"] = dyn["mass"]
        if "lateral_friction" in dyn:
            kw["lateralFriction"] = dyn["lateral_friction"]
        if "rolling_friction" in dyn:
            kw["rollingFriction"] = dyn["rolling_friction"]
        if "spinning_friction" in dyn:
            kw["spinningFriction"] = dyn["spinning_friction"]
        if "restitution" in dyn:
            kw["restitution"] = dyn["restitution"]
        if dyn.get("local_inertia_diagonal"):
            kw["localInertiaDiagonal"] = dyn["local_inertia_diagonal"]
        if kw:
            try:
                p.changeDynamics(oid, -1, **kw)
            except Exception:
                pass

        env.obj_ids.append(oid)
        env.obj_names.append(obj["name"])
        if hasattr(env, "obj_positions"):
            env.obj_positions.append(list(pos))
        if hasattr(env, "obj_orientations"):
            env.obj_orientations.append(list(orn))
        if hasattr(env, "obj_grasps"):
            env.obj_grasps.append(None)
        if hasattr(env, "obj_grasp_rects"):
            env.obj_grasp_rects.append(None)

        restored_ids.append(oid)
        if obj.get("obj_id") is not None:
            id_remap[int(obj["obj_id"])] = int(oid)

    # Contacts-only settle (objects already at rest).
    for _ in range(max(0, int(settle_steps))):
        p.stepSimulation()
    if hasattr(env, "update_obj_states"):
        try:
            env.update_obj_states()
        except Exception:
            pass

    # ── compare / marked_comparison: re-apply baked badge textures ─────
    # A teleport-restore reloads the URDFs with their ORIGINAL textures, so
    # re-apply the persisted A/B/C badge PNGs to both PyBullet and Blender.
    _restore_badges(demo, snapshot, snapshot_dir, id_remap, verbose=verbose)

    # ── Table texture ──────────────────────────────────────────────────
    _restore_table_texture(env, snapshot, snapshot_dir, verbose=verbose)

    if verbose:
        n_ok = sum(1 for i in restored_ids if i is not None)
        print(f"[snapshot] restored {n_ok}/{len(restored_ids)} objects "
              f"(teleport, no physics replay)")

    # ── Re-hydrate the evaluator descriptor (GT) with restored ids ─────
    # If the snapshot carries an annotation, translate its id-keyed fields
    # from generation-time ids → restored ids so the eval loop can score.
    rehydrated_desc = None
    annotation = snapshot.get("annotation") or {}
    saved_desc = annotation.get("question_descriptor")
    if saved_desc:
        rehydrated_desc = remap_descriptor_ids(saved_desc, id_remap)

    return {
        "restored_obj_ids": restored_ids,
        "id_remap": id_remap,
        "question_descriptor": rehydrated_desc,
    }


def _restore_badges(demo, snapshot, snapshot_dir, id_remap, *, verbose=True):
    """Re-apply persisted A/B/C badge textures for a marked_comparison scene.

    Mirrors ``experiments.marked_size.scene_bank.apply_packaged_badges`` but
    keyed on our snapshot's ``badges`` block (obj_id → badge_texture, ep-dir
    relative).  Applies to BOTH PyBullet (``changeVisualShape``) and Blender
    (``camera.set_object_texture``) so the VLM-facing render shows the badges.
    """
    badges = snapshot.get("badges")
    if not badges:
        return
    import pybullet as p
    env = demo.env
    ep_dir = (os.path.dirname(snapshot_dir.rstrip("/"))
              if snapshot_dir else None)
    n_ok = 0
    for b in badges:
        rel = b.get("badge_texture")
        if not rel:
            continue
        png = os.path.normpath(os.path.join(ep_dir, rel)) if ep_dir else rel
        if not os.path.isfile(png):
            if verbose:
                print(f"[snapshot] WARNING: badge texture missing: {png}")
            continue
        # Map the snapshot's saved obj_id → the freshly restored body id.
        snap_oid = b.get("obj_id")
        oid = id_remap.get(int(snap_oid)) if snap_oid is not None else None
        if oid is None:
            if verbose:
                print(f"[snapshot] WARNING: badge {b.get('label')} — no "
                      f"restored body for snapshot obj_id {snap_oid}")
            continue
        try:
            tid = p.loadTexture(png)
            p.changeVisualShape(oid, -1, textureUniqueId=tid)
        except Exception as _pe:
            if verbose:
                print(f"[snapshot] WARNING: PyBullet badge apply failed "
                      f"({b.get('label')}): {_pe}")
        try:
            if hasattr(env.camera, "set_object_texture"):
                env.camera.set_object_texture(oid, png)
        except Exception as _be:
            if verbose:
                print(f"[snapshot] WARNING: Blender badge apply failed "
                      f"({b.get('label')}): {_be}")
        n_ok += 1
    if verbose and n_ok:
        print(f"[snapshot] re-applied {n_ok}/{len(badges)} badge textures")


def _restore_table_texture(env, snapshot, snapshot_dir, *, verbose=True):
    """Apply the snapshot's table texture.

    Resolution order:
      1) shared content-hashed store  (_texture_store/<hash>/, referenced by
         ``texture_store_rel`` relative to the EPISODE dir)
      2) legacy per-episode copy       (``texture_copied`` inside the snapshot)
      3) original cache source path    (``texture_source``)
      4) recompute from ``texture_seed``
    """
    table = snapshot.get("table", {})
    tt = snapshot.get("table_texture", {}) or {}
    tex_dir = None
    # The episode dir is the parent of the scene_snapshot/ dir.
    ep_dir = (os.path.dirname(snapshot_dir.rstrip("/"))
              if snapshot_dir else None)

    # 0) Documented table_texture{} block: per-package dir OR shared store_rel.
    if snapshot_dir and tt.get("dir"):
        cand = os.path.join(snapshot_dir, tt["dir"])
        if os.path.isdir(cand):
            tex_dir = cand
    if tex_dir is None and ep_dir and tt.get("store_rel"):
        cand = os.path.normpath(os.path.join(ep_dir, tt["store_rel"]))
        if os.path.isdir(cand):
            tex_dir = cand

    # 1) Shared content-hashed store (legacy table{} field).
    store_rel = table.get("texture_store_rel")
    if tex_dir is None and store_rel and ep_dir:
        cand = os.path.normpath(os.path.join(ep_dir, store_rel))
        if os.path.isdir(cand):
            tex_dir = cand
    # 2) Legacy per-episode copy inside the snapshot dir.
    if tex_dir is None:
        copied_rel = table.get("texture_copied")
        if copied_rel and snapshot_dir:
            cand = os.path.join(snapshot_dir, copied_rel)
            if os.path.isdir(cand):
                tex_dir = cand
    # 3) Fall back to the original cache source path.
    if tex_dir is None:
        src = table.get("texture_source")
        if src and os.path.isdir(src):
            tex_dir = src
    # 4) Last resort: recompute from the saved seed (new block or legacy).
    tex_seed = tt.get("texture_seed", table.get("texture_seed"))
    if tex_dir is None and tex_seed is not None:
        try:
            from robo_blender.renderer import get_random_table_texture
            tex_dir = get_random_table_texture(seed=tex_seed)
        except Exception:
            tex_dir = None

    # Pin the seed regardless so any deferred pick matches.
    if tex_seed is not None and \
            hasattr(env.camera, "set_table_texture_seed"):
        env.camera.set_table_texture_seed(tex_seed)

    renderer = getattr(env.camera, "_blender_renderer", None)
    if tex_dir and renderer is not None and hasattr(renderer, "set_table_texture"):
        try:
            renderer.set_table_texture(tex_dir)
            # Cancel the deferred seed-based pick so it doesn't override us.
            if hasattr(env.camera, "_pending_table_texture"):
                env.camera._pending_table_texture = False
            if verbose:
                print(f"[snapshot] applied table texture "
                      f"'{os.path.basename(tex_dir)}'")
        except Exception as e:
            if verbose:
                print(f"[snapshot] WARNING: could not apply table texture: {e}")
