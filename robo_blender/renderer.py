"""
BlenderRenderer — Drop-in replacement for pybullet.getCameraImage().

Automatically syncs PyBullet scene to Blender and renders via EEVEE or Cycles.

Usage:
    import pybullet as p
    from robo_blender import BlenderRenderer

    pc = p.connect(p.GUI)
    robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

    renderer = BlenderRenderer(physicsClientId=pc)

    view = p.computeViewMatrix([2, -1.5, 1.8], [0, 0, 0.5], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(50, 640/480, 0.1, 100)
    w, h, rgb, depth, seg = renderer.getCameraImage(640, 480, view, proj)

    # rgb is (H, W, 4) uint8 RGBA, same as pybullet
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import random as _random
import tempfile
import warnings
from typing import Sequence

import numpy as np
import pybullet as p

from robo_blender.client.bridge_client import BridgeClient
from robo_blender.server_launcher import BlenderProcess


# ── HDRI download (Poly Haven API + HuggingFace fallback) ────

_HDRI_HF_REPO = "ZhengGuangze/HDRI"
_HDRI_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "robo_blender", "hdri",
)

# Cache the Poly Haven indoor list so we only hit the API once
_polyhaven_indoor_cache: list[str] | None = None

# Curated whitelist of neutrally-lit indoor HDRIs — studios, labs, lofts,
# overcast interiors with soft daylight.  Avoids sunset / candle-lit /
# stage-lit / tungsten-dominated environments that push a strong red or
# orange cast into the render (observed in April-17 rollout — see
# FIXES_FOR_April18 P2-1).
#
# All of these are from Poly Haven's "indoor" category, resolution 2k.
# The full catalogue has ~288 entries; this subset picks ~25 consistently
# neutral ones by eye.  Falls back to the full Poly Haven list if the
# whitelist is exhausted or unreachable.
# HDRIs with strong directional window / skylight that cast visible
# shadows in EEVEE.  Neutral colour temperature — no warm/cold cast.
# Used by hard lighting mode.
_POLYHAVEN_INDOOR_SHADOW = [
    "dancing_hall",                 # bright side windows, clear cast shadows
    "hotel_room",                   # window light, visible shadows
    "museum_of_ethnography",        # bright skylight, strong shadows
    "old_hall",                     # tall windows, directional shadows
    "empty_workshop",               # overhead windows, neutral
    "events_hall_interior",         # bright ambient, some shadows
    "climbing_gym",                 # overhead skylights, neutral
    "carpentry_shop_02",            # directional light, shadows
    "lythwood_room",                # bright window, neutral
    "reading_room",                 # window light, neutral
    "yoga_room",                    # bright windows, neutral
    "wrestling_gym",                # overhead lights, neutral
]

# Kept for backwards-compat; maps to _POLYHAVEN_INDOOR_SHADOW now.
_POLYHAVEN_INDOOR_NEUTRAL = _POLYHAVEN_INDOOR_SHADOW


def _get_polyhaven_indoor_hdris(neutral_only: bool = True) -> list[str]:
    """Fetch list of indoor HDRI names from Poly Haven API (cached).

    When ``neutral_only`` is True (default), returns the curated neutral
    whitelist above.  Otherwise returns the full catalogue of ~288
    indoor HDRI asset names (studios, rooms, churches, garages, tunnels,
    etc., many with strong colour casts).
    """
    if neutral_only:
        return list(_POLYHAVEN_INDOOR_NEUTRAL)
    global _polyhaven_indoor_cache
    if _polyhaven_indoor_cache is not None:
        return _polyhaven_indoor_cache
    import requests
    r = requests.get(
        "https://api.polyhaven.com/assets?type=hdris&categories=indoor",
        timeout=15,
    )
    r.raise_for_status()
    names = sorted(r.json().keys())
    _polyhaven_indoor_cache = names
    return names


def _download_polyhaven_hdri(name: str, resolution: str = "2k") -> str:
    """Download a specific HDRI from Poly Haven by name.

    Downloads the EXR file at the given resolution (1k/2k/4k).
    Cached locally so each HDRI is downloaded only once.
    """
    import requests
    cache_path = os.path.join(_HDRI_CACHE_DIR, f"{name}_{resolution}.exr")
    if os.path.exists(cache_path):
        return cache_path

    # Get download URL from Poly Haven files API
    r = requests.get(f"https://api.polyhaven.com/files/{name}", timeout=15)
    r.raise_for_status()
    data = r.json()

    # Navigate: hdri -> <resolution> -> exr -> url
    try:
        url = data["hdri"][resolution]["exr"]["url"]
    except KeyError:
        # Fall back to any available resolution
        hdri_data = data["hdri"]
        for res in [resolution, "2k", "1k", "4k"]:
            if res in hdri_data and "exr" in hdri_data[res]:
                url = hdri_data[res]["exr"]["url"]
                break
        else:
            raise RuntimeError(f"No EXR download found for HDRI '{name}'")

    print(f"[HDRI] Downloading {name} ({resolution}) from Poly Haven...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(cache_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"[HDRI] Saved to {cache_path} ({os.path.getsize(cache_path) / 1e6:.1f} MB)")
    return cache_path


def download_random_hdri(
    seed: int | None = None,
    style: str = "any",
) -> str:
    """Download a random HDRI environment map.

    Args:
        seed:  Optional random seed for reproducible selection.
        style: ``"any"``          — pick from HuggingFace dataset
                                    (150 outdoor maps)
               ``"indoor"``       — pick from the curated neutral-indoor
                                    whitelist (avoids warm / coloured casts,
                                    see FIXES_FOR_April18 P2-1); ~25 entries
               ``"indoor_any"``   — pick from Poly Haven's FULL indoor
                                    catalogue (~288 entries, includes
                                    sunset-through-window, candle-lit,
                                    stage-lit rooms — only use if you
                                    explicitly want colour diversity).

    Returns:
        Absolute path to the downloaded ``.exr`` file.
    """
    rng = _random.Random(seed)

    if style in ("indoor", "indoor_any"):
        # Use Poly Haven API for genuine indoor HDRIs.
        neutral_only = (style == "indoor")
        try:
            names = _get_polyhaven_indoor_hdris(neutral_only=neutral_only)
            if names:
                chosen = rng.choice(names)
                return _download_polyhaven_hdri(chosen, resolution="2k")
        except Exception as e:
            print(f"[HDRI] Poly Haven API failed ({e}), falling back to HuggingFace")

    # Fallback / style="any": use HuggingFace dataset
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(_HDRI_HF_REPO, repo_type="dataset")
    exr_files = sorted(f for f in files if f.endswith(".exr"))
    if not exr_files:
        raise RuntimeError(f"No .exr files found in {_HDRI_HF_REPO}")

    chosen = rng.choice(exr_files)

    path = hf_hub_download(
        repo_id=_HDRI_HF_REPO,
        filename=chosen,
        repo_type="dataset",
        cache_dir=_HDRI_CACHE_DIR,
    )
    return os.path.abspath(path)


# ── Random table texture (CCO / ambientCG) ──────────────────────

_TABLE_TEXTURE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "robo_blender", "table_textures",
    "cco_textures",
)


_TABLE_TEXTURE_WARN_SHOWN = False


def get_random_table_texture(seed: int | None = None) -> str | None:
    """Return a random wood texture directory path for table surfaces.

    Looks in the CCO textures cache for Wood* directories.  The textures
    are CC0-licensed (originally from ambientCG.com) and are bundled
    into our HuggingFace dataset ``vineet-datasets/agentic-mqa-objects``
    under ``cco_textures/``.

    To populate the cache, run::

        python scripts/download_assets.py

    which fetches the textures into
    ``~/.cache/robo_blender/table_textures/cco_textures/``.

    Returns:
        Absolute path to a texture directory, or None if the cache is
        missing.  Callers should treat None as "use plain URDF color"
        rather than as an error.
    """
    global _TABLE_TEXTURE_WARN_SHOWN
    if not os.path.isdir(_TABLE_TEXTURE_DIR):
        if not _TABLE_TEXTURE_WARN_SHOWN:
            print(
                f"[robo_blender] CC0 wood texture cache not found at "
                f"{_TABLE_TEXTURE_DIR}.\n"
                f"  Tables will render with the plain URDF color. "
                f"Run `python scripts/download_assets.py` to populate.",
                flush=True,
            )
            _TABLE_TEXTURE_WARN_SHOWN = True
        return None

    # Find Wood/PaintedWood directories (good for tables)
    wood_dirs = sorted([
        os.path.join(_TABLE_TEXTURE_DIR, d)
        for d in os.listdir(_TABLE_TEXTURE_DIR)
        if d.startswith(("Wood", "PaintedWood")) and
           not d.startswith("WoodChips") and
           not d.startswith("WoodSiding") and
           os.path.isdir(os.path.join(_TABLE_TEXTURE_DIR, d))
    ])
    if not wood_dirs:
        return None

    rng = _random.Random(seed)
    return rng.choice(wood_dirs)


def _decompose_view_matrix(
    flat_view: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    """Extract eye, target, up from a PyBullet/OpenGL view matrix.

    PyBullet's computeViewMatrix returns a flat 16-element tuple in
    column-major (OpenGL) order.  We reconstruct eye position, forward
    direction, and up vector.

    Returns:
        (eye, target, up) — each a 3-element list of floats.
        target is 1 unit in front of the eye along the forward direction.
    """
    # Reshape column-major → row-major (transpose)
    V = np.array(flat_view, dtype=np.float64).reshape(4, 4).T

    # 3×3 rotation
    R = V[:3, :3]
    t = V[:3, 3]

    # eye = -R^T * t
    eye = (-R.T @ t).tolist()

    # Forward = -row2 of R (camera looks down -Z in OpenGL convention)
    forward = (-R[2, :]).tolist()

    # Up = row1 of R
    up = R[1, :].tolist()

    # target = eye + forward (1 unit ahead)
    target = [eye[i] + forward[i] for i in range(3)]

    return eye, target, up


def _decompose_projection_matrix(
    flat_proj: Sequence[float],
) -> tuple[float, float, float]:
    """Extract vertical FOV, near, far from a PyBullet/OpenGL projection matrix.

    PyBullet's computeProjectionMatrixFOV returns a flat 16-element tuple
    in column-major order.

    Returns:
        (fov_degrees, near, far)
    """
    P = np.array(flat_proj, dtype=np.float64).reshape(4, 4)
    # Column-major: P[row][col] with OpenGL layout
    # P[1][1] = 1 / tan(fov_y / 2)
    fov_y = 2.0 * math.degrees(math.atan(1.0 / P[1][1]))

    # P[2][2] = -(f+n)/(f-n),  P[3][2] = -2fn/(f-n)
    A = P[2][2]
    B = P[3][2]

    # Avoid division by zero
    if abs(A - 1.0) < 1e-12 or abs(A + 1.0) < 1e-12:
        near, far = 0.1, 100.0
    else:
        near = B / (A - 1.0)
        far = B / (A + 1.0)

    return fov_y, abs(near), abs(far)



def _get_visual_frame(body_id: int, link_index: int, pc: int) -> tuple[list[float], list[float]]:
    """Get the world-space position and orientation for placing a visual mesh.

    For non-base links, getLinkState()[4:6] gives the URDF link frame directly.
    For the base link, getBasePositionAndOrientation() returns the *inertial*
    frame, not the link frame.  We undo the local inertial offset to recover
    the link (visual) frame.

    Returns:
        (position, orientation_xyzw) in world coordinates.
    """
    if link_index == -1:
        inertial_pos, inertial_orn = p.getBasePositionAndOrientation(
            body_id, physicsClientId=pc,
        )
        # getDynamicsInfo returns (..., localInertialPos, localInertialOrn, ...)
        dyn = p.getDynamicsInfo(body_id, -1, physicsClientId=pc)
        local_inertial_pos = dyn[3]
        local_inertial_orn = dyn[4]

        # If there's no inertial offset, just return as-is
        if (abs(local_inertial_pos[0]) < 1e-8 and
            abs(local_inertial_pos[1]) < 1e-8 and
            abs(local_inertial_pos[2]) < 1e-8):
            return list(inertial_pos), list(inertial_orn)

        # link_frame = inertial_frame * inverse(local_inertial_transform)
        # Using PyBullet's multiplyTransforms to do the math
        inv_local_pos, inv_local_orn = p.invertTransform(
            local_inertial_pos, local_inertial_orn,
        )
        link_pos, link_orn = p.multiplyTransforms(
            inertial_pos, inertial_orn,
            inv_local_pos, inv_local_orn,
        )
        return list(link_pos), list(link_orn)
    else:
        state = p.getLinkState(body_id, link_index, physicsClientId=pc)
        return list(state[4]), list(state[5])


def _geom_to_primitive(geom_type: int) -> str:
    """Map PyBullet geometry type to our primitive type string."""
    mapping = {
        p.GEOM_BOX: "box",
        p.GEOM_SPHERE: "sphere",
        p.GEOM_CYLINDER: "cylinder",
        p.GEOM_CAPSULE: "cylinder",  # approximate capsule as cylinder
        p.GEOM_PLANE: "checker_plane",
    }
    return mapping.get(geom_type, "")


def _geom_to_dims(geom_type: int, dimensions: tuple) -> list[float]:
    """Convert PyBullet geometry dimensions to our primitive_dims format.

    PyBullet's getVisualShapeData returns:
      GEOM_BOX:      (full_extent_x, full_extent_y, full_extent_z)
      GEOM_SPHERE:   (radius, 0, 0)
      GEOM_CYLINDER: (length, radius, 0)
      GEOM_CAPSULE:  (length, radius, 0)
    """
    if geom_type == p.GEOM_BOX:
        # dimensions are already full extents
        return [dimensions[0], dimensions[1], dimensions[2]]
    elif geom_type == p.GEOM_SPHERE:
        return [dimensions[0]]  # radius
    elif geom_type == p.GEOM_CYLINDER:
        # (length, radius, 0) → primitive_dims = [radius, length]
        return [dimensions[1], dimensions[0]]
    elif geom_type == p.GEOM_CAPSULE:
        return [dimensions[1], dimensions[0]]
    elif geom_type == p.GEOM_PLANE:
        return [100, 100]
    return []


class BlenderRenderer:
    """Drop-in replacement for ``pybullet.getCameraImage()``.

    If ``auto_start=True`` (the default) and no server is already running,
    a Blender background process is spawned automatically — no separate
    terminal needed.

    Parameters:
        host: Blender server address.
        port: Blender server port.
        engine: ``"EEVEE"`` (fast, ~8fps) or ``"CYCLES"`` (quality).
            Also accepts ``"BLENDER_EEVEE"``, ``"BLENDER_EEVEE_NEXT"``.
        samples: Render samples. Matters for Cycles; EEVEE ignores it.
        denoise: Enable denoising (Cycles only).
        lighting: Lighting preset — ``"studio"``, ``"pybullet_match"``,
            ``"random_hdri"`` (random outdoor from HuggingFace), or
            ``"random_indoor_hdri"`` (random indoor from Poly Haven).
        physicsClientId: PyBullet physics client to track.
        auto_sync: If True, automatically detect new/removed bodies each frame.
        auto_start: If True, auto-start a Blender server if none is running.
            Set to False if you manage the server yourself.
        blender_path: Path to the Blender binary. If None, auto-detected
            from ``BLENDER`` env var, PATH, or common install locations.
        quiet: Suppress Blender server stdout/stderr output.
        connect: If True, connect to Blender server immediately.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        engine: str = "EEVEE",
        samples: int = 16,
        denoise: bool = False,
        lighting: str = "studio",
        light_intensity: float = 0.9,
        hdri_path: str = "",
        physicsClientId: int = 0,
        auto_sync: bool = True,
        auto_start: bool = True,
        blender_path: str | None = None,
        quiet: bool = False,
        connect: bool = True,
        show_floor: bool = True,
        show_robot: bool = True,
    ):
        self.host = host
        self.port = port
        self.engine = engine
        self.samples = samples
        self.denoise = denoise
        self.lighting = lighting
        self.light_intensity = light_intensity
        self.add_directional_light = False  # set via set_directional_light()
        self.physicsClientId = physicsClientId
        self.auto_sync = auto_sync
        # Degenerate-frame guard.  Normally ON (catches truncated EEVEE
        # frames).  Callers that deliberately render near-empty scenes
        # (e.g. verification-image generation — a single isolated object on
        # an empty table) can set this False to skip the false-positive
        # "partial-gray" retries that waste ~10s/frame.
        self.check_degenerate = True
        self.show_floor = show_floor
        self.show_robot = show_robot
        # Explicit allowlist / denylist of PyBullet body IDs.  The env
        # populates this after spawn so the renderer can identify
        # infrastructure bodies (robot, gripper, stand, trays, target
        # table) *by ID* rather than by heuristic name-matching —
        # preventing misclassification of assets whose name happens to
        # contain a robot keyword (e.g. ``panda_android_figure``).  An
        # empty set falls back to the name-based heuristic in
        # ``_is_robot_body``.
        self.infrastructure_body_ids: set = set()
        # Bodies to unconditionally skip when syncing (not loaded into
        # Blender at all).  Independent of show_robot / show_floor — use
        # for hero renders where we want the HDRI to show through where
        # the PyBullet ground plane would otherwise appear.
        self.skipped_body_ids: set = set()

        # Resolve HDRI path — download from HuggingFace if requested
        if hdri_path:
            self.hdri_path = hdri_path
        elif lighting in ("random_hdri", "random_indoor_hdri"):
            style = "indoor" if lighting == "random_indoor_hdri" else "any"
            self.hdri_path = download_random_hdri(style=style)
            hdri_name = os.path.basename(self.hdri_path)
            print(f"[BlenderRenderer] Random HDRI ({style}): {hdri_name}")
        else:
            self.hdri_path = ""

        self._blender_process: BlenderProcess | None = None
        self._client = BridgeClient(host=host, port=port)
        self._synced_bodies: dict[int, set[int]] = {}  # body_id → set of link_indices
        # Cache URDF visual origin offsets so _update_all_transforms can
        # apply them.  Key: (body_id, link_index) → (local_pos, local_orn).
        self._visual_offsets: dict[tuple[int, int], tuple] = {}
        self._scene_initialized = False
        self._blender_path = blender_path   # saved for restart
        self._quiet = quiet
        self._dae_cache_dir = os.path.join(tempfile.gettempdir(), "robo_blender_dae_cache")
        os.makedirs(self._dae_cache_dir, exist_ok=True)

        if connect:
            if auto_start:
                self._blender_process = BlenderProcess(
                    host=host,
                    port=port,
                    blender_path=blender_path,
                    quiet=quiet,
                )
                self._blender_process.start()
            self._client.connect(retries=15, delay=1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def getCameraImage(
        self,
        width: int,
        height: int,
        viewMatrix: Sequence[float] | None = None,
        projectionMatrix: Sequence[float] | None = None,
        lightDirection: Sequence[float] | None = None,
        lightColor: Sequence[float] | None = None,
        lightDistance: float = 1.0,
        shadow: int = 1,
        lightAmbientCoeff: float = 0.6,
        lightDiffuseCoeff: float = 0.35,
        lightSpecularCoeff: float = 0.05,
        renderer: int = 0,
        flags: int = 0,
        physicsClientId: int | None = None,
    ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        """Render the scene through Blender.

        Accepts the same arguments as ``pybullet.getCameraImage()`` and returns
        the same ``(width, height, rgbaPixels, depthPixels, segmentationMask)``
        tuple.

        Parameters that control PyBullet's built-in lighting/renderer are
        accepted but ignored (lighting is configured via Blender).

        Returns:
            ``(width, height, rgba, depth, segmentation)`` where:
            - ``rgba``: ``np.ndarray`` shape ``(H, W, 4)`` dtype ``uint8``
            - ``depth``: ``np.ndarray`` shape ``(H, W)`` dtype ``float32``
              (linearized depth in [0, 1] matching PyBullet's encoding)
            - ``segmentation``: ``np.ndarray`` shape ``(H, W)`` dtype ``int32``
              (placeholder — filled with -1)
        """
        pc = physicsClientId if physicsClientId is not None else self.physicsClientId

        # Treat a persistently uniform frame as a renderer failure, not as
        # valid visual evidence.  A same-process full re-sync can itself leave
        # Blender in this state, so restart the service and replay the frozen
        # PyBullet scene before trying again.  This prevents a gray fallback
        # image from ever being sent to an evaluated VLM.
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                return self._do_render(pc, width, height,
                                       viewMatrix, projectionMatrix)
            except (ConnectionError, BrokenPipeError, OSError) as e:
                if attempt >= max_retries:
                    raise
                print(f"[BlenderRenderer] Blender render unavailable: {e}")
                print(f"[BlenderRenderer] Restarting Blender process "
                      f"(attempt {attempt + 1}/{max_retries})...")
                self._restart_blender()

        # unreachable, but keeps type checkers happy
        raise RuntimeError("getCameraImage: retry loop exited unexpectedly")

    def _restart_blender(self):
        """Kill the dead Blender process, start a fresh one, reconnect."""
        # Tear down old connection
        self._client.close()
        if self._blender_process is not None:
            self._blender_process.stop()
            self._blender_process = None

        # Start fresh
        self._blender_process = BlenderProcess(
            host=self.host,
            port=self.port,
            blender_path=self._blender_path,
            quiet=self._quiet,
        )
        self._blender_process.start()
        self._client.connect(retries=15, delay=1.0)

        # Force full scene re-sync on next render
        self._scene_initialized = False
        self._synced_bodies.clear()
        self._visual_offsets.clear()
        print("[BlenderRenderer] Blender process restarted successfully.")

    def _do_render(self, pc, width, height, viewMatrix, projectionMatrix):
        """Core render logic — separated so getCameraImage can retry."""
        # Sync scene
        if self.auto_sync:
            self._sync_scene(pc)

        # Update all transforms
        self._update_all_transforms(pc)

        # Decompose PyBullet matrices → camera parameters
        if viewMatrix is not None and projectionMatrix is not None:
            eye, target, up = _decompose_view_matrix(viewMatrix)
            fov, near, far = _decompose_projection_matrix(projectionMatrix)
        else:
            # Default camera if no matrices provided
            eye = [2.0, -1.5, 1.8]
            target = [0.0, 0.0, 0.5]
            up = [0.0, 0.0, 1.0]
            fov, near, far = 60.0, 0.1, 100.0

        cam_kw = dict(eye=eye, target=target, up=up,
                      fov=fov, near=near, far=far,
                      resolution=[width, height])

        # Set camera + render.
        self._client.set_camera(**cam_kw)
        rgba, rw, rh = self._render_rgba()

        # Degenerate-frame guard.  A frame can come back bad in two ways:
        #   (a) fully uniform gray  — Blender just (re)started, scene empty;
        #   (b) partially gray      — EEVEE returned an incomplete tile/region
        #       (a large near-uniform block, e.g. the right half of the frame).
        # A partial-gray frame has HIGH global std (real content on one side),
        # so the old ``std < 2.0`` test missed it.  ``_frame_is_degenerate``
        # catches both (it subsumes the marked branch's ``std < 2.0`` check).
        # On failure we force a FULL re-sync (which runs ``clear_scene`` → the
        # camera is deleted) so we MUST re-set the camera before re-rendering,
        # otherwise Blender raises "Cannot render, no camera" (previously
        # crashed the episode).
        max_render_retries = 2 if self.check_degenerate else 0
        for attempt in range(max_render_retries):
            bad, why = self._frame_is_degenerate(rgba)
            if not bad:
                break
            import time
            print(f"[BlenderRenderer] Degenerate frame ({why}) — forcing full "
                  f"re-sync + camera reset and retry "
                  f"({attempt + 1}/{max_render_retries})...")
            self._scene_initialized = False
            self._synced_bodies.clear()
            self._sync_scene(pc)
            self._update_all_transforms(pc)
            # CRITICAL: clear_scene() removed the camera — restore it.
            self._client.set_camera(**cam_kw)
            time.sleep(0.5)
            rgba, rw, rh = self._render_rgba()
        else:
            # Exhausted retries and the last frame is still degenerate.
            if self.check_degenerate:
                bad, why = self._frame_is_degenerate(rgba)
                if bad:
                    print(f"[BlenderRenderer] WARNING: frame still degenerate "
                          f"after {max_render_retries} retries ({why}); "
                          f"returning best-effort frame.")

        # Depth: placeholder (Blender depth pass not implemented yet)
        # PyBullet depth encoding: far*near / (far - (far-near)*d) where d in [0,1]
        # We return 1.0 everywhere (= far plane)
        depth = np.ones((rh, rw), dtype=np.float32)

        # Segmentation: placeholder
        seg = np.full((rh, rw), -1, dtype=np.int32)

        return width, height, rgba, depth, seg

    def _render_rgba(self):
        """Render one frame and decode it to an ``(H, W, 4)`` uint8 array.

        Returns ``(rgba, width, height)``.
        """
        result = self._client.render_frame_bytes(
            samples=self.samples,
            engine=self.engine,
            denoise=self.denoise,
        )
        raw = base64.b64decode(result["data"])
        rw = result["width"]
        rh = result["height"]
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(rh, rw, 4)
        return rgba, rw, rh

    @staticmethod
    def _frame_is_degenerate(rgba: np.ndarray) -> tuple[bool, str]:
        """Detect a broken Blender frame (fully OR partially uniform gray).

        Two failure modes seen in practice:
          * Fully uniform gray  — Blender (re)started with an empty scene
            (global std ≈ 0).
          * Partially gray       — EEVEE returned an incomplete frame where a
            large contiguous region (e.g. the right half) is near-uniform gray
            while the rest holds real content.  The GLOBAL std is high, so we
            must inspect regions.

        Strategy: split the frame into a 4×4 grid of blocks and flag the frame
        if a large fraction of blocks are individually near-uniform (very low
        per-block std).  A healthy cluttered tabletop has texture in almost
        every block; a broken half-frame has ≥1 full column of flat blocks.

        Returns ``(is_degenerate, reason)``.
        """
        rgb = rgba[:, :, :3].astype(np.float32)
        # (a) fully uniform
        if rgb.std() < 2.0:
            return True, "fully-uniform"
        # (b) partial: count near-uniform blocks in a 4×4 grid
        h, w = rgb.shape[:2]
        gh, gw = 4, 4
        flat = 0
        total = gh * gw
        for by in range(gh):
            for bx in range(gw):
                blk = rgb[by * h // gh:(by + 1) * h // gh,
                          bx * w // gw:(bx + 1) * w // gw]
                if blk.std() < 1.5:
                    flat += 1
        # A truncated EEVEE frame leaves roughly HALF the image a solid gray
        # block (the real ep8 defect = 8/16).  Legitimately sparse top-down
        # scenes (lots of empty tabletop around a central pile) can still show
        # several flat blocks on a freshly-rendered frame (ep10 restore hit
        # 6/16 even though it was perfectly fine).  So require ≥ 7 blocks —
        # essentially half the frame — before flagging, to avoid false
        # positives on valid sparse scenes.
        if flat >= 7:
            return True, f"partial-gray ({flat}/{total} flat blocks)"
        return False, ""

    def sync(self, physicsClientId: int | None = None):
        """Manually sync all PyBullet bodies to Blender.

        Call this after loadURDF/loadSDF/removeBody if auto_sync is disabled.
        """
        pc = physicsClientId if physicsClientId is not None else self.physicsClientId
        self._sync_scene(pc)

    def close(self):
        """Disconnect from Blender.

        If the server was auto-started, it is also stopped.
        """
        self._client.close()
        if self._blender_process is not None:
            self._blender_process.stop()
            self._blender_process = None

    def save_blend(self, output_path: str):
        """Save the current Blender scene to a .blend file for inspection."""
        return self._client.save_blend(output_path)

    def insert_keyframes(self, keyframes: list, fps: int = 30):
        """Insert animation keyframes into the Blender timeline.

        Args:
            keyframes: List of frame snapshots from Environment.get_motion_log().
                       Each entry: {frame: int, transforms: [{body_id,
                       link_index, pos, orn_xyzw}, ...]}
            fps:       Playback frame rate. Sets scene.render.fps and
                       frame_start/end in Blender.
        """
        return self._client.insert_keyframes(keyframes, fps=fps)

    def render_animation(self, output_path: str, samples: int = 16,
                         engine: str = "EEVEE", frame_step: int = 3):
        """Render the animated timeline to an MP4 video.

        Frame range and FPS are set by insert_keyframes().

        Args:
            frame_step: Render every Nth frame (3 = 3× faster, 10fps from 30).
        """
        return self._client.render_animation(
            output_path=output_path, samples=samples, engine=engine,
            frame_step=frame_step)

    def set_directional_light(self, enabled: bool) -> None:
        """Enable/disable a directional Sun lamp alongside the HDRI.

        When True, ``configure_lighting`` will add a Sun lamp that casts
        sharp shadows on the table surface (``hard`` lighting mode).
        When False (default), only the HDRI ambient is used (``easy`` mode).

        Call this before the first ``get_cam_img()`` render, or before a
        scene re-sync that triggers ``configure_lighting`` again.
        """
        self.add_directional_light = bool(enabled)

    def set_skipped_body_ids(self, ids) -> None:
        """Never load these body IDs into Blender.  Applied on next sync.

        Unlike ``set_infrastructure_body_ids`` (which is only consulted when
        ``show_robot=False``), this list is honoured unconditionally.
        """
        self.skipped_body_ids = set(int(i) for i in ids if i is not None)

    def set_infrastructure_body_ids(self, ids) -> None:
        """Register the PyBullet body IDs that should be hidden from the
        VLM (robot arm, gripper, stand, trays, target table, etc).

        Once set, ``_is_robot_body`` consults this ID set as its sole
        decision — overriding all name-based heuristics.  Pass an empty
        collection to disable and fall back to heuristics.

        See the 2026-04-21 ``panda_android_figure`` regression for why
        this explicit allowlist exists.
        """
        self.infrastructure_body_ids = set(int(i) for i in ids if i is not None)

    def load_robot_into_scene(self, pc: int):
        """Load robot bodies that were previously skipped (show_robot=False).

        Call this after the VLM loop ends to add the robot arm into the
        Blender scene for animation export.

        Note: ``_is_robot_body()`` also classifies the target table
        (x > 0.5) and tray as "robot" bodies, so they are hidden during
        VLM perception.  Loading them here means they appear for the first
        time — we must re-apply the stored table texture afterwards.
        """
        self.show_robot = True

        # Find robot bodies that were marked as handled but not loaded
        num_bodies = p.getNumBodies(physicsClientId=pc)
        for i in range(num_bodies):
            body_id = p.getBodyUniqueId(i, physicsClientId=pc)
            if body_id not in self._synced_bodies:
                continue
            # If the body was skipped (empty link set), reload it
            if not self._synced_bodies[body_id]:
                body_info = p.getBodyInfo(body_id, physicsClientId=pc)
                body_name = (body_info[1].decode("utf-8")
                             if body_info[1] else f"body{body_id}")
                if self._is_robot_body(body_id, body_name, pc):
                    del self._synced_bodies[body_id]  # remove so _sync loads it
        # Re-sync to load the robot bodies (+ target table, tray)
        self._sync_scene(pc)
        self._update_all_transforms(pc)

        # Re-apply table texture — the target table was just loaded for the
        # first time and only has its default material.
        if self._last_table_texture:
            try:
                self.set_table_texture(self._last_table_texture)
            except Exception as e:
                print(f"[BlenderRenderer] Table texture re-apply failed: {e}")

    def set_object_texture(self, body_id: int, texture_path: str,
                           link_index: int = -1):
        """Apply a baked texture (e.g. a label badge) to one body in Blender.

        Recorded so _restart_blender() can replay it, like the table texture.
        """
        if not hasattr(self, "_object_textures"):
            self._object_textures = {}
        self._object_textures[(int(body_id), int(link_index))] = texture_path
        return self._client.set_object_texture(body_id, texture_path,
                                                link_index=link_index)

    def set_table_texture(self, texture_dir: str, uv_scale: float = 2.0):
        """Apply a PBR wood texture to the table/tabletop in Blender.

        Args:
            texture_dir: Path to a CCO-style texture directory containing
                         ``*_Color.jpg``, ``*_Roughness.jpg``, ``*_Normal.jpg``.
            uv_scale: UV tiling factor (default 2.0).
        """
        self._last_table_texture = texture_dir
        return self._client.set_table_texture(texture_dir, uv_scale=uv_scale)

    def shutdown(self):
        """Shut down the Blender server and disconnect."""
        try:
            self._client.shutdown()
        except Exception:
            pass
        if self._blender_process is not None:
            self._blender_process.stop()
            self._blender_process = None

    # ------------------------------------------------------------------
    # Scene synchronization
    # ------------------------------------------------------------------

    def _sync_scene(self, pc: int):
        """Detect new/removed bodies and sync to Blender.

        Any body texture applied via set_object_texture() is re-applied at the
        end: a sync can re-import a mesh (or a full clear can rebuild the whole
        scene), which restores the asset's ORIGINAL image and silently discards
        runtime textures such as the marked-comparison label badges.
        """
        try:
            self._sync_scene_inner(pc)
        finally:
            for (bid, lidx), path in (getattr(self, "_object_textures", {}) or {}).items():
                try:
                    self._client.set_object_texture(bid, path, link_index=lidx)
                except Exception:
                    pass

    def _sync_scene_inner(self, pc: int):
        self._last_sync_was_full_clear = False  # default: no clear
        if not self._scene_initialized:
            self._last_sync_was_full_clear = True
            self._client.clear_scene()
            self._client.configure_lighting(
                preset=self.lighting, intensity_multiplier=self.light_intensity,
                hdri_path=self.hdri_path,
                add_directional_light=self.add_directional_light,
            )
            self._synced_bodies.clear()
            self._visual_offsets.clear()
            self._scene_initialized = True

        # Get current body list
        num_bodies = p.getNumBodies(physicsClientId=pc)
        current_bodies: set[int] = set()
        for i in range(num_bodies):
            body_id = p.getBodyUniqueId(i, physicsClientId=pc)
            current_bodies.add(body_id)

        # Detect removed bodies — use incremental removal to avoid
        # the expensive full clear+reload cycle (~20 s savings).
        # Sets ``_last_sync_was_full_clear = False`` so callers (camera.py)
        # know that lighting / table texture are still intact.
        removed = set(self._synced_bodies.keys()) - current_bodies
        if removed:
            self._last_sync_was_full_clear = False
            self._client.remove_bodies(list(removed))
            for body_id in removed:
                del self._synced_bodies[body_id]
                # Clean up cached visual offsets for removed body
                stale = [k for k in self._visual_offsets if k[0] == body_id]
                for k in stale:
                    del self._visual_offsets[k]

        # Load new bodies
        for body_id in current_bodies:
            if body_id not in self._synced_bodies:
                self._load_body(body_id, pc)

    def _load_body(self, body_id: int, pc: int):
        """Load a single PyBullet body into Blender."""
        body_info = p.getBodyInfo(body_id, physicsClientId=pc)
        body_name = body_info[1].decode("utf-8") if body_info[1] else f"body{body_id}"

        # Unconditional skip (hero renders: hide the floor plane so the
        # HDRI dome shows through, hide unwanted infra like the target
        # table even when show_robot=True).
        if body_id in self.skipped_body_ids:
            self._synced_bodies[body_id] = set()
            return

        # Skip robot arm if show_robot is disabled — the top-down camera
        # Uses explicit infrastructure-ID allowlist (set by env) when
        # available; falls back to name-heuristic in ``_is_robot_body``.
        # sees through to the table surface instead of the arm blocking the view.
        if not self.show_robot and self._is_robot_body(body_id, body_name, pc):
            self._synced_bodies[body_id] = set()  # mark as handled
            return

        # Floor/ground plane handling:
        # When show_floor is disabled, load it as a plain neutral surface
        # instead of a checkered pattern so the area beyond the table
        # doesn't appear as empty void in the camera view.
        is_floor = self._is_floor_body(body_id, body_name, pc)
        force_plain_floor = is_floor and not self.show_floor

        shapes = p.getVisualShapeData(body_id, physicsClientId=pc)
        link_indices: set[int] = set()

        for shape in shapes:
            link_index = shape[1]
            geom_type = shape[2]
            dimensions = shape[3]
            mesh_path = shape[4].decode("utf-8") if shape[4] else ""
            local_pos = shape[5]
            local_orn = shape[6]
            rgba = list(shape[7])

            link_indices.add(link_index)

            # Get world-space transform for the visual frame.
            # _get_visual_frame gives the LINK frame; we must also apply
            # the local visual origin offset from the URDF <visual><origin>
            # tag (returned by getVisualShapeData as local_pos/local_orn).
            # Without this, objects with non-zero visual origins are
            # placed at the wrong position in Blender (e.g. 8 cm too high).
            link_pos, link_orn = _get_visual_frame(body_id, link_index, pc)
            has_offset = (any(abs(v) > 1e-8 for v in local_pos) or
                          any(abs(v) > 1e-8 for v in local_orn))
            if has_offset:
                pos, orn = p.multiplyTransforms(
                    link_pos, link_orn, local_pos, local_orn)
                pos, orn = list(pos), list(orn)
                self._visual_offsets[(body_id, link_index)] = (
                    tuple(local_pos), tuple(local_orn))
            else:
                pos, orn = link_pos, link_orn

            # Get link name
            link_name = "base"
            if link_index >= 0:
                jinfo = p.getJointInfo(body_id, link_index, physicsClientId=pc)
                link_name = jinfo[12].decode("utf-8")

            # When the floor body is loaded but show_floor is off,
            # override whatever geometry it has (mesh or primitive)
            # with a plain neutral plane.
            if force_plain_floor:
                mesh_path = ""
                mesh_type = "primitive"
                prim_type = "plain_plane"
                prim_dims = [100, 100]
                scale = [1, 1, 1]
            elif geom_type == p.GEOM_MESH and mesh_path:
                # Determine mesh type and primitive info
                # Make path absolute so Blender (which may run in a different
                # working directory) can find the file.
                if not os.path.isabs(mesh_path):
                    mesh_path = os.path.abspath(mesh_path)
                if mesh_path.endswith(".stl"):
                    mesh_type = "stl"
                elif mesh_path.endswith(".dae"):
                    # Convert DAE → OBJ on the client side so Blender 5.x
                    # (which dropped collada_import) can load it.
                    mesh_path = self._convert_dae_to_obj(mesh_path)
                    mesh_type = "obj"
                else:
                    mesh_type = "obj"
                prim_type = ""
                prim_dims: list[float] = []
                scale = list(dimensions)  # GEOM_MESH dimensions = mesh scale
            else:
                mesh_path = ""
                mesh_type = "primitive"
                prim_type = _geom_to_primitive(geom_type)
                prim_dims = _geom_to_dims(geom_type, dimensions)
                scale = [1, 1, 1]

                if not prim_type:
                    # Unknown geometry type — skip
                    warnings.warn(
                        f"Unsupported geometry type {geom_type} for "
                        f"body {body_id} link {link_index}, skipping"
                    )
                    continue

            self._client.load_body(
                body_id=body_id,
                link_index=link_index,
                mesh_path=mesh_path,
                mesh_type=mesh_type,
                position=list(pos),
                orientation_xyzw=list(orn),
                scale=scale,
                rgba=rgba,
                body_name=body_name,
                link_name=link_name,
                primitive_type=prim_type,
                primitive_dims=prim_dims,
            )

        self._synced_bodies[body_id] = link_indices

    def _update_all_transforms(self, pc: int):
        """Send current transforms for all synced bodies."""
        updates = []
        for body_id, link_indices in self._synced_bodies.items():
            for link_index in link_indices:
                link_pos, link_orn = _get_visual_frame(body_id, link_index, pc)

                # Apply cached URDF visual origin offset (if any)
                key = (body_id, link_index)
                if key in self._visual_offsets:
                    lp, lo = self._visual_offsets[key]
                    pos, orn = p.multiplyTransforms(
                        link_pos, link_orn, lp, lo)
                    pos, orn = list(pos), list(orn)
                else:
                    pos, orn = link_pos, link_orn

                updates.append({
                    "body_id": body_id,
                    "link_index": link_index,
                    "position": pos,
                    "orientation_xyzw": orn,
                })

        if updates:
            self._client.batch_update_transforms(updates)

    def _convert_dae_to_obj(self, dae_path: str) -> str:
        """Convert a Collada (.dae) file to OBJ via trimesh.

        Collada files carry a ``<unit meter="…"/>`` tag that specifies how
        many meters one file-unit equals.  trimesh ignores this, so we
        parse it ourselves and apply the scale factor before export.

        Results are cached by file path hash so each DAE is converted only
        once per session.

        Returns:
            Absolute path to the converted .obj file.
        """
        # Deterministic cache key from the absolute path
        key = hashlib.md5(dae_path.encode()).hexdigest()
        obj_path = os.path.join(self._dae_cache_dir, f"{key}.obj")
        if os.path.isfile(obj_path):
            return obj_path

        try:
            import trimesh

            # ── Parse Collada <unit meter="…"/> ──────────────
            unit_scale = 1.0
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(dae_path)
                root = tree.getroot()
                # Collada namespace varies; search with and without it
                for ns in [
                    "{http://www.collada.org/2005/11/COLLADASchema}",
                    "",
                ]:
                    unit_el = root.find(f".//{ns}asset/{ns}unit")
                    if unit_el is not None:
                        meter_str = unit_el.get("meter")
                        if meter_str:
                            unit_scale = float(meter_str)
                            break
            except Exception:
                pass  # If parsing fails, fall through with scale=1.0

            scene = trimesh.load(dae_path, force="scene")

            # Use to_geometry() (or dump(concatenate=True)) to apply scene
            # graph transforms before flattening.  The naive approach of
            # concatenating geometry values ignores node transforms and
            # produces a mesh with wrong rotations/positions.
            try:
                combined = scene.to_geometry()
            except AttributeError:
                # Older trimesh
                combined = scene.dump(concatenate=True)

            # Apply unit conversion so the exported OBJ is in meters
            if abs(unit_scale - 1.0) > 1e-6:
                combined.apply_scale(unit_scale)

            combined.export(obj_path)
        except Exception as exc:
            warnings.warn(
                f"Failed to convert DAE → OBJ ({dae_path}): {exc}. "
                "Using a small placeholder cube instead."
            )
            # Create a tiny placeholder OBJ so Blender gets a valid mesh file
            placeholder_path = obj_path + ".placeholder.obj"
            try:
                with open(placeholder_path, "w") as fp:
                    fp.write("# placeholder for failed DAE conversion\n")
                    fp.write("v -0.01 -0.01 -0.01\nv  0.01 -0.01 -0.01\n")
                    fp.write("v  0.01  0.01 -0.01\nv -0.01  0.01 -0.01\n")
                    fp.write("v -0.01 -0.01  0.01\nv  0.01 -0.01  0.01\n")
                    fp.write("v  0.01  0.01  0.01\nv -0.01  0.01  0.01\n")
                    fp.write("f 1 2 3 4\nf 5 6 7 8\nf 1 2 6 5\n")
                    fp.write("f 3 4 8 7\nf 1 4 8 5\nf 2 3 7 6\n")
                return placeholder_path
            except Exception:
                return dae_path  # last resort
        return obj_path

    @staticmethod
    def _is_floor_body(body_id: int, body_name: str, pc: int) -> bool:
        """Check if a PyBullet body is a ground plane / floor.

        Heuristic: matches ``GEOM_PLANE`` primitives, or mesh bodies whose
        name contains 'plane', 'floor', or 'ground'.
        """
        # Match *tokens*, not substrings: an ordinary asset may contain the
        # word "airplane" in its name.  Treating that as the ground plane
        # replaces the object with a 100 m floor and hides the whole scene.
        # The GEOM_PLANE check below remains the authoritative fallback.
        import re as _re
        tokens = set(_re.split(r"[_/\-\s]+", body_name.lower()))
        if tokens & {"plane", "floor", "ground"}:
            return True
        shapes = p.getVisualShapeData(body_id, physicsClientId=pc)
        for s in shapes:
            if s[2] == p.GEOM_PLANE:
                return True
        return False

    def _is_robot_body(self, body_id: int, body_name: str, pc: int) -> bool:
        """Check if a PyBullet body is the robot arm, gripper, stand,
        or the tray / target table used for placing removed objects.

        These are hidden in the top-down VLM camera view since they
        obstruct the table surface and are not relevant to the VLM.

        Decision order:
          1. If ``self.infrastructure_body_ids`` is populated and
             contains ``body_id``, return True.  If it is populated and
             *doesn't* contain ``body_id``, return False (the ID set is
             authoritative — skip all heuristics so we never mis-hide a
             spawned object).
          2. Otherwise fall back to token-level name matching (still
             used by test scripts that don't register infrastructure
             IDs).

        The 2026-04-21 regression (``panda_android_figure`` invisible
        in Blender because ``"panda"`` appears in its body name)
        motivated step 1.
        """
        # Primary path: explicit infrastructure allowlist set by the env.
        if self.infrastructure_body_ids:
            return body_id in self.infrastructure_body_ids

        # Fallback: name + joint-count heuristic (test harnesses).
        import re as _re
        name_lower = body_name.lower()
        tokens = set(_re.split(r"[_/\-\s]+", name_lower))
        robot_tokens = {
            "robot", "ur5", "kuka", "panda", "iiwa", "lbr",
            "gripper", "robotiq", "banchetto", "ur5_stand",
        }
        if tokens & robot_tokens:
            return True
        # Tray / tabletop (clearing basket for removed objects).
        if "tabletop" in name_lower or "tray" in name_lower:
            return True
        # Target table (second "table" body at x~0.7, away from scene center).
        if "table" in tokens:
            pos, _orn = p.getBasePositionAndOrientation(body_id, physicsClientId=pc)
            if pos[0] > 0.5:
                return True
        # Robot arms typically have many joints.
        n_joints = p.getNumJoints(body_id, physicsClientId=pc)
        if n_joints > 5:
            return True
        return False
