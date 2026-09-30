"""
RoboBlender Protocol — Message framing and serialization.
"""
from __future__ import annotations
"""
Wire format:
    [4 bytes: big-endian uint32 payload length][JSON payload bytes]

Message format (request):
    {"cmd": "COMMAND", "params": {...}, "id": "unique-string"}

Message format (response):
    {"id": "unique-string", "status": "ok"|"error", "data": {...}}

Commands:
    PING                → {"pong": true}
    LOAD_BODY           → imports mesh into Blender scene
    UPDATE_TRANSFORM    → moves an existing object
    SET_CAMERA          → configures render camera
    CONFIGURE_LIGHTING  → adjusts lighting rig
    RENDER_FRAME        → renders current scene to file
    CLEAR_SCENE         → removes all objects
    SHUTDOWN            → stops the bridge server
"""

import json
import struct
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


# --- Wire framing -----------------------------------------------------------

HEADER_SIZE = 4  # bytes
HEADER_FORMAT = ">I"  # big-endian unsigned 32-bit int


def encode_message(msg: dict) -> bytes:
    """Serialize a message dict to length-prefixed JSON bytes."""
    payload = json.dumps(msg, default=str).encode("utf-8")
    header = struct.pack(HEADER_FORMAT, len(payload))
    return header + payload


def read_exact(sock, n: int) -> bytes:
    """Read exactly n bytes from a socket, or raise ConnectionError."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        buf.extend(chunk)
    return bytes(buf)


def decode_message(sock) -> dict:
    """Read one length-prefixed JSON message from a socket."""
    header = read_exact(sock, HEADER_SIZE)
    (length,) = struct.unpack(HEADER_FORMAT, header)
    payload = read_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


# --- Message builders --------------------------------------------------------

def make_id() -> str:
    return str(uuid.uuid4())[:8]


def request(cmd: str, **params) -> dict:
    return {"cmd": cmd, "params": params, "id": make_id()}


def response_ok(msg_id: str, **data) -> dict:
    return {"id": msg_id, "status": "ok", "data": data}


def response_error(msg_id: str, error: str) -> dict:
    return {"id": msg_id, "status": "error", "data": {"error": error}}


# --- Convenience command builders -------------------------------------------

def cmd_ping() -> dict:
    return request("PING")


def cmd_load_body(
    body_id: int,
    link_index: int,
    mesh_path: str,
    mesh_type: str,  # "obj", "stl", "dae", "primitive"
    position: list[float],
    orientation_xyzw: list[float],
    scale: list[float],
    rgba: list[float],
    body_name: str = "",
    link_name: str = "",
    primitive_type: str = "",  # "box", "sphere", "cylinder"
    primitive_dims: list[float] | None = None,
) -> dict:
    return request(
        "LOAD_BODY",
        body_id=body_id,
        link_index=link_index,
        mesh_path=mesh_path,
        mesh_type=mesh_type,
        position=position,
        orientation_xyzw=orientation_xyzw,
        scale=scale,
        rgba=rgba,
        body_name=body_name,
        link_name=link_name,
        primitive_type=primitive_type,
        primitive_dims=primitive_dims or [],
    )


def cmd_update_transform(
    body_id: int,
    link_index: int,
    position: list[float],
    orientation_xyzw: list[float],
) -> dict:
    return request(
        "UPDATE_TRANSFORM",
        body_id=body_id,
        link_index=link_index,
        position=position,
        orientation_xyzw=orientation_xyzw,
    )


def cmd_batch_update_transforms(updates: list[dict]) -> dict:
    """Send multiple transform updates in one message.
    Each update: {"body_id", "link_index", "position", "orientation_xyzw"}
    """
    return request("BATCH_UPDATE_TRANSFORMS", updates=updates)


def cmd_set_camera(
    eye: list[float],
    target: list[float],
    up: list[float] | None = None,
    fov: float = 60.0,
    near: float = 0.1,
    far: float = 100.0,
    resolution: list[int] | None = None,
) -> dict:
    return request(
        "SET_CAMERA",
        eye=eye,
        target=target,
        up=up or [0, 0, 1],
        fov=fov,
        near=near,
        far=far,
        resolution=resolution or [1280, 720],
    )


def cmd_configure_lighting(
    preset: str = "studio",  # "studio", "outdoor", "dramatic"
    intensity_multiplier: float = 1.0,
    hdri_path: str = "",
    add_directional_light: bool = False,
) -> dict:
    return request(
        "CONFIGURE_LIGHTING",
        preset=preset,
        intensity_multiplier=intensity_multiplier,
        hdri_path=hdri_path,
        add_directional_light=add_directional_light,
    )


def cmd_render_frame(
    output_path: str,
    samples: int = 64,
    engine: str = "CYCLES",
    denoise: bool = True,
    *,
    hero_quality: bool = False,
    film_transparent: bool = False,
    color_depth: str = "8",
    resolution=None,
) -> dict:
    return request(
        "RENDER_FRAME",
        output_path=output_path,
        samples=samples,
        engine=engine,
        denoise=denoise,
        hero_quality=hero_quality,
        film_transparent=film_transparent,
        color_depth=color_depth,
        resolution=resolution,
    )


def cmd_render_frame_bytes(
    samples: int = 64,
    engine: str = "CYCLES",
    denoise: bool = True,
) -> dict:
    """Render and return raw pixel data (base64-encoded) instead of file."""
    return request(
        "RENDER_FRAME_BYTES",
        samples=samples,
        engine=engine,
        denoise=denoise,
    )


def cmd_clear_scene() -> dict:
    return request("CLEAR_SCENE")


def cmd_remove_bodies(body_ids: list[int]) -> dict:
    """Remove specific bodies from the Blender scene."""
    return request("REMOVE_BODIES", body_ids=body_ids)


def cmd_save_blend(output_path: str) -> dict:
    """Save the current Blender scene to a .blend file."""
    return request("SAVE_BLEND", output_path=output_path)


def cmd_set_table_texture(texture_dir: str, uv_scale: float = 2.0) -> dict:
    """Apply a PBR texture to table/tabletop objects."""
    return request("SET_TABLE_TEXTURE", texture_dir=texture_dir, uv_scale=uv_scale)


def cmd_set_object_texture(body_id: int, texture_path: str,
                           link_index: int = -1) -> dict:
    """Repoint one already-loaded body's image texture to a new file.

    Needed by the marked-comparison question type: the label badge is baked
    into a copy of the asset's texture map, and Blender must pick it up (it
    imported the original at load time).
    """
    return {
        "cmd": "SET_OBJECT_TEXTURE",
        "params": {
            "body_id": int(body_id),
            "link_index": int(link_index),
            "texture_path": texture_path,
        },
    }


def cmd_set_world_hdri(image_path: str, strength: float = 1.0,
                       rotation_deg: float = 0.0) -> dict:
    """Force the world background to a specific HDRI / image file.

    Bypasses ``configure_lighting``'s preset machinery — used by paper-figure
    rendering scripts to reliably set the visible HDRI background.
    """
    return request("SET_WORLD_HDRI",
                   image_path=image_path,
                   strength=strength,
                   rotation_deg=rotation_deg)


def cmd_insert_keyframes(keyframes: list, fps: int = 30) -> dict:
    """Insert animation keyframes for all bodies across multiple frames.

    Args:
        keyframes: List of frame snapshots, each with ``frame`` (int) and
                   ``transforms`` (list of body/link transform dicts).
        fps:       Playback frame rate — sets scene.render.fps and
                   scene.frame_start/end in Blender.
    """
    return request("INSERT_KEYFRAMES", keyframes=keyframes, fps=fps)


def cmd_render_animation(
    output_path: str,
    samples: int = 16,
    engine: str = "EEVEE",
    frame_step: int = 3,
) -> dict:
    """Render the animated timeline to an MP4 video.

    Frame range and FPS are already set by ``INSERT_KEYFRAMES``.
    ``frame_step`` renders every Nth frame (3 = 3× faster).
    """
    return request(
        "RENDER_ANIMATION",
        output_path=output_path,
        samples=samples,
        engine=engine,
        frame_step=frame_step,
    )


def cmd_load_robot_bodies() -> dict:
    """Ask the renderer to re-sync the scene with show_robot=True.

    Used after the VLM loop ends to add the robot into the Blender scene
    for animation export (it was hidden during perception).
    """
    return request("LOAD_ROBOT_BODIES")


def cmd_shutdown() -> dict:
    return request("SHUTDOWN")
