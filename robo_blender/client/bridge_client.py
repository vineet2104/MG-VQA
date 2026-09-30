"""
Bridge Client — High-level API for the Blender bridge server.
"""
from __future__ import annotations

import socket
import time
from robo_blender.protocol.messages import (
    encode_message, decode_message,
    cmd_ping, cmd_load_body, cmd_update_transform,
    cmd_batch_update_transforms, cmd_set_camera,
    cmd_configure_lighting, cmd_render_frame,
    cmd_render_frame_bytes, cmd_save_blend,
    cmd_set_table_texture, cmd_set_world_hdri,
    cmd_insert_keyframes, cmd_render_animation,
    cmd_clear_scene, cmd_remove_bodies, cmd_shutdown,
)


class BridgeClient:
    def __init__(self, host="127.0.0.1", port=9876, timeout=300.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._sock: socket.socket | None = None

    def connect(self, retries=30, delay=1.0):
        for attempt in range(retries):
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(self.timeout)
                self._sock.connect((self.host, self.port))
                print(f"[Client] Connected to {self.host}:{self.port}")
                return
            except (ConnectionRefusedError, OSError):
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise ConnectionError(f"Cannot connect to {self.host}:{self.port}")

    def close(self):
        if self._sock:
            try: self._sock.close()
            except Exception: pass
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *a):
        self.close()

    def _send(self, msg: dict) -> dict:
        if not self._sock:
            raise RuntimeError("Not connected")
        self._sock.sendall(encode_message(msg))
        resp = decode_message(self._sock)
        if resp.get("status") == "error":
            raise RuntimeError(f"Server error: {resp.get('data',{}).get('error','?')}")
        return resp.get("data", {})

    def ping(self) -> bool:
        return self._send(cmd_ping()).get("pong", False)

    def clear_scene(self):
        return self._send(cmd_clear_scene())

    def remove_bodies(self, body_ids: list[int]):
        """Remove specific bodies from Blender (incremental, no full re-sync)."""
        return self._send(cmd_remove_bodies(body_ids))

    def load_body(self, body_id, link_index, mesh_path="", mesh_type="obj",
                  position=None, orientation_xyzw=None, scale=None, rgba=None,
                  body_name="", link_name="", primitive_type="", primitive_dims=None):
        return self._send(cmd_load_body(
            body_id=body_id, link_index=link_index,
            mesh_path=mesh_path, mesh_type=mesh_type,
            position=position or [0,0,0],
            orientation_xyzw=orientation_xyzw or [0,0,0,1],
            scale=scale or [1,1,1],
            rgba=rgba or [0.5,0.5,0.5,1.0],
            body_name=body_name, link_name=link_name,
            primitive_type=primitive_type,
            primitive_dims=primitive_dims,
        ))

    def update_transform(self, body_id, link_index, position, orientation_xyzw):
        return self._send(cmd_update_transform(body_id, link_index, position, orientation_xyzw))

    def batch_update_transforms(self, updates):
        return self._send(cmd_batch_update_transforms(updates))

    def set_camera(self, eye, target, up=None, fov=60.0, near=0.1, far=100.0, resolution=None):
        return self._send(cmd_set_camera(
            eye=eye, target=target, up=up, fov=fov, near=near, far=far, resolution=resolution,
        ))

    def configure_lighting(self, preset="studio", intensity_multiplier=1.0,
                           hdri_path="", add_directional_light=False):
        return self._send(cmd_configure_lighting(
            preset=preset, intensity_multiplier=intensity_multiplier,
            hdri_path=hdri_path, add_directional_light=add_directional_light))

    def render_frame(self, output_path, samples=64, engine="CYCLES", denoise=True,
                      hero_quality=False, film_transparent=False,
                      color_depth="8", resolution=None):
        return self._send(cmd_render_frame(
            output_path=output_path, samples=samples, engine=engine, denoise=denoise,
            hero_quality=hero_quality, film_transparent=film_transparent,
            color_depth=color_depth, resolution=resolution,
        ))

    def set_object_texture(self, body_id: int, texture_path: str,
                           link_index: int = -1):
        """Repoint one body's image texture (marked-comparison badges)."""
        from robo_blender.protocol.messages import cmd_set_object_texture
        return self._send(cmd_set_object_texture(body_id, texture_path,
                                                 link_index=link_index))

    def render_frame_bytes(self, samples=64, engine="CYCLES", denoise=True):
        """Render and return raw RGBA pixels as numpy array.

        Returns:
            dict with keys: width, height, channels, data (base64 string)
        """
        return self._send(cmd_render_frame_bytes(samples=samples, engine=engine, denoise=denoise))

    def save_blend(self, output_path):
        """Save the current Blender scene to a .blend file."""
        return self._send(cmd_save_blend(output_path=output_path))

    def set_table_texture(self, texture_dir, uv_scale=2.0):
        """Apply a PBR texture to table/tabletop objects."""
        return self._send(cmd_set_table_texture(texture_dir=texture_dir, uv_scale=uv_scale))

    def set_world_hdri(self, image_path, strength=1.0, rotation_deg=0.0):
        """Force the world background to a specific HDRI / image file."""
        return self._send(cmd_set_world_hdri(
            image_path=image_path, strength=strength,
            rotation_deg=rotation_deg))

    def insert_keyframes(self, keyframes, fps=30):
        """Insert animation keyframes into the Blender timeline.

        Args:
            keyframes: List of frame snapshots from Environment.get_motion_log().
            fps:       Playback frame rate.
        """
        return self._send(cmd_insert_keyframes(keyframes=keyframes, fps=fps))

    def render_animation(self, output_path, samples=16, engine="EEVEE",
                         frame_step=3):
        """Render the animated timeline to an MP4 video.

        Temporarily removes the socket timeout since rendering can take
        many minutes (e.g. 890 frames × ~2s/frame ≈ 30 min).
        """
        if self._sock:
            self._sock.settimeout(None)  # block indefinitely during render
        try:
            return self._send(cmd_render_animation(
                output_path=output_path, samples=samples, engine=engine,
                frame_step=frame_step))
        finally:
            if self._sock:
                self._sock.settimeout(self.timeout)  # restore original

    def shutdown(self):
        try: self._send(cmd_shutdown())
        except Exception: pass
        self.close()
