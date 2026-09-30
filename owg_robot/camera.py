from __future__ import annotations

import pybullet as p
from datetime import datetime
import numpy as np
import math
import os
import time
import cv2

# Optional Blender renderer — only imported when actually used.
_BlenderRenderer = None

def _get_blender_renderer_class():
    """Lazy-import BlenderRenderer to avoid import-time pybullet dependency loops."""
    global _BlenderRenderer
    if _BlenderRenderer is None:
        from robo_blender import BlenderRenderer
        _BlenderRenderer = BlenderRenderer
    return _BlenderRenderer

class Camera2:
    def __init__(self, 
                 position, 
                 orientation, 
                 image_size = (720, 720),
                 intrinsics = (360., 0, 360., 0, 360., 360., 0, 0, 1),
                 zrange = (0.01, 10.),
                 noise=True,
                 shadow=1,
    ):
        # Camera parameters.
        self.x, self.y, self.z = position
        self.position = position
        self.euler = orientation
        self.quaternion = p.getQuaternionFromEuler(self.euler)
        self.rotmat = p.getMatrixFromQuaternion(self.quaternion)
        self.zrange = zrange
        self.znear, self.zfar = self.zrange
        self.noise = noise
        self.shadow = shadow

        self.intrinsics  = intrinsics
        self.focal_len = intrinsics[0]
        self.image_size = image_size
       
        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotm = np.float32(self.rotmat).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        
        self.view_matrix = p.computeViewMatrix(position, lookat, updir)
        self.fovh = (image_size[0] / 2) / self.focal_len
        self.fovh = 180 * np.arctan(self.fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        self.aspect_ratio = image_size[1] / image_size[0]
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fovh, self.aspect_ratio, self.znear, self.zfar)

        self.rec_id = None

    def start_recording(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file = f'{save_dir}/{now}.mp4'

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.rec_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file)

    def stop_recording(self):
        p.stopStateLogging(self.rec_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    def get_cam_img(self):
        """
        Method to get images from camera
        return:
        rgb
        depth
        segmentation mask
        """
        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=self.image_size[1],
            height=self.image_size[0],
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            shadow=1,
            #flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            #renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        return color[..., :3], depth, segm
        image_size = self.image_size
        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if self.noise:
          color = np.int32(color)
          color += np.int32(np.random.normal(0, 3, color.shape))
          color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (self.zfar + self.znear - (2 * zbuffer - 1) * (self.zfar - self.znear))
        depth = (2 * self.znear * self.zfar) / depth
        if self.noise:
          depth += np.random.normal(0, 0.003, depth.shape)
        return color, depth, segm

class Camera:
    def __init__(self, cam_pos, cam_target, near, far, size, fov):
        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size[0:2]
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])

        # Compute intrinsic parameters
        self.fx = self.height / (2 * math.tan(math.radians(fov) / 2))
        self.fy = self.fx * aspect
        self.cx = self.width / 2
        self.cy = self.height / 2
        self.intrinsics = np.array([self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3,3)

        self.rec_id = None

        # Blender renderer — initialised by enable_blender_rendering()
        self._blender_renderer = None

        # Visualization camera — third-person view matching the GUI debug visualizer.
        # Used by capture_frame() so recorded videos show the robot, not the top-down view.
        self.vis_view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.1, -0.30, 0.95],
            distance=1.5,
            yaw=30,
            pitch=-50,
            roll=0,
            upAxisIndex=2,
        )
        self.vis_projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=aspect, nearVal=0.1, farVal=20.0
        )

    # ------------------------------------------------------------------
    # Blender rendering integration
    # ------------------------------------------------------------------

    def enable_blender_rendering(self, physics_client_id: int,
                                  engine: str = "EEVEE",
                                  samples: int = 16,
                                  lighting: str = "random_indoor_hdri",
                                  light_intensity: float = 0.9,
                                  random_table_texture: bool = False,
                                  add_directional_light: bool = False,
                                  show_floor: bool = False,
                                  show_robot: bool = False,
                                  blender_path: str | None = None,
                                  port: int = 9876,
                                  quiet: bool = False):
        """Attach a BlenderRenderer that mirrors the PyBullet scene.

        Once enabled, ``get_cam_img()`` returns Blender RGB while keeping
        PyBullet depth and segmentation.

        Args:
            lighting: Lighting preset — ``"studio"``, ``"random_hdri"``,
                      ``"random_indoor_hdri"`` (Poly Haven indoor HDRI),
                      or ``"pybullet_match"`` (flat ambient).
            light_intensity: HDRI environment strength (default 0.9).
                             Lower = darker scene; 1.0 = unmodified HDRI.
                             Raised from 0.6 to 0.9 in FIXES_FOR_April18
                             P2-1 — at 0.6 the colour cast of warm HDRIs
                             dominated the signal.
            random_table_texture: If True, apply a random wood PBR texture to the
                                  table surface (from CCO textures dataset).
            add_directional_light: If True, add a Sun lamp alongside the HDRI
                                   for directional lighting with cast shadows
                                   ("hard" mode).  Default False ("easy" mode).
            show_floor: If False (default), hide the PyBullet ground plane in
                        Blender so the checkered pattern doesn't appear.
            show_robot: If False (default), hide the robot arm/gripper in
                        Blender so it doesn't block the top-down camera view.
        """
        Cls = _get_blender_renderer_class()
        self._blender_renderer = Cls(
            port=port,
            engine=engine,
            samples=samples,
            lighting=lighting,
            light_intensity=light_intensity,
            physicsClientId=physics_client_id,
            show_floor=show_floor,
            show_robot=show_robot,
            blender_path=blender_path,
            quiet=quiet,
        )
        # Set directional light flag before first render triggers _sync_scene
        if add_directional_light:
            self._blender_renderer.set_directional_light(True)
        # Deferred: apply table texture after first scene sync (when table exists)
        self._pending_table_texture = random_table_texture
        # Optional explicit seed for the texture picker.  When set via
        # ``set_table_texture_seed`` by the caller, it overrides the
        # global-RNG-based selection and makes texture choice
        # deterministic per episode even when other agentic code
        # perturbs the global state.
        self._table_texture_seed = getattr(
            self, "_table_texture_seed", None)
        # If the env already told us which body IDs are infrastructure
        # before we enabled rendering, apply them now.
        pending_ids = getattr(self, "_pending_infrastructure_ids", None)
        if pending_ids:
            self._blender_renderer.set_infrastructure_body_ids(pending_ids)
        print(f"[Camera] Blender rendering enabled (engine={engine}).")

    def set_blender_infrastructure_ids(self, ids) -> None:
        """Tell the Blender renderer which PyBullet body IDs are the
        robot / gripper / trays / tables so it can hide them *by ID*
        rather than by name-heuristic.

        Safe to call before ``enable_blender_rendering`` — the IDs are
        cached and applied on renderer creation.

        See 2026-04-21 regression: ``panda_android_figure`` (a spawned
        object) was hidden because its body name contains the substring
        ``"panda"``.  Passing explicit IDs avoids that class of bug.
        """
        ids = set(int(i) for i in ids if i is not None)
        self._pending_infrastructure_ids = ids
        if self._blender_renderer is not None:
            self._blender_renderer.set_infrastructure_body_ids(ids)

    def set_object_texture(self, body_id: int, texture_path: str,
                           link_index: int = -1):
        """Forward a baked badge texture to Blender (no-op if not enabled)."""
        if self._blender_renderer is None:
            return None
        return self._blender_renderer.set_object_texture(
            body_id, texture_path, link_index=link_index)

    def set_table_texture_seed(self, seed) -> None:
        """Fix the seed used to pick the random wood table texture.

        When set, the deferred texture pick that fires on the first
        Blender render uses ``random.Random(seed).choice(...)`` instead
        of the global RNG.  This guarantees the same episode seed
        selects the same texture regardless of how many random calls
        happened in other agentic code paths (e.g. perception,
        prompt rendering).

        Call this once per episode BEFORE the first ``get_cam_img``.
        """
        self._table_texture_seed = None if seed is None else int(seed)

    def set_table_texture_path(self, texture_dir: str | None) -> None:
        """Apply one exact saved table texture directory on first Blender render.

        Unlike :meth:`set_table_texture_seed`, this does not sample the local
        texture cache. It is used by frozen scene-bank replay, where a scene's
        visual environment must remain identical even if the cache contents or
        sorting changes between generation and evaluation. ``texture_dir`` is
        an absolute path because the scene package owns/copies this directory.
        """
        self._saved_table_texture_path = (None if texture_dir is None
                                          else os.path.abspath(texture_dir))
        # Distinguish "this package deliberately uses plain URDF" from
        # "ordinary rollout: choose a random cached table material".
        self._has_saved_table_texture_path = True

    def disable_blender_rendering(self):
        """Detach and shut down the Blender renderer."""
        if self._blender_renderer is not None:
            try:
                self._blender_renderer.shutdown()
            except Exception:
                pass
            self._blender_renderer = None
            print("[Camera] Blender rendering disabled.")

    @property
    def blender_enabled(self) -> bool:
        return self._blender_renderer is not None

    def get_cam_img_pybullet(self):
        """Get images from PyBullet only (bypass Blender).

        Useful for internal computations (e.g. occlusion masks) that need
        fast rendering and don't require photorealistic RGB.

        return:
            rgb   — (H, W, 3) uint8
            depth — (H, W) float32
            seg   — (H, W) int32
        """
        _w, _h, pb_rgb, depth, seg = p.getCameraImage(
            self.width, self.height,
            self.view_matrix, self.projection_matrix,
        )
        rgb = pb_rgb[:, :, :3]
        return rgb, depth, seg

    def get_cam_img(self):
        """
        Method to get images from camera.

        If Blender rendering is enabled, RGB comes from Blender while
        depth and segmentation still come from PyBullet (Blender does not
        produce compatible segmentation masks).

        return:
            rgb   — (H, W, 3) uint8
            depth — (H, W) float32
            seg   — (H, W) int32
        """
        # Always get PyBullet output for depth + segmentation
        _w, _h, pb_rgb, depth, seg = p.getCameraImage(
            self.width, self.height,
            self.view_matrix, self.projection_matrix,
        )

        if self._blender_renderer is not None:
            # Render RGB via Blender (same camera matrices)
            _, _, bl_rgba, _, _ = self._blender_renderer.getCameraImage(
                self.width, self.height,
                self.view_matrix, self.projection_matrix,
            )
            rgb = bl_rgba[:, :, :3]  # drop alpha

            # Apply deferred table texture (after first scene sync loads
            # all bodies) OR re-apply after a full clear wiped it.
            _need_rerender = False
            if getattr(self, "_pending_table_texture", False):
                self._pending_table_texture = False
                try:
                    from robo_blender.renderer import get_random_table_texture
                    saved_requested = getattr(
                        self, "_has_saved_table_texture_path", False)
                    saved = getattr(self, "_saved_table_texture_path", None)
                    _tex_seed = getattr(self, "_table_texture_seed", None)
                    # A frozen package takes precedence over a cache sample.
                    # A package may explicitly request the plain URDF material
                    # (saved_requested=True, saved=None), which must NOT fall
                    # back to a random table during replay.
                    tex_dir = (saved if saved_requested
                               else get_random_table_texture(seed=_tex_seed))
                    if tex_dir:
                        result = self._blender_renderer.set_table_texture(tex_dir)
                        applied_to = (result.get("applied_to", [])
                                      if isinstance(result, dict) else [])
                        source = "frozen package" if saved_requested else f"seed={_tex_seed}"
                        print(f"[Camera] Table texture: {os.path.basename(tex_dir)} "
                              f"({source})"
                              + (f" → {applied_to}" if applied_to else " (WARNING: no table body found!)"))
                        _need_rerender = bool(applied_to)
                    elif saved_requested:
                        print("[Camera] Table texture: plain URDF (frozen package)")
                except Exception as e:
                    print(f"[Camera] Table texture failed: {e}")
            elif getattr(self._blender_renderer, '_last_sync_was_full_clear', False):
                # Full clear wiped the texture — re-apply.  Incremental
                # body removal preserves textures so no re-render needed.
                _saved_tex = getattr(self._blender_renderer,
                                     '_last_table_texture', None)
                if _saved_tex:
                    try:
                        self._blender_renderer.set_table_texture(_saved_tex)
                        _need_rerender = True
                    except Exception:
                        pass

            # Re-render to pick up the newly applied texture
            if _need_rerender:
                _, _, bl_rgba, _, _ = self._blender_renderer.getCameraImage(
                    self.width, self.height,
                    self.view_matrix, self.projection_matrix,
                )
                rgb = bl_rgba[:, :, :3]
        else:
            rgb = pb_rgb[:, :, :3]

        return rgb, depth, seg

    def get_pointcloud(self):
        """Get 3D pointcloud from perspective depth image.
        Args:xyz
          depth: HxW float array of perspective depth in meters.
          intrinsics: 3x3 float array of camera intrinsics matrix.
        Returns:
          points: HxWx3 float array of 3D points in camera coordinates.
        """
        _, depth, _ = self.get_cam_img()
        depth_image_size = depth.shape
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (self.far + self.near - (2 * zbuffer - 1) * (self.far - self.near))
        depth = (2 * self.near * self.far) / depth
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - self.intrinsics[0, 2]) * (depth / self.intrinsics[0, 0])
        py = (py - self.intrinsics[1, 2]) * (depth / self.intrinsics[1, 1])
        points = np.float32([px, py, depth]).transpose(1, 2, 0)
        return points

    # def start_recording(self, save_dir):
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #     file = f'{save_dir}/{now}.mp4'

    #     p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    #     self.rec_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file)

    # Frames captured per second when recording with capture_frame().
    VIDEO_CAPTURE_FPS = 10

    def start_recording(self, video_path):
        parent = os.path.dirname(video_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._rec_video_path = video_path
        self._rec_writer = None       # lazily created on first frame
        self._rec_last_frame_time = 0.0

    def capture_frame(self):
        """Capture a dual-view frame: 3rd-person (left) + top-down (right).

        Both views use PyBullet rendering (fast ~5ms each) regardless of
        whether Blender is enabled.  Frames are time-throttled to
        VIDEO_CAPTURE_FPS.
        """
        if not getattr(self, "_rec_video_path", None):
            return
        now = time.time()
        if now - self._rec_last_frame_time < 1.0 / self.VIDEO_CAPTURE_FPS:
            return

        # 3rd-person view (left panel)
        _, _, color_3rd, _, _ = p.getCameraImage(
            self.width, self.height,
            self.vis_view_matrix, self.vis_projection_matrix,
        )
        rgb_3rd = np.ascontiguousarray(
            np.array(color_3rd, dtype=np.uint8).reshape(
                self.height, self.width, 4)[:, :, :3])

        # Top-down perception view (right panel)
        _, _, color_top, _, _ = p.getCameraImage(
            self.width, self.height,
            self.view_matrix, self.projection_matrix,
        )
        rgb_top = np.ascontiguousarray(
            np.array(color_top, dtype=np.uint8).reshape(
                self.height, self.width, 4)[:, :, :3])

        # Add small text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rgb_3rd, "3rd Person", (8, 24),
                     font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(rgb_top, "Top-Down (Perception)", (8, 24),
                     font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Stitch side by side
        frame = np.concatenate([rgb_3rd, rgb_top], axis=1)

        if self._rec_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._rec_writer = cv2.VideoWriter(
                self._rec_video_path, fourcc, self.VIDEO_CAPTURE_FPS, (w, h)
            )
        self._rec_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self._rec_last_frame_time = time.time()

        # ── Optional live view ────────────────────────────────────────
        # When MGVQA_LIVE_VIEW=<dir> is set, mirror the current frame to
        # <dir>/frame.jpg so an external HTTP server can stream the
        # rollout while it runs.  PyBullet is intentionally DIRECT-only
        # (see Environment.__init__), so this is the supported way to
        # watch an episode live.  No-op when the env var is unset, so
        # headless benchmark runs are unaffected.
        _live_dir = os.environ.get("MGVQA_LIVE_VIEW", "")
        if _live_dir:
            try:
                os.makedirs(_live_dir, exist_ok=True)
                _tmp = os.path.join(_live_dir, ".frame.tmp.jpg")
                _dst = os.path.join(_live_dir, "frame.jpg")
                cv2.imwrite(_tmp, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                os.replace(_tmp, _dst)   # atomic: readers never see a partial file
            except Exception:
                pass   # live view must never break a run

    def stop_recording(self):
        writer = getattr(self, "_rec_writer", None)
        if writer is not None:
            writer.release()
            self._rec_writer = None
        self._rec_video_path = None
    # ------------------------------------------------------------------
    # Blender animation export
    # ------------------------------------------------------------------

    def export_blend_animation(
        self,
        output_blend_path: str,
        motion_log: list,
        physics_client_id: int,
        playback_fps: int = 30,
    ) -> str:
        """Export a .blend file with manipulation keyframes on the timeline.

        Loads the robot into the Blender scene (it was hidden during VLM
        perception), inserts keyframes from the motion log, and saves.

        Args:
            output_blend_path: Where to save the .blend file.
            motion_log:        Frame snapshots from Environment.get_motion_log().
            physics_client_id: PyBullet physics client ID.
            playback_fps:      Timeline playback rate.

        Returns:
            The output path.
        """
        if self._blender_renderer is None:
            raise RuntimeError("Blender renderer not enabled")
        if not motion_log:
            print("[Camera] No motion frames recorded, skipping blend export.")
            return output_blend_path

        # Load robot bodies into Blender (were hidden for VLM)
        print("[Camera] Loading robot into Blender scene for animation...")
        self._blender_renderer.load_robot_into_scene(physics_client_id)

        # Insert all keyframes
        print(f"[Camera] Inserting {len(motion_log)} keyframes...")
        result = self._blender_renderer.insert_keyframes(
            motion_log, fps=playback_fps)
        print(f"[Camera] Keyframes inserted: {result}")

        # Save the .blend
        print(f"[Camera] Saving blend file → {output_blend_path}")
        self._blender_renderer.save_blend(output_blend_path)
        return output_blend_path

    def render_blend_video(
        self,
        output_video_path: str,
        motion_log: list,
        physics_client_id: int,
        playback_fps: int = 30,
        samples: int = 16,
        engine: str = "EEVEE",
    ) -> str:
        """Render a Blender animation video from recorded manipulation.

        Loads the robot, inserts keyframes, and renders to MP4.

        Args:
            output_video_path: Where to save the .mp4 file.
            motion_log:        Frame snapshots from Environment.get_motion_log().
            physics_client_id: PyBullet physics client ID.
            playback_fps:      Timeline playback rate.
            samples:           Render samples (EEVEE TAA samples).
            engine:            Render engine ("EEVEE" or "CYCLES").

        Returns:
            The output path.
        """
        if self._blender_renderer is None:
            raise RuntimeError("Blender renderer not enabled")
        if not motion_log:
            print("[Camera] No motion frames recorded, skipping video render.")
            return output_video_path

        # Load robot bodies into Blender (were hidden for VLM)
        print("[Camera] Loading robot into Blender scene for video render...")
        self._blender_renderer.load_robot_into_scene(physics_client_id)

        # Insert all keyframes
        print(f"[Camera] Inserting {len(motion_log)} keyframes...")
        result = self._blender_renderer.insert_keyframes(
            motion_log, fps=playback_fps)
        print(f"[Camera] Keyframes inserted: {result}")

        # Render the animation
        total_frames = result.get("total_frames", len(motion_log))
        duration = total_frames / playback_fps
        print(f"[Camera] Rendering {total_frames} frames ({duration:.1f}s) → {output_video_path}")
        render_result = self._blender_renderer.render_animation(
            output_video_path, samples=samples, engine=engine)
        print(f"[Camera] Render complete: {render_result}")
        return output_video_path
