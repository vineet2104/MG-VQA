import time
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter, HtmlFormatter
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
from typing import *

from owg_robot.env import *
from owg_robot.camera import Camera
try:
    from owg.policy import OwgPolicy
    _OWG_POLICY_AVAILABLE = True
except ImportError:
    _OWG_POLICY_AVAILABLE = False
from owg.utils.config import load_config
from owg.utils.grasp import Grasp2D
try:
    from third_party.grconvnet import load_grasp_generator
except ImportError:
    load_grasp_generator = None  # GR-ConvNet not available; using GraspGen

import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "third_party", "graspgen"))

# GUI stuff
# Function to create a text input dialog using Tkinter
def ask_for_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_input = input("User Input: ")
    root.destroy()
    return user_input

class RobotEnvUI:
    def __init__(self,
             config: Union[Dict[str, Any], str],
             spawn_dir: Optional[str] = None,
             selected_obj_names: Optional[List[str]] = None,
             selected_obj_folders: Optional[List[str]] = None,
             asset_path: str = "./owg_robot/assets/unified_objects",
             use_basket: bool = False,
             basket_scale: float = 1.0,
             basket_wall_height: float = 0.12,
             use_planes: bool = False,
             gripper_type: str = '140',
             two_pass_spawn: bool = True,
             spawn_drop_heights: Optional[List[float]] = None,
             cover_over_target_count: int = 0):
        """
        Args:
            config:              Environment config dict or path to YAML.
            spawn_dir:           Directory to save spawn screenshots/video.
            selected_obj_names:  List of display names for the objects to spawn
                                 (e.g. ["soccer cleat", "banana"]).  May contain
                                 duplicates if multiple instances of the same object
                                 type are desired.
            selected_obj_folders: Parallel list of asset folder names for URDF
                                  resolution (e.g. ["11pro_sl_trx_fg", "banana"]).
                                  When None, names are used as folder names (legacy).
            asset_path:          Root directory containing per-object folders.
            use_basket:          If True, load a shallow basket on the table and
                                 constrain object spawning to its interior.
            basket_scale:        Scale factor for the basket URDF (default 1.0).
                                 Increase to get a wider basket for more objects.
            two_pass_spawn:      If True (default), use two-pass spawning for
                                 denser clutter. See spawn() docstring.
        """
        self.cfg = load_config(config) if isinstance(config, str) else config
        self.n_objects = self.cfg.n_objects
        self.seed = self.cfg.seed
        self.selected_obj_names = selected_obj_names
        self.selected_obj_folders = selected_obj_folders
        self.asset_path = asset_path
        self._two_pass_spawn = two_pass_spawn
        # Optional per-object drop heights (meters above table), parallel to
        # selected_obj_names.  When set, overrides pass-based heights so the
        # caller can control the layering (used by counting: targets high,
        # distractors half-height, big cover last on top).  Bypasses two-pass.
        self._spawn_drop_heights = spawn_drop_heights
        # Counting only: when > 0, the FIRST ``cover_over_target_count`` spawned
        # objects are the target instances, and the LAST spawned object (the
        # "big cover" distractor) is dropped directly onto one of those settled
        # targets so it reliably buries a target instance (higher gate yield).
        self._cover_over_target_count = int(cover_over_target_count or 0)

        # init env
        print("[RobotEnvUI] Initializing camera and PyBullet environment...", flush=True)
        center_x, center_y, center_z = CAM_X, CAM_Y, CAM_Z
        cam_center = (self.cfg.camera.center_x, self.cfg.camera.center_y,
                      self.cfg.camera.center_z)
        cam_target = (self.cfg.camera.target_x, self.cfg.camera.target_y,
                      self.cfg.camera.target_z)
        self.img_size = (self.cfg.camera.img_size, self.cfg.camera.img_size)
        self.camera = Camera(cam_center, cam_target, self.cfg.camera.znear,
                             self.cfg.camera.zfar, self.img_size,
                             self.cfg.camera.fov)
        self.n_grasp_attempts = self.cfg.n_grasp_attempts
        self.env = Environment(self.camera,
                               vis=self.cfg.policy.vis,
                               asset_root='./owg_robot/assets',
                               debug=False,
                               gripper_type=gripper_type,
                               finger_length=self.cfg.finger_length,
                               n_grasp_attempts=self.cfg.n_grasp_attempts,
                               use_basket=use_basket,
                               basket_scale=basket_scale,
                               basket_wall_height=basket_wall_height,
                               use_planes=use_planes)
        print("[RobotEnvUI] Environment ready.", flush=True)

        self.env.dummy_simulation_steps(10)

        # init OWG policy (optional — only needed for legacy grasp ranking)
        self.policy = None
        self.grasp_rank_3d = False
        if _OWG_POLICY_AVAILABLE:
            print("[RobotEnvUI] Loading OWG policy...", flush=True)
            self.policy = OwgPolicy(
                self.cfg.policy.config_path,
                verbose=self.cfg.policy.verbose,
                vis=self.cfg.policy.vis,
                use_grasp_ranker=self.cfg.policy.use_grasp_ranker)
            print("[RobotEnvUI] OWG policy loaded.", flush=True)
            if self.cfg.policy.use_grasp_ranker:
                self.grasp_rank_3d = self.policy.grasp_ranker.use_3d_prompt
        else:
            print("[RobotEnvUI] OWG policy not available (owg.visual_prompt missing) — "
                  "using GraspGen directly via tools.py.", flush=True)

        # spawn scene
        print("[RobotEnvUI] Spawning scene...", flush=True)
        obs = self.spawn(self.n_objects, spawn_dir=spawn_dir,
                         two_pass=self._two_pass_spawn)
        print("[RobotEnvUI] Scene spawned.", flush=True)

        # Grasp generators
        if load_grasp_generator is not None:
            print("[RobotEnvUI] Loading GR-ConvNet grasp generator...", flush=True)
            self.grasp_generator = load_grasp_generator(self.env.camera)
        else:
            self.grasp_generator = None
        self.graspgen_sampler = None  # assigned by demo.py after loading
        print("[RobotEnvUI] Init complete.", flush=True)

        self.n_action_attempts = self.cfg.n_action_attempts
        self.n_grasp_attempts = self.cfg.n_grasp_attempts
    
    def load_graspgen_sampler(self, gripper_config: str):
        """Load the GraspGen diffusion model for 6-DOF grasp generation.

        Args:
            gripper_config: Path to the gripper YAML config (e.g.
                ``GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml``).
        """
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        print(f"[GraspGen] Loading config: {gripper_config}")
        cfg = load_grasp_cfg(gripper_config)
        print("[GraspGen] Initializing sampler (loading checkpoints)...")
        self.graspgen_sampler = GraspGenSampler(cfg)
        print("[GraspGen] Sampler ready.")

    def spawn(self, n_objects, spawn_dir: Optional[str] = None,
              two_pass: bool = False): 
        """Spawn objects into the scene.

        Args:
            n_objects: Number of objects to spawn.
            spawn_dir: If set, save per-object spawn screenshots here.
            two_pass: If True, use two-pass spawning (forced stacking).
                Default False — single-pass with moderate friction
                produces natural-looking clutter (horizontal overlap
                without unnatural vertical towers).
        """
        self.n_objects = n_objects

        if self.seed is not None:
            random.seed(int(self.seed))
            np.random.seed(int(self.seed))
            
        video_path = None
        if spawn_dir is not None:
            os.makedirs(spawn_dir, exist_ok=True)
            video_path = os.path.join(spawn_dir, "spawn.mp4")
            print(f"[spawn] Starting recording to {video_path}", flush=True)
            self.env.camera.start_recording(video_path)

        try:
            spawn_names = list(self.selected_obj_names or [])[:self.n_objects]
            # Build parallel folder list; fall back to name==folder (sungsu)
            if self.selected_obj_folders is not None:
                spawn_folders = list(self.selected_obj_folders)[:self.n_objects]
            else:
                spawn_folders = list(spawn_names)

            import time as _time
            n_total = len(spawn_names)

            # Per-object drop-height plan (counting): overrides pass logic so
            # targets, fillers and the big cover land in a controlled order.
            drop_plan = self._spawn_drop_heights
            _use_drop_plan = (drop_plan is not None
                              and len(drop_plan) >= n_total)

            # ── Two-pass split ──────────────────────────────────────────
            # Pass 1 (base layer): first ~60% of objects with normal region
            # Pass 2 (top layer): remaining ~40% dropped from tight center
            #   at higher altitude to land ON TOP of the base layer
            if _use_drop_plan:
                n_pass1 = n_total  # drop-plan drives heights; no pass split
            elif two_pass and n_total >= 5:
                n_pass1 = max(3, int(n_total * 0.6))
            else:
                n_pass1 = n_total  # single pass (fallback)
            n_pass2 = n_total - n_pass1

            # Pass 2 uses a tight spawn radius (±4cm × ±3cm) and higher
            # drop height to create stacking/occlusion.
            _PASS2_RADIUS = (0.04, 0.03)   # meters (half-extents)
            _PASS2_DROP_H = 0.25 + 0.05    # meters above table (higher than pass 1)

            print(f"[spawn] Spawning {n_total} objects: {spawn_names}", flush=True)
            if n_pass2 > 0:
                print(f"[spawn] Two-pass: {n_pass1} base + {n_pass2} top", flush=True)

            _t_load_total = 0.0
            _t_physics_total = 0.0
            _t_screenshot_total = 0.0

            # Counting: track ids of the target instances (first _nc spawned)
            # so the big-cover distractor can be dropped onto one of them.
            _n_cover_tgt = self._cover_over_target_count
            _target_ids_spawned = []

            for i, (obj_name, obj_folder) in enumerate(zip(spawn_names, spawn_folders)):
                urdf_path = os.path.join(self.asset_path, obj_folder, "model.urdf")

                # Determine spawn parameters based on pass (or drop plan)
                is_pass2 = (i >= n_pass1)
                spawn_kwargs = {}
                if _use_drop_plan:
                    # Controlled layering INSIDE the containment walls: only
                    # override the per-object drop height (targets low = base
                    # pile, fillers higher = land on top, big cover highest).
                    # Keep the wall-region XY so nothing bounces/scatters out.
                    spawn_kwargs["drop_height"] = float(drop_plan[i])
                    # Big cover = LAST object: drop it onto a settled target
                    # instance so it reliably buries one (better gate yield).
                    if (_n_cover_tgt > 0 and i == n_total - 1
                            and _target_ids_spawned):
                        _tid = random.choice(_target_ids_spawned)
                        try:
                            _tpos = self.env.get_obj_pos(_tid)
                            spawn_kwargs["drop_xy"] = (float(_tpos[0]),
                                                       float(_tpos[1]))
                            print(f"[spawn] big cover → dropping over target "
                                  f"instance id={_tid} at "
                                  f"({_tpos[0]:.3f}, {_tpos[1]:.3f})",
                                  flush=True)
                        except Exception as _e:
                            print(f"[spawn] big-cover targeting failed: {_e}",
                                  flush=True)
                elif is_pass2:
                    spawn_kwargs["spawn_radius"] = _PASS2_RADIUS
                    spawn_kwargs["drop_height"] = _PASS2_DROP_H

                _t0 = _time.time()
                self.env.load_isolated_obj(urdf_path, obj_name,
                                           mod_orn=False, mod_stiffness=False,
                                           **spawn_kwargs)
                # Record target-instance id right after it settles (targets are
                # the first _n_cover_tgt objects in the drop plan).
                if _n_cover_tgt > 0 and i < _n_cover_tgt and self.env.obj_ids:
                    _target_ids_spawned.append(self.env.obj_ids[-1])
                _t1 = _time.time()
                self.env.dummy_simulation_steps(30)
                _t2 = _time.time()
                _t_load_total += (_t1 - _t0)
                _t_physics_total += (_t2 - _t1)
                pass_tag = " [TOP]" if is_pass2 else ""
                print(f"[spawn] [{i+1}/{n_total}] {obj_name}: "
                      f"load={_t1-_t0:.2f}s physics={_t2-_t1:.2f}s{pass_tag}",
                      flush=True)

                if spawn_dir is not None:
                    _ts0 = _time.time()
                    rgb, _, _ = self.env.camera.get_cam_img_pybullet()
                    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in obj_name)
                    img_path = os.path.join(spawn_dir, f"spawn_{i:03d}_{safe_name}.png")
                    cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    _t_screenshot_total += (_time.time() - _ts0)

                # ── Inter-pass settle: after base layer is complete,
                # let everything come to rest before dropping the top layer.
                if i == n_pass1 - 1 and n_pass2 > 0:
                    _tmid = _time.time()
                    _vis_saved = self.env.vis
                    self.env.vis = False
                    self.env.wait_until_all_still(max_wait_epochs=1000)
                    self.env.vis = _vis_saved
                    print(f"[spawn] Base layer settled ({_time.time()-_tmid:.2f}s), "
                          f"dropping top layer...", flush=True)

            # ── Final global settle: let ALL objects reach equilibrium after
            # the last drop.  Earlier objects may have been disturbed by
            # subsequent drops and need time to re-settle as a group.
            # Temporarily disable vis so step_simulation() skips the
            # expensive capture_frame() + sleep during settle.
            _tsettle0 = _time.time()
            print("[spawn] Final global settle (all objects)...", flush=True)
            _vis_saved = self.env.vis
            self.env.vis = False
            self.env.wait_until_all_still(max_wait_epochs=2000)
            self.env.vis = _vis_saved
            _tsettle1 = _time.time()
            print(f"[spawn] TIMING: load={_t_load_total:.2f}s "
                  f"physics={_t_physics_total:.2f}s "
                  f"screenshots={_t_screenshot_total:.2f}s "
                  f"global_settle={_tsettle1-_tsettle0:.2f}s", flush=True)

            print("[spawn] Getting obs...", flush=True)
            self.init_obj_state = self.env.get_obj_states()
            obs = self.env.get_obs()
            print("[spawn] Removing out-of-bounds objects...", flush=True)
            self.env.remove_out_of_bounds_objs(width=0.8)
            return obs

        finally:
            if spawn_dir is not None:
                self.env.camera.stop_recording()

    def reset_same(self):
        assert self.init_obj_state is not None, "Have to spawn once to initialize state"
        self.env.reset_robot()
        self.env.set_obj_state(self.init_obj_state)
        self.env.dummy_simulation_steps(10)
        obs = self.update()
        self.init_obj_state = self.env.get_obj_states()
        for _ in range(30):
            self.env.step_simulation()
        return obs

    def reset(self, new=False):
        if new:
            self.env.remove_all_obj()
            for _ in range(30):
                self.env.step_simulation()
            self.env.dummy_simulation_steps(10)
            return self.spawn(self.n_objects)
        return self.reset_same()

    def update(self):
        self.env.dummy_simulation_steps(10)
        self.env.update_obj_states()
        obs = self.env.get_obs()
        self.env.dummy_simulation_steps(10)
        return obs

    def compute_grasps_from_mask(
        self,
        obj_id: int,
        sam_mask: np.ndarray,
    ):
        """Compute grasps for a single object using a SAM3 segmentation mask.

        This is called on-demand when the VLM requests ``grasp_and_remove``.
        The SAM3 mask (from the VLM perception step) is used instead of the
        ground-truth PyBullet segmentation mask, so the grasp pipeline only
        uses information the agent actually has access to.

        The computed grasps are stored via ``env.set_obj_grasps()`` so that
        the existing ``put_obj_in_tray`` / ``pick_obj_by_id`` methods work
        unchanged.

        Requires GR-ConvNet (``load_grasp_generator``).

        Args:
            obj_id:   PyBullet body ID of the target object.
            sam_mask: (H, W) boolean mask from SAM3 at the camera resolution.
        """
        if self.grasp_generator is None:
            print("[compute_grasps_from_mask] GR-ConvNet not loaded; skipping.")
            return None
        # Get fresh PyBullet observation (depth + pointcloud)
        rgb_pb, depth, _ = self.env.camera.get_cam_img_pybullet()
        pointcloud_cam = self.env.camera.get_pointcloud()
        img_size = self.grasp_generator.IMG_WIDTH

        # Work on copies so we don't modify originals
        rgb = rgb_pb.copy()
        depth = depth.copy()
        mask = sam_mask.copy()

        if img_size != self.env.camera.width:
            rgb = cv2.resize(rgb, (img_size, img_size))
            depth = cv2.resize(depth, (img_size, img_size))
            pointcloud_cam = cv2.resize(
                pointcloud_cam, (img_size, img_size),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = np.array(
                Image.fromarray(mask).resize(
                    (img_size, img_size), Image.LANCZOS
                )
            )

        grasps, grasp_rects = self.grasp_generator.predict_grasp_from_mask_3d(
            rgb, depth, mask, pointcloud_cam,
            n_grasps=self.n_grasp_attempts, show_output=False,
        )

        if img_size != self.env.camera.width:
            for j, gr in enumerate(grasp_rects):
                grasp_rects[j][0] = int(gr[0] / img_size *
                                        self.env.camera.width)
                grasp_rects[j][1] = int(gr[1] / img_size *
                                        self.env.camera.width)
                grasp_rects[j][4] = int(gr[4] / img_size *
                                        self.env.camera.width)
                grasp_rects[j][3] = int(gr[3] / img_size *
                                        self.env.camera.width)

        grasp_rects = [
            Grasp2D.from_vector(
                x=g[1], y=g[0], w=g[4], h=g[3], theta=g[2],
                W=self.env.camera.width, H=self.env.camera.width,
                normalized=False, line_offset=5,
            ) for g in grasp_rects
        ]
        self.env.set_obj_grasps(obj_id, grasps, grasp_rects)
        return grasps

    # ── GraspGen constants (see GraspGen_TUNING.MD) ──────────────────────
    _GRIPPER_DEPTH    = 0.195   # Robotiq 2F-140 base → contact along +Z
    _MAX_APPROACH_DEG = 60.0    # reject grasps steeper than this from vertical
    _MAX_CONTACT_DIST = 0.05    # contact must be within 5 cm of object centroid
    _TABLE_Z          = 0.785   # table surface height in world frame

    def compute_grasps_graspgen(
        self,
        obj_id: int,
        sam_mask: np.ndarray,
    ):
        """Compute 6-DOF grasps using GraspGen from a SAM3 mask.

        Full pipeline matching the proven ``test_graspgen.py`` approach:
          1. Build robot-frame point cloud from masked depth.
          2. Run GraspGen inference (500 candidates → 200 top-k).
          3. Filter: angle ≤ 60° from vertical, contact near centroid,
             base above table.
          4. Apply frame conversion: ``R_ee = R_graspgen @ Ry(-π/2)``.
          5. Rank by confidence descending.

        Returns:
            List of dicts with keys ``pos``, ``quat_ee``, ``conf``,
            ``angle_vert``, sorted by confidence descending.
            Empty list if no feasible grasps found.
        """
        if self.graspgen_sampler is None:
            raise RuntimeError("GraspGen sampler not loaded. "
                               "Call load_graspgen_sampler() first.")

        from scipy.spatial.transform import Rotation as R

        # Frame conversion: URDF ee_link→gripper_base has rpy=[0, π/2, 0]
        R_GRIP_TO_EE = R.from_euler('xyz', [0, np.pi / 2, 0]).inv()

        # ── 1. Build robot-frame point cloud ──
        pc_cam = self.env.camera.get_pointcloud()  # (H, W, 3) camera frame
        mask_flat = sam_mask.reshape(-1).astype(bool)
        obj_pc_cam = pc_cam.reshape(-1, 3)[mask_flat]

        if len(obj_pc_cam) < 50:
            print(f"[GraspGen] Too few points ({len(obj_pc_cam)}), skipping.")
            return []

        # Camera → robot base: negate y,z then apply extrinsic
        obj_pc = obj_pc_cam.copy()
        obj_pc[:, 1] = -obj_pc[:, 1]
        obj_pc[:, 2] = -obj_pc[:, 2]
        pc_hom = np.vstack([obj_pc.T, np.ones((1, len(obj_pc)))])
        pc_robot = (self.env.cam_to_robot_base @ pc_hom)[:3].T
        centroid = pc_robot.mean(axis=0)

        # ── 2. GraspGen inference ──
        from grasp_gen.grasp_server import GraspGenSampler
        grasps_tensor, confs_tensor = GraspGenSampler.run_inference(
            pc_robot.astype(np.float32),
            self.graspgen_sampler,
            grasp_threshold=-1.0,
            num_grasps=500,
            topk_num_grasps=200,
            min_grasps=20,
            max_tries=5,
        )
        if len(grasps_tensor) == 0:
            print("[GraspGen] No grasps generated.")
            return []

        grasps_np = grasps_tensor.cpu().numpy()  # (K, 4, 4)
        confs_np  = confs_tensor.cpu().numpy()    # (K,)

        # ── 3. Filter + 4. Frame conversion ──
        results = []
        for i in range(len(grasps_np)):
            pose     = grasps_np[i]
            pos      = pose[:3, 3]
            approach = pose[:3, 2]                   # GraspGen +Z = approach
            contact  = pos + self._GRIPPER_DEPTH * approach

            # Angle from vertical (world -Z)
            angle = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
            dist  = np.linalg.norm(contact - centroid)

            if (angle > self._MAX_APPROACH_DEG
                    or dist > self._MAX_CONTACT_DIST
                    or pos[2] < self._TABLE_Z):
                continue

            # Frame conversion: R_ee = R_graspgen @ Ry(-π/2)
            R_ee = R.from_matrix(pose[:3, :3]) * R_GRIP_TO_EE
            results.append(dict(
                pos=tuple(pos),
                quat_ee=tuple(R_ee.as_quat()),      # (x,y,z,w) for PyBullet
                conf=float(confs_np[i]),
                angle_vert=angle,
            ))

        # ── 5. Rank by confidence descending ──
        results.sort(key=lambda g: -g['conf'])

        n_raw = len(grasps_np)
        print(f"[GraspGen] {len(results)}/{n_raw} grasps after filter "
              f"(≤{self._MAX_APPROACH_DEG}°)"
              + (f", top conf={results[0]['conf']:.3f}" if results else ""))
        return results

    def setup_grasps(self,
                     obs: Dict[str, Any],
                     visualise_grasps: bool = False):
        """
        Run inference with GR-ConvNet grasp generator on current observation.

        Uses the camera pointcloud to recover exact 3D grasp poses instead of
        projecting through a pixel-to-meter conversion chain.

        Requires GR-ConvNet (``load_grasp_generator``).
        """
        if self.grasp_generator is None:
            print("[setup_grasps] GR-ConvNet not loaded; skipping.")
            return
        rgb, depth, seg = obs['image'], obs['depth'], obs['seg']
        img_size = self.grasp_generator.IMG_WIDTH

        # Get pointcloud in camera frame (HxWx3) for direct 3D lookup
        pointcloud_cam = self.env.camera.get_pointcloud()

        if img_size != self.env.camera.width:
            rgb = cv2.resize(rgb, (img_size, img_size))
            depth = cv2.resize(depth, (img_size, img_size))
            # Resize pointcloud to match GR-ConvNet input resolution.
            # Each channel (x, y, z) is resized independently with bilinear
            # interpolation so the 3D coordinates stay consistent with the
            # resized depth/rgb images.
            pointcloud_cam = cv2.resize(pointcloud_cam, (img_size, img_size),
                                        interpolation=cv2.INTER_LINEAR)

        for obj_id in self.env.obj_ids:
            mask = seg == obj_id
            if img_size != self.env.camera.width:
                mask = np.array(
                    Image.fromarray(mask).resize((img_size, img_size),
                                                 Image.LANCZOS))
            grasps, grasp_rects = self.grasp_generator.predict_grasp_from_mask_3d(
                rgb, depth, mask, pointcloud_cam,
                n_grasps=self.n_grasp_attempts, show_output=False)
            if img_size != self.env.camera.width:
                # normalize to original size
                for j, gr in enumerate(grasp_rects):
                    grasp_rects[j][0] = int(gr[0] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][1] = int(gr[1] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][4] = int(gr[4] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][3] = int(gr[3] / img_size *
                                            self.env.camera.width)
            grasp_rects = [
                Grasp2D.from_vector(
                    x=g[1],
                    y=g[0],
                    w=g[4],
                    h=g[3],
                    theta=g[2],
                    W=self.env.camera.width,
                    H=self.env.camera.width,
                    normalized=False,
                    line_offset=5,
                ) for g in grasp_rects
            ]
            self.env.set_obj_grasps(obj_id, grasps, grasp_rects)

        if visualise_grasps:
            LID = []
            for obj_id in self.env.obj_ids:
                grasps = self.env.get_obj_grasps(obj_id)
                color = np.random.rand(3).tolist()
                for g in grasps:
                    LID = self.env.draw_predicted_grasp(g,
                                                        color=color,
                                                        lineIDs=LID)

            time.sleep(1)
            self.env.remove_drawing(LID)

    def step(self, action):
        '''
        Wrapper around OWG action predictions and implemented robot primitives.

        Args:
          action: Predicted action by OWG 
            - `action`: Either `remove` to place blocking object in free space, or `pick` to put target in tray.
            - `input`: The object ID of object to manipulate.
        '''
        #for attempt in range(self.cfg.n_action_attempts):
        if action['action'] == 'remove':
            success_grasp, success_target = self.env.put_obj_in_free_space(
                action['input'], grasp_indices=action['grasps'])
        elif action['action'] == 'pick':
            success_grasp, success_target = self.env.put_obj_in_tray(
                action['input'], grasp_indices=action['grasps'])
        for _ in range(30):
            self.env.step_simulation()

        if not success_grasp or not success_target:
            print(f'Action failed...')
            success, done = False, False

        elif action['input'] != action['target_id']:
            # successfull action, but not terminal
            print(f'Done {action["action"]} {action["input"]}')
            success, done = True, False

        else:
            # successfull terminal action
            print(f'Done {action["action"]} {action["input"]}')
            success, done = True, True

        return success, done
    
# Sungsu's help functions
    def close(self):
        """Cleanly destroy the physics world and shut down any renderers."""
        self.env.camera.stop_recording()
        self.env.camera.disable_blender_rendering()
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass

    def set_objects_alpha(self, obj_ids, alpha: float):
        """
        Set alpha for all visual shapes (all links) of given objects.
        Returns a dict to restore original RGBA later.
        """
        restore = {}
        for oid in obj_ids:
            vs = p.getVisualShapeData(oid)
            # vs entries: (objectUniqueId, linkIndex, visualGeometryType, dimensions, filename, localInertialFramePos,
            #              localInertialFrameOrn, rgbaColor, specularColor)
            restore[oid] = []
            for entry in vs:
                link_idx = entry[1]
                rgba = entry[7]
                restore[oid].append((link_idx, rgba))
                new_rgba = (rgba[0], rgba[1], rgba[2], alpha)
                p.changeVisualShape(oid, link_idx, rgbaColor=new_rgba)
        return restore

    def restore_objects_rgba(self, restore_dict):
        for oid, link_rgba_list in restore_dict.items():
            for link_idx, rgba in link_rgba_list:
                p.changeVisualShape(oid, link_idx, rgbaColor=rgba)

    def pick_most_occluded_target(self):
        """
        Choose the most occluded object using seg masks:
        occlusion = 1 - visible_area / full_area_alone
        Returns: (target_name, stats_dict)

        Uses PyBullet-only rendering internally (fast) since it only needs
        segmentation masks, not photorealistic RGB.
        """
        # current visible seg (PyBullet — fast, ~5ms)
        _, _, seg_vis = self.env.camera.get_cam_img_pybullet()

        obj_ids = list(self.env.obj_ids)
        if not obj_ids:
            raise RuntimeError("No objects in scene.")

        # visible areas from current seg
        visible_area = {oid: int(np.count_nonzero(seg_vis == oid)) for oid in obj_ids}

        # Hide ALL objects once
        # store original RGBA so we can restore at the end
        restore_all = self.set_objects_alpha(obj_ids, alpha=0.0)

        # To get "full area", hide all other objects and render seg for each oid
        full_area = {}
        occlusion = {}

        try:
            for oid in obj_ids:
                # show only this object
                restore_one = self.set_objects_alpha([oid], alpha=1.0)

                # render seg with only this object visible (PyBullet — fast)
                _, _, seg_single = self.env.camera.get_cam_img_pybullet()
                full = int(np.count_nonzero(seg_single == oid))

                # hide this object again (back to 0)
                self.set_objects_alpha([oid], alpha=0.0)

                # guard against weird cases (object fully out of view)
                full_area[oid] = full
                if full <= 0:
                    occlusion[oid] = 0.0
                else:
                    occlusion[oid] = 1.0 - (visible_area[oid] / float(full))

        finally:
            # restore original RGBA for all objects no matter what
            self.restore_objects_rgba(restore_all)

        # pick max occlusion (tie-breaker: smaller visible area)
        best_id = max(obj_ids, key=lambda oid: (occlusion[oid], -visible_area[oid]))
        target_name = self.env.obj_names[self.env.obj_ids.index(best_id)]

        id_to_name_map = {
            int(oid): str(self.env.obj_names[self.env.obj_ids.index(oid)])
            for oid in obj_ids
        }
        name_to_id_map = {
            str(self.env.obj_names[self.env.obj_ids.index(oid)]): int(oid)
            for oid in obj_ids
        }

        stats = {
            "best_id": int(best_id),
            "best_name": str(target_name),
            "id_to_name": id_to_name_map,
            "name_to_id": name_to_id_map,
            "visible_area": {int(k): int(v) for k, v in visible_area.items()},
            "full_area": {int(k): int(v) for k, v in full_area.items()},
            "occlusion": {int(k): float(v) for k, v in occlusion.items()},
        }
        return target_name, stats

    def compute_infov_footprint(self, obj_ids=None) -> Dict[int, int]:
        """Return each object's UNOCCLUDED footprint (pixels) in the current
        top-down perception FoV.

        Uses the same alpha-trick as ``pick_most_occluded_target``: for each
        object, hide every other object and render the PyBullet seg so we get
        the object's own silhouette regardless of what is stacked on top of
        it.  An object that has been pushed off-frame, dropped off the table,
        or carried into the tray zone renders ~0 pixels here — whereas one
        that is merely *covered* by a still-present neighbour still returns a
        large footprint.  This makes it the right signal for V2 rearrangement
        "removed from view" scoring (compare against the spawn-time
        ``full_area`` with a tolerance).

        Args:
            obj_ids: optional subset of ids to MEASURE; defaults to all
                current ``env.obj_ids``.  Regardless of the subset, EVERY
                scene object is hidden first — otherwise a measured object
                that is covered by a *non-measured* neighbour (e.g. a mouse
                buried under a plate that is not in the subset) would render
                0 px and be wrongly treated as gone.

        Returns:
            {obj_id: unoccluded_footprint_pixels_in_current_fov}
        """
        all_ids = list(self.env.obj_ids)
        if obj_ids is None:
            measure_ids = all_ids
        else:
            measure_ids = [o for o in obj_ids if o in self.env.obj_ids]
        if not measure_ids:
            return {}

        footprint: Dict[int, int] = {}
        # Hide ALL scene objects (not just the subset) so each measured
        # object's silhouette is its own unoccluded footprint.
        restore_all = self.set_objects_alpha(all_ids, alpha=0.0)
        try:
            for oid in measure_ids:
                self.set_objects_alpha([oid], alpha=1.0)
                _, _, seg_single = self.env.camera.get_cam_img_pybullet()
                footprint[int(oid)] = int(np.count_nonzero(seg_single == oid))
                self.set_objects_alpha([oid], alpha=0.0)
        finally:
            self.restore_objects_rgba(restore_all)
        return footprint

    def compute_cover_hidden_pairs(self, occ_stats: Dict = None,
                                   min_reveal_area: int = 200,
                                   min_cover_fraction: float = 0.0) -> List[Dict]:
        """Find (cover, hidden) object pairs using alpha-transparency rendering.

        For each candidate cover object (visible, low occlusion), hide it and
        re-render the segmentation mask.  Any object whose visible area increases
        significantly was *under* the cover.

        Args:
            occ_stats: Pre-computed occlusion stats (from pick_most_occluded_target).
                       If None, computes fresh.
            min_reveal_area: Minimum pixel area gain to count as "revealed".
            min_cover_fraction: Minimum fraction of the hidden object's TOTAL
                       in-FoV footprint that must be occluded *by this cover*
                       for the pair to count.  ``cover_fraction`` is measured
                       as ``reveal_gain / full_footprint`` where full_footprint
                       is the object's own unoccluded silhouette (alpha-hide
                       all-others).  Default 0.0 = disabled (V1 behaviour —
                       any partial overlap ≥ ``min_reveal_area`` counts).  The
                       V2 "beneath" task sets this to 0.70 so a cover only
                       qualifies when it hides ≥70% of the target (a barely-
                       clipped edge no longer makes a valid pair).

        Returns:
            List of dicts, each: {
                "cover_id": int, "cover_name": str,
                "hidden_id": int, "hidden_name": str,
                "reveal_area": int,      # pixels of hidden obj newly visible
                "full_footprint": int,   # obj's own unoccluded silhouette px
                "cover_fraction": float, # reveal_area / full_footprint
            }
            Sorted by reveal_area descending (strongest pairs first).
        """
        obj_ids = list(self.env.obj_ids)
        if len(obj_ids) < 2:
            return []

        if occ_stats is None:
            _, occ_stats = self.pick_most_occluded_target()

        occlusion = occ_stats.get("occlusion", {})
        id_to_name = occ_stats.get("id_to_name", {})
        visible_area = occ_stats.get("visible_area", {})
        # Per-object UNOCCLUDED footprint (spawn-time silhouette, alpha-hide
        # all-others).  Denominator for the cover-fraction gate.
        full_area = occ_stats.get("full_area", {})

        # Candidate covers: objects that are reasonably visible (< 50% occluded)
        cover_candidates = [
            oid for oid in obj_ids
            if occlusion.get(oid, occlusion.get(str(oid), 1.0)) < 0.5
        ]

        if not cover_candidates:
            return []

        # Baseline: current visible area per object (PyBullet seg)
        _, _, seg_baseline = self.env.camera.get_cam_img_pybullet()
        baseline_vis = {
            oid: int(np.count_nonzero(seg_baseline == oid))
            for oid in obj_ids
        }

        pairs = []
        for cover_id in cover_candidates:
            # Hide the cover object
            restore = self.set_objects_alpha([cover_id], alpha=0.0)
            try:
                _, _, seg_without = self.env.camera.get_cam_img_pybullet()

                # Check which other objects gained visible area
                for oid in obj_ids:
                    if oid == cover_id:
                        continue
                    new_vis = int(np.count_nonzero(seg_without == oid))
                    gain = new_vis - baseline_vis.get(oid, 0)
                    if gain < min_reveal_area:
                        continue
                    # Cover-fraction gate: how much of the hidden object's
                    # OWN silhouette does this cover account for?  Uses the
                    # object's full unoccluded footprint as the denominator.
                    full_fp = int(full_area.get(oid, full_area.get(str(oid), 0)))
                    if full_fp <= 0:
                        # No silhouette baseline — fall back to gain as its
                        # own denominator (fraction = 1.0) so V1 behaviour is
                        # preserved when full_area is unavailable.
                        cover_fraction = 1.0
                    else:
                        cover_fraction = gain / float(full_fp)
                    if min_cover_fraction > 0.0 and cover_fraction < min_cover_fraction:
                        continue
                    cover_name = id_to_name.get(cover_id,
                                                id_to_name.get(str(cover_id)))
                    hidden_name = id_to_name.get(oid,
                                                 id_to_name.get(str(oid)))
                    if cover_name and hidden_name:
                        pairs.append({
                            "cover_id": int(cover_id),
                            "cover_name": str(cover_name),
                            "hidden_id": int(oid),
                            "hidden_name": str(hidden_name),
                            "reveal_area": int(gain),
                            "full_footprint": int(full_fp),
                            "cover_fraction": round(float(cover_fraction), 3),
                        })
            finally:
                self.restore_objects_rgba(restore)

        # Sort: largest reveal first (strongest spatial relationship)
        pairs.sort(key=lambda p: -p["reveal_area"])
        return pairs

    def run(self, target_name: str, base_path: str):    
        # error handler
        if not isinstance(base_path, str) or len(base_path.strip()) == 0:
            raise ValueError("base_path must be a non-empty string")
 
        # define output paths
        ep_dir = os.path.dirname(base_path)
        os.makedirs(ep_dir, exist_ok=True)

        img_path = f"{base_path}_image.png"
        vid_path = f"{base_path}_video.mp4"

        viz_dir = os.path.join(ep_dir, "inference_steps")
        os.makedirs(viz_dir, exist_ok=True)

        # save image
        rgb, _, _ = self.env.camera.get_cam_img()
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # start video recording
        self.env.camera.start_recording(vid_path)

        # ground-truth target id (from the simulator scene)
        target_name_prompt = target_name.replace("_", " ")
        try:
            gt_target_id = self.env.obj_ids[self.env.obj_names.index(target_name)]
        except ValueError:
            raise ValueError(f"target_name={target_name_prompt} not found in env.obj_names")
        
        # ask user for command
        self.env.reset_robot()

        # actual query
        # call OWG Policy to predict next action
        attempt = 0
        status = "fail"
        fail_reason = None
        steps = 0

        while True:
            grounding_path = os.path.join(viz_dir, f"step_{steps}_grounding.png")
            grasp_path     = os.path.join(viz_dir, f"step_{steps}_grasp_ranking.png")

            self.env.reset_robot()
            obs = self.update()

            # save a dict for all predicted grasps for policy grasp ranking
            if self.grasp_rank_3d:
                all_grasps = {
                    k: self.env.get_obj_grasps(k)
                    for k in self.env.obj_ids
                }
            else:
                all_grasps = {
                    k: self.env.get_obj_grasp_rects(k)
                    for k in self.env.obj_ids
                }

            action = self.policy.predict(obs, target_name_prompt, all_grasps, save_grounding_path=grounding_path, save_grasp_path=grasp_path)
            steps += 1

            if action['action'] == 'fail':
                fail_reason = action.get('reason', 'Unknown policy failure.')
                break
            # convert marker IDs to state IDs
            success, done = self.step(action)
            if success and done:
                # task finished
                # only count as success if it truly picked the requested target_name
                if action['input'] == gt_target_id:
                    status = "success"
                    fail_reason = None
                else:
                    status = "fail"
                    fail_reason = f"Terminate with incorrect object {action['input']}."
                break
            elif not success:
                attempt += 1
                if attempt == self.cfg.n_action_attempts:
                    print(f'Action failed. No more atempts.')
                    fail_reason = f"Maximum action attempts exceeded (attempts={attempt}, limit={self.cfg.n_action_attempts})."
                    break
                print(f'Action failed. {attempt} attempt. Retrying..')
                continue
            else:
                # plan step finished, move to next
                attempt = 0
                self.env.dummy_simulation_steps(30)
                continue
        
        return {
            "seed": int(self.seed),
            "n_objects": int(self.n_objects),
            "obj_names": list(self.env.obj_names),
            "target_name": str(target_name),
            "status": status,
            "steps": int(steps),
            "fail_reason": fail_reason,
        }