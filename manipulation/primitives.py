"""Contact-based grasp and push implementations shared by the public tools.

Relocated without physics changes from tools.py. Initial masks come from SAM3;
retry prompting may use simulator segmentation centroids (see re_perceive_object).
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple, Optional


def _sam_segment(*args, **kwargs):
    # Lazy import avoids a cycle with the legacy compatibility module.
    from tools import _sam_segment as segment
    return segment(*args, **kwargs)

# =====================================================================
#  MANIPULATION TOOLS — Grasp (GraspGen hybrid) & Push
# =====================================================================
#
# Implements the 6-DOF GraspGen pipeline (test_graspgen.py) and the
# directional push tool (test_push.py) for use in the agentic loop.
#
# ALL masks come from SAM3 — no GT segmentation is used.
#   - Grasp: initial SAM mask from perception; re-perception between
#     attempts uses SAM3 with a point prompt at the previous centroid.
#   - Push: target mask from perception; "occupied space" built from
#     the union of all SAM masks in the last perception step.
#
# Grasp: 3 × 6-DOF attempts per object.  GraspGen diffusion →
#   geometric filter (angle/contact/table) → angle-first ranking,
#   mask dilation, re-perceive per attempt.  No collision filter and
#   no IK-abort (both hurt performance in dense clutter — see
#   GRASPING_EXPERIMENTS §3 / valts comparison).
#
# Frame conversion:  R_ee = R_graspgen × R_GRIP_TO_EE
#   with  R_GRIP_TO_EE = Rz(−π/2) · Ry(−π/2)
# (Replaces the older "Ry(−π/2) inverse + 90° Z finger offset" combo
#  which was mathematically equivalent but ad-hoc; the Rz·Ry form
#  explains the mapping as: GraspGen mesh frame → Robotiq URDF base
#  frame (Rz −π/2, because URDF has Y = finger-open, GraspGen has
#  X = finger-open) → EE frame (Ry −π/2 from arm_gripper_joint).)
#
# Push: closed-gripper sweep through object in image-relative direction,
#   descent point found via segmentation ray-trace over SAM masks.
# =====================================================================

from scipy.spatial.transform import Rotation as _R
import pybullet as _pb

# ── Grasp constants ──────────────────────────────────────────────────

_GRIPPER_DEPTH      = 0.195     # Robotiq 2F-140 base → fingertip (m)
_TABLE_Z            = 0.785     # table surface z in robot frame (m)
_MAX_APPROACH_DEG   = 60.0      # max angle from vertical for 6-DOF (°)
_MAX_CONTACT_DIST   = 0.08      # max contact-to-centroid distance (m)
_MIN_GRASP_CONF     = 0.80      # skip grasps below this confidence
_MASK_BORDER_FRAC   = 0.10      # dilate mask by 10% of bbox diagonal

_N_4DOF_ATTEMPTS    = 0         # disabled (§12: 4-DOF second attempt never works)
_N_6DOF_ATTEMPTS    = 2         # 2 attempts (attempt 2 only recovers 6/88 grasps)
_SETTLE_STEPS       = 60        # physics steps between attempts

# Post-release settle: after transport/release, let the object come to
# rest so a bounce-out-of-bin or slip-during-transport is reflected in
# its final resting position before we test whether it is in the bin.
_POST_RELEASE_SETTLE     = 40   # extra physics steps after gripper opens

# GraspGen-frame → EE-frame conversion (Valts's corrected transform).
#   R_ee = R_graspgen · R_GRIP_TO_EE
#   R_GRIP_TO_EE = Rz(−π/2) · Ry(−π/2)
# No separate finger-offset multiplication is applied.
_R_GRIP_TO_EE = _R.from_euler('z', -np.pi / 2) * _R.from_euler('y', -np.pi / 2)

# Friction boost
_FINGER_LAT_FRIC  = 2.0
_FINGER_SPIN_FRIC = 0.5
_FINGER_ROLL_FRIC = 0.01
_OBJ_LAT_FRIC     = 1.5
_OBJ_SPIN_FRIC    = 0.3
_OBJ_ROLL_FRIC    = 0.01

# ── Push constants ───────────────────────────────────────────────────

_PUSH_DISTANCE      = 0.10     # sweep distance from descent point (m)
_MIN_CLEARANCE_PX   = 60       # pixels of clear space for gripper descent (~10cm)
_MAX_PUSH_ATTEMPTS  = 2

# Direction mapping: image direction → world-frame unit vector
#   Image left  → world −X,  Image right → world +X
#   Image up    → world +Y,  Image down  → world −Y
_DIRECTION_VECTORS = {
    "left":  np.array([-1.0,  0.0,  0.0]),
    "right": np.array([ 1.0,  0.0,  0.0]),
    "up":    np.array([ 0.0,  1.0,  0.0]),
    "down":  np.array([ 0.0, -1.0,  0.0]),
}

# Anti-push direction in image coords (row_step, col_step).
# Used to trace away from object to find empty descent space.
_ANTI_PUSH_IMAGE = {
    "left":  ( 0,  1),   # push left  → trace right
    "right": ( 0, -1),
    "up":    ( 1,  0),   # push up    → trace down
    "down":  (-1,  0),
}

# Finger yaw: wide side of closed gripper faces push direction
_PUSH_FINGER_YAW = {
    "left":  np.pi / 2,
    "right": np.pi / 2,
    "up":    0.0,
    "down":  0.0,
}


def set_finger_offset(degrees: float):
    """Deprecated.  No-op retained for backward compatibility.

    The grasp pipeline now uses Valts's corrected ``R_GRIP_TO_EE``
    (``Rz(−π/2) · Ry(−π/2)``) which already produces the correct finger
    alignment for the Robotiq 2F-140 — no additional rotation is needed.
    This function is kept so older call sites do not break, but any
    non-zero value is ignored with a warning.
    """
    if abs(degrees) > 1e-6:
        print(f"[tools] warning: --finger_offset_deg={degrees}° is ignored; "
              f"the new R_GRIP_TO_EE already handles finger alignment.")


# ── SAM-based re-perception ─────────────────────────────────────────

def build_sam_occupancy(detections: List[dict], shape: Tuple[int, int]) -> np.ndarray:
    """Build a combined binary occupancy mask from all SAM detection masks.

    Args:
        detections:  List of detection dicts from perceive_scene/find/obstructors,
                     each with a ``"mask"`` key (H, W) bool array.
        shape:       (H, W) of the output mask.

    Returns:
        (H, W) uint8 mask where 1 = occupied by some detected object.
    """
    occ = np.zeros(shape, dtype=np.uint8)
    for det in detections:
        m = det.get("mask")
        if m is not None:
            occ[m.astype(bool)] = 1
    return occ


def re_perceive_object(
    camera,
    sam_model: SAM,
    prev_mask: np.ndarray,
    obj_id: int = None,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """Get a fresh SAM3 mask for an object after the scene may have changed.

    Called between grasp attempts.  A failed grasp can nudge the object,
    making the previous SAM mask centroid stale.  To handle this:

    1. If ``obj_id`` is provided, we use the **GT segmentation buffer**
       from PyBullet (returned by ``get_cam_img()``) to find the object's
       *current* pixel centroid.  This is the only GT information used —
       the actual mask still comes from SAM3.
    2. If the GT seg doesn't contain the object (e.g. it fell off the table
       or is fully occluded), we fall back to the previous mask centroid.

    Args:
        camera:     Camera object (has ``get_cam_img()``).
        sam_model:  Loaded SAM3 model.
        prev_mask:  (H, W) bool/uint8 mask from the previous attempt.
        obj_id:     PyBullet body ID of the target object (optional).
                    When provided, GT seg is used to locate the object's
                    current position for a more accurate SAM3 prompt.
        verbose:    Print diagnostics.

    Returns:
        (H, W) bool mask, or None if re-segmentation failed.
    """
    rgb, _, seg = camera.get_cam_img()
    H, W = rgb.shape[:2]

    cx, cy = None, None
    source = None

    # Strategy 1: use GT seg to find current object centroid
    if obj_id is not None and seg is not None:
        gt_mask = (seg == obj_id)
        gt_ys, gt_xs = np.where(gt_mask)
        if len(gt_ys) >= 5:
            cy, cx = int(np.mean(gt_ys)), int(np.mean(gt_xs))
            source = "gt_seg"

    # Strategy 2: fall back to previous SAM mask centroid
    if cx is None:
        ys, xs = np.where(prev_mask.astype(bool))
        if len(ys) >= 5:
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            source = "prev_mask"

    if cx is None:
        if verbose:
            print("  [re-perceive] Cannot locate object — "
                  "GT seg empty and prev mask too small")
        return None

    if verbose:
        print(f"  [re-perceive] Prompting SAM3 at ({cx},{cy}) via {source}")

    # SAM3 point prompt at located centroid
    mask = _sam_segment(sam_model, rgb, cx, cy, verbose=verbose,
                        label="re-perceive")
    if mask is not None and int(mask.sum()) >= 20:
        return mask
    if verbose:
        print(f"  [re-perceive] SAM3 returned no mask at ({cx}, {cy})")
    return None


# ── Point cloud helpers ──────────────────────────────────────────────

# Module-level cap on point-cloud size fed to GraspGen.  Tunable at startup
# via ``set_num_pc_points(...)`` (driven by ``--num_pc_points`` CLI flag).
#
# Default 4000 matches GraspGen's published demos.  At img_size=1024, a
# typical dilated mask has 20–80 k pixels; passing the full cloud causes
# ``torch.cdist(X, X)`` inside GraspGen's knn_points to blow up with
# ``CUDA error: invalid configuration argument`` (observed ep_010 of
# rollout_vlm_20260417_030531).
_MAX_PC_POINTS = 4000


def set_num_pc_points(n: int) -> None:
    """Set the hard ceiling on point-cloud size passed to GraspGen.

    Called once at startup from ``demo.py`` with the value of
    ``--num_pc_points``.  Values ≤ 0 disable the subsample (not recommended —
    CUDA cdist will crash on dense masks at 1024²).
    """
    global _MAX_PC_POINTS
    _MAX_PC_POINTS = int(n)


def _build_pc_from_mask(camera, cam_to_robot_base, mask):
    """Binary mask + depth → robot-frame point cloud (N, 3).

    If the masked cloud has more than ``_MAX_PC_POINTS`` points, a random
    subsample (no replacement) is taken before coordinate-frame conversion.
    Uses a local ``numpy.random`` Generator seeded per-call from the mask's
    hash so repeated calls on identical inputs are deterministic.
    """
    pc_cam = camera.get_pointcloud()           # (H, W, 3) OpenCV convention
    mask_flat = mask.reshape(-1).astype(bool)
    obj_cam = pc_cam.reshape(-1, 3)[mask_flat]
    if len(obj_cam) < 10:
        return np.zeros((0, 3))

    # ── Subsample to keep GraspGen's cdist tractable ──
    if _MAX_PC_POINTS > 0 and len(obj_cam) > _MAX_PC_POINTS:
        # Deterministic per-mask seed — keeps identical inputs reproducible
        # while still decorrelating across calls within an episode.
        seed = int(mask_flat.sum()) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(obj_cam), size=_MAX_PC_POINTS, replace=False)
        obj_cam = obj_cam[idx]

    obj = obj_cam.copy()
    obj[:, 1] = -obj[:, 1]                    # OpenCV→world Y flip
    obj[:, 2] = -obj[:, 2]                    # OpenCV→world Z flip
    hom = np.vstack([obj.T, np.ones((1, len(obj)))])
    return (cam_to_robot_base @ hom)[:3].T


def _dilate_mask(mask, border_frac):
    """Dilate binary mask by fraction of bbox diagonal."""
    if border_frac <= 0:
        return mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return mask
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    diag = np.sqrt((r_max - r_min + 1)**2 + (c_max - c_min + 1)**2)
    radius = max(1, int(round(border_frac * diag)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def _robot_to_pixel(pos_robot, cam_to_robot_base, intrinsics):
    """Project robot-frame 3D point to image pixel (row, col)."""
    T_inv = np.linalg.inv(cam_to_robot_base)
    p = (T_inv @ np.append(pos_robot, 1.0))[:3]
    x_cv, y_cv, z_cv = p[0], -p[1], -p[2]
    if z_cv <= 0:
        return None
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * x_cv / z_cv + cx
    v = fy * y_cv / z_cv + cy
    return int(round(v)), int(round(u))


def _pixel_to_world(camera, cam_to_robot_base, row, col):
    """Single image pixel → world (x, y, z) via point cloud."""
    pc_cam = camera.get_pointcloud()
    pt = pc_cam[row, col].copy()
    pt[1] = -pt[1]
    pt[2] = -pt[2]
    hom = np.array([pt[0], pt[1], pt[2], 1.0])
    return (cam_to_robot_base @ hom)[:3]


# ── GraspGen inference ───────────────────────────────────────────────

def _run_graspgen(pc_robot, sampler):
    """Run GraspGen diffusion → (K, 4, 4) poses + (K,) confidences."""
    # Support both direct GraspGenSampler and RPC client (for multi-env setups).
    try:
        from graspgen_client import GraspGenSamplerRPC
        _is_rpc = isinstance(sampler, GraspGenSamplerRPC)
    except ImportError:
        _is_rpc = False

    if _is_rpc:
        g, c, _ = sampler.sample(
            pc_robot.astype(np.float32),
            threshold=-1.0,
            num_grasps=500,
            topk_num_grasps=200,
            min_grasps=20,
            max_tries=5,
        )
    else:
        from grasp_gen.grasp_server import GraspGenSampler
        g, c = GraspGenSampler.run_inference(
            pc_robot.astype(np.float32), sampler,
            grasp_threshold=-1.0,
            num_grasps=500,
            topk_num_grasps=200,
            min_grasps=20,
            max_tries=5,
        )
    if len(g) == 0:
        return np.zeros((0, 4, 4)), np.zeros((0,))
    return g.cpu().numpy(), c.cpu().numpy()


# ── Grasp geometric filters ─────────────────────────────────────────

def _make_topdown_quat(finger_yaw):
    """EE quaternion for straight-down approach with given finger yaw."""
    c, s = np.cos(finger_yaw), np.sin(finger_yaw)
    R_grip = np.array([[ c,  s,  0],
                        [ s, -c,  0],
                        [ 0,  0, -1]])
    R_ee = _R.from_matrix(R_grip) * _R_GRIP_TO_EE
    return tuple(R_ee.as_quat())


def _filter_6dof(grasps_np, confs_np, centroid):
    """6-DOF filter: angle ≤ 60°, contact ≤ 8 cm from centroid, above table."""
    results = []
    for i in range(len(grasps_np)):
        pose = grasps_np[i]
        pos = pose[:3, 3]
        approach = pose[:3, 2]
        contact = pos + _GRIPPER_DEPTH * approach
        angle = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
        dist = np.linalg.norm(contact - centroid)
        if angle > _MAX_APPROACH_DEG or dist > _MAX_CONTACT_DIST or pos[2] < _TABLE_Z:
            continue
        # R_ee = R_graspgen · R_GRIP_TO_EE (no separate finger offset —
        # R_GRIP_TO_EE = Rz(−π/2) · Ry(−π/2) already handles finger
        # alignment for the Robotiq 2F-140).
        R_ee = _R.from_matrix(pose[:3, :3]) * _R_GRIP_TO_EE
        results.append(dict(
            pos=tuple(pos), quat_ee=tuple(R_ee.as_quat()),
            conf=float(confs_np[i]), angle_vert=float(angle),
            contact_dist=float(dist), dof=6,
        ))
    return results


def _filter_4dof(grasps_np, confs_np, centroid,
                 mask, cam_to_robot_base, intrinsics):
    """4-DOF filter: mask projection, horizontal distance, above table.

    The mask here is the SAM3-predicted mask (not GT seg).
    """
    H, W = mask.shape[:2]
    results = []
    for i in range(len(grasps_np)):
        pose = grasps_np[i]
        pos = pose[:3, 3]
        if pos[2] < _TABLE_Z:
            continue
        if np.linalg.norm(pos[:2] - centroid[:2]) > _MAX_CONTACT_DIST:
            continue
        px = _robot_to_pixel(pos, cam_to_robot_base, intrinsics)
        if px is None:
            continue
        row, col = px
        if not (0 <= row < H and 0 <= col < W and mask[row, col] > 0):
            continue
        finger_dir = pose[:3, 0]
        # No finger offset — R_GRIP_TO_EE (Rz(−π/2) · Ry(−π/2)) already
        # aligns the Robotiq 2F-140 fingers correctly.
        finger_yaw = float(np.arctan2(finger_dir[1], finger_dir[0]))
        results.append(dict(
            pos=tuple(pos), quat_ee=_make_topdown_quat(finger_yaw),
            conf=float(confs_np[i]), angle_vert=0.0,
            contact_dist=float(np.linalg.norm(pos[:2] - centroid[:2])),
            finger_yaw=finger_yaw, dof=4,
        ))
    return results


def _rank_angle_first(grasps):
    """Angle-first ranking: most vertical first, confidence tiebreaker."""
    if not grasps:
        return None
    return sorted(grasps, key=lambda g: (g["angle_vert"], -g["conf"]))[0]


# ── Friction boost ───────────────────────────────────────────────────

def _boost_friction(env, obj_id):
    """Boost gripper pad + object friction for stable grasps."""
    for pad in ("left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint"):
        link_id = env.joints[pad].id
        _pb.changeDynamics(env.robot_id, link_id,
                           lateralFriction=_FINGER_LAT_FRIC,
                           spinningFriction=_FINGER_SPIN_FRIC,
                           rollingFriction=_FINGER_ROLL_FRIC)
    _pb.changeDynamics(obj_id, -1,
                       lateralFriction=_OBJ_LAT_FRIC,
                       spinningFriction=_OBJ_SPIN_FRIC,
                       rollingFriction=_OBJ_ROLL_FRIC)


# ── FSM grasp execution ─────────────────────────────────────────────

def _fsm_grasp(env, grasp, obj_id):
    """Execute one grasp attempt.  Returns (grasped: bool, placed: bool).

    The two bits are strictly physical and independent:
      - ``grasped`` : after the +20 cm lift the object was still in the
                      gripper (``check_grasped_id``).
      - ``placed``  : after transport + release + a post-release settle the
                      object is resting inside the tray/bin AABB
                      (``check_target_reached``).

    So the three real outcomes are ``(False, False)`` = grasp failed,
    ``(True, False)`` = grasped & lifted but slipped before reaching the
    bin (object is still somewhere in the scene), and ``(True, True)`` =
    genuinely deposited in the bin.  The caller turns these into the
    planner-facing "succeeded / grasped-but-slipped / failed" message.
    """
    bx, by, bz = grasp["pos"]
    orn = grasp["quat_ee"]
    tz = env.TARGET_ZONE_POS
    z_grasp = np.clip(bz, *env.ee_position_limit[2])

    # Fingertip-floor guard (FIXES_FOR_April18 P1-3): keep the fingertip at
    # least 0.5 cm above the table surface so the descent can't aim below
    # it even for very low grasps.  Since ``bz`` is the gripper-base z and
    # the fingertip is ``_GRIPPER_DEPTH`` below that, the minimum base-z
    # is ``_TABLE_Z + 0.005 + _GRIPPER_DEPTH``.
    _Z_GRASP_MIN = _TABLE_Z + 0.005 + _GRIPPER_DEPTH
    if z_grasp < _Z_GRASP_MIN:
        # Don't silently clamp without a trace — this usually signals a
        # low-quality grasp (finger-below-table) that should be noted.
        print(f"    [_fsm_grasp] z_grasp={z_grasp:.3f} below floor "
              f"{_Z_GRASP_MIN:.3f} — clamping (fingertip would be below table)")
        z_grasp = _Z_GRASP_MIN

    # OPEN → APPROACH → DESCEND
    env.move_gripper(0.085)
    env.move_ee([bx, by, env.GRIPPER_MOVING_HEIGHT, orn])
    env.move_ee([bx, by, z_grasp, orn])

    # Safety check: if the EE ended up far above the target z (>3 cm),
    # the grasp is hopeless (IK couldn't reach the pose) — abort early
    # instead of closing gripper on air and wasting time on lift/transport.
    try:
        import pybullet as _p
        ee_pos, _ = _p.getLinkState(env.robot_id, env.eef_id)[0:2]
        ik_dz = float(ee_pos[2] - z_grasp)
        if ik_dz > 0.03:
            print(f"\n    [_fsm_grasp] EE parked {ik_dz*100:.1f} cm above "
                  f"target z ({ee_pos[2]:.3f} vs {z_grasp:.3f}) — aborting")
            return False, False
    except Exception:
        pass

    # CLOSE → SETTLE
    env.auto_close_gripper(check_contact=True)
    for _ in range(40):
        env.step_simulation()

    # LIFT (+20 cm)
    env.move_ee([bx, by, z_grasp + 0.20, orn])
    for _ in range(40):
        env.step_simulation()

    # CHECK
    grasped_ids = env.check_grasped_id()
    if not (len(grasped_ids) == 1 and grasped_ids[0] == obj_id):
        return False, False

    # TRANSPORT → RELEASE → RETRACT
    env.move_ee([tz[0], tz[1], 1.25, orn])
    env.move_ee([tz[0], tz[1], tz[2] + 0.25, orn])
    env.move_gripper(0.085)
    env.move_ee([tz[0], tz[1], env.GRIPPER_MOVING_HEIGHT, orn])
    for _ in range(20):
        env.step_simulation()

    # Extra post-release settle: let the object come to rest so a
    # bounce-out-of-tray or slip-during-transport is reflected in its
    # final resting position before we judge whether it is in the bin.
    for _ in range(_POST_RELEASE_SETTLE):
        env.step_simulation()

    # ``placed`` is strictly "resting inside the bin AABB".  If the object
    # slipped mid-transport it is NOT in the bin here, so this returns
    # (True, False) and the caller reports "grasped but may have slipped".
    placed = env.check_target_reached(obj_id)
    return True, placed


# ── Full grasp-and-remove pipeline ───────────────────────────────────

def execute_grasp_and_remove(
    demo,
    obj_id: int,
    sam_mask: np.ndarray,
    sam_model: SAM,
    graspgen_sampler,
    verbose: bool = True,
) -> dict:
    """Run the 3 × 6-DOF GraspGen pipeline using SAM3 masks.

    Re-perceives with SAM3 (point prompt at previous centroid) before
    each retry.  Grasps are filtered by approach angle (≤ 60° from
    vertical), contact-to-centroid distance (≤ 8 cm), and table
    clearance, then ranked angle-first (most vertical grasps first,
    confidence as tiebreaker).  The corrected ``R_GRIP_TO_EE`` transform
    (``Rz(−π/2) · Ry(−π/2)``) is used — no finger-offset workaround
    is needed.  No collision filter or IK-abort (both hurt performance
    in dense clutter; see GRASPING_EXPERIMENTS §3 / valts comparison).

    Args:
        demo:              RobotEnvUI instance.
        obj_id:            PyBullet body ID of the target object.
        sam_mask:          (H, W) bool/uint8 SAM3 mask from the perception step.
        sam_model:         Loaded SAM3 model for re-perception.
        graspgen_sampler:  Loaded GraspGenSampler.
        verbose:           Print per-attempt diagnostics.

    Returns:
        dict: success (placed in tray), grasped (lifted), n_attempts,
              attempt_details.
    """
    env = demo.env
    camera = env.camera
    cam2rb = env.cam_to_robot_base
    intrinsics = camera.intrinsics

    _boost_friction(env, obj_id)

    total_attempts = _N_4DOF_ATTEMPTS + _N_6DOF_ATTEMPTS
    attempt_details = []
    grasped = False
    placed = False
    current_mask = sam_mask.astype(np.uint8) if sam_mask.dtype != np.uint8 else sam_mask

    for attempt in range(total_attempts):
        dof = 4 if attempt < _N_4DOF_ATTEMPTS else 6
        n_pixels = int(current_mask.astype(bool).sum())

        detail = dict(attempt=attempt, dof=dof, n_pixels=n_pixels,
                      exec_result="no_feasible_grasps", n_filtered=0)

        if n_pixels < 20:
            detail["exec_result"] = "barely_visible"
            attempt_details.append(detail)
            if verbose:
                print(f"    attempt {attempt} [{dof}D]: barely visible ({n_pixels}px)")
            for _ in range(_SETTLE_STEPS):
                env.step_simulation()
            # Re-perceive for next attempt
            new_mask = re_perceive_object(camera, sam_model, current_mask, obj_id, verbose)
            if new_mask is not None:
                current_mask = new_mask.astype(np.uint8)
            continue

        # ── Build point clouds from SAM mask ──
        pc_object = _build_pc_from_mask(camera, cam2rb, current_mask)
        if len(pc_object) < 50:
            detail["exec_result"] = "too_few_points"
            attempt_details.append(detail)
            for _ in range(_SETTLE_STEPS):
                env.step_simulation()
            new_mask = re_perceive_object(camera, sam_model, current_mask, obj_id, verbose)
            if new_mask is not None:
                current_mask = new_mask.astype(np.uint8)
            continue

        centroid = pc_object.mean(0)
        dilated = _dilate_mask(current_mask, _MASK_BORDER_FRAC)
        pc_context = _build_pc_from_mask(camera, cam2rb, dilated)

        # Guard: GraspGen's forward_inference does
        #     depth.reshape([-1, num_points, 3])
        # which crashes the entire process with
        #     "cannot reshape tensor of 0 elements"
        # when ``pc_context`` has 0 points (rare but can happen if the
        # dilated mask drifts onto background-only / out-of-frustum
        # pixels).  Treat as a soft attempt failure instead of letting
        # the RuntimeError escape and kill the episode.
        if len(pc_context) < 10:
            detail["exec_result"] = "too_few_context_points"
            attempt_details.append(detail)
            if verbose:
                print(f"    attempt {attempt} [{dof}D]: skipping — context PC has "
                      f"{len(pc_context)} points (< 10)")
            for _ in range(_SETTLE_STEPS):
                env.step_simulation()
            new_mask = re_perceive_object(camera, sam_model, current_mask, obj_id, verbose)
            if new_mask is not None:
                current_mask = new_mask.astype(np.uint8)
            continue

        # ── GraspGen inference ──
        # Wrapped in try/except: CUDA kernel-launch failures inside
        # GraspGen's cdist (if the subsample cap were ever disabled, or on
        # transient driver hiccups) should abort *this attempt* only, not
        # the whole episode.  We also clear the CUDA cache between attempts
        # to prevent slow memory growth across a long rollout.
        try:
            grasps_raw, confs_raw = _run_graspgen(pc_context, graspgen_sampler)
        except RuntimeError as e:
            msg = str(e)
            # Treat ANY GraspGen RuntimeError as a soft attempt failure.
            # Originally we only caught CUDA-prefixed errors, but
            # ``cannot reshape tensor of 0 elements`` (zero-point PC, which
            # already has an upstream guard now but might still happen via
            # other code paths) used to crash whole episodes.  Any
            # GraspGen-internal failure should kill the attempt only.
            is_cuda = ("CUDA" in msg or "cuda" in msg or "cudnn" in msg)
            tag = "cuda_error" if is_cuda else "graspgen_runtime_error"
            if True:
                if verbose:
                    label = "CUDA" if is_cuda else "RuntimeError"
                    print(f"    attempt {attempt} [{dof}D]: GraspGen {label} — "
                          f"skipping ({msg.splitlines()[0][:120]})")
                detail["exec_result"] = tag
                attempt_details.append(detail)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                for _ in range(_SETTLE_STEPS):
                    env.step_simulation()
                new_mask = re_perceive_object(camera, sam_model, current_mask, obj_id, verbose)
                if new_mask is not None:
                    current_mask = new_mask.astype(np.uint8)
                continue
            # (unreachable: every RuntimeError above hits ``continue``)

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if len(grasps_raw) == 0:
            attempt_details.append(detail)
            for _ in range(_SETTLE_STEPS):
                env.step_simulation()
            new_mask = re_perceive_object(camera, sam_model, current_mask, obj_id, verbose)
            if new_mask is not None:
                current_mask = new_mask.astype(np.uint8)
            continue

        # ── Filter (using SAM mask, not GT) ──
        if dof == 4:
            filtered = _filter_4dof(grasps_raw, confs_raw, centroid,
                                    current_mask, cam2rb, intrinsics)
        else:
            filtered = _filter_6dof(grasps_raw, confs_raw, centroid)
        detail["n_filtered"] = len(filtered)

        if not filtered:
            attempt_details.append(detail)
            if verbose:
                print(f"    attempt {attempt} [{dof}D]: 0/{len(grasps_raw)} "
                      f"survived filter")
            for _ in range(_SETTLE_STEPS):
                env.step_simulation()
            new_mask = re_perceive_object(camera, sam_model, current_mask, obj_id, verbose)
            if new_mask is not None:
                current_mask = new_mask.astype(np.uint8)
            continue

        # ── Rank → top-1 ──
        best = _rank_angle_first(filtered)
        detail["grasp_angle"] = best["angle_vert"]
        detail["grasp_conf"] = best["conf"]

        if verbose:
            print(f"    attempt {attempt} [{dof}D]: "
                  f"{len(filtered)}/{len(grasps_raw)} grasps, "
                  f"angle={best['angle_vert']:.1f}° "
                  f"conf={best['conf']:.3f}", end="", flush=True)

        # ── Confidence gate — skip low-quality grasps ──
        if best["conf"] < _MIN_GRASP_CONF:
            detail["exec_result"] = "low_confidence"
            attempt_details.append(detail)
            if verbose:
                print(f" → skipped (conf < {_MIN_GRASP_CONF})")
            for _ in range(_SETTLE_STEPS):
                env.step_simulation()
            new_mask = re_perceive_object(camera, sam_model, current_mask, obj_id, verbose)
            if new_mask is not None:
                current_mask = new_mask.astype(np.uint8)
            continue

        # ── Execute ──
        if attempt > 0:
            env.reset_robot()
            for _ in range(10):
                env.step_simulation()
            _boost_friction(env, obj_id)

        g, p = _fsm_grasp(env, best, obj_id)

        if g and p:
            detail["exec_result"] = "placed"
            grasped, placed = True, True
            if verbose:
                print(" → PLACED ✓")
        elif g:
            detail["exec_result"] = "grasped_slipped"
            grasped = True
            if verbose:
                print(" → GRASPED but slipped before bin (not removed)")
        else:
            detail["exec_result"] = "no_contact"
            if verbose:
                print(" → no contact")

        attempt_details.append(detail)

        if g:   # grasped (placed or dropped) — stop retrying
            break

        # Settle + re-perceive before next attempt
        for _ in range(_SETTLE_STEPS):
            env.step_simulation()
        new_mask = re_perceive_object(camera, sam_model, current_mask, obj_id, verbose)
        if new_mask is not None:
            current_mask = new_mask.astype(np.uint8)

    return dict(
        success=placed,
        grasped=grasped,
        n_attempts=len(attempt_details),
        attempt_details=attempt_details,
    )


# ── Push helpers ─────────────────────────────────────────────────────

def _find_push_params(camera, cam_to_robot_base, target_mask, occupancy_mask,
                      direction):
    """Find descent point via ray-trace over SAM occupancy mask.

    Traces from the target object centroid in the anti-push direction.
    Uses ``occupancy_mask`` (union of all SAM masks from last perception)
    to detect occupied pixels, avoiding descent into any object.

    Returns dict with descent_world, z_push, centroid_world,
    or None if no clear space found.
    """
    H, W = occupancy_mask.shape[:2]

    # Target centroid in image
    ys, xs = np.where(target_mask.astype(bool))
    n_pixels = len(ys)
    if n_pixels < 20:
        return None
    cr, cc = int(np.mean(ys)), int(np.mean(xs))

    # Target centroid in world (for z)
    obj_pc = _build_pc_from_mask(camera, cam_to_robot_base,
                                 target_mask.astype(np.uint8))
    if len(obj_pc) < 10:
        return None
    centroid_world = obj_pc.mean(0)

    # Ray-trace in anti-push direction over the SAM occupancy mask
    dr, dc = _ANTI_PUSH_IMAGE[direction]
    clear_count = 0
    edge_step = None
    descent_r, descent_c = None, None

    for step in range(1, max(H, W)):
        r = cr + dr * step
        c = cc + dc * step
        if r < 0 or r >= H or c < 0 or c >= W:
            break

        if occupancy_mask[r, c] > 0:      # pixel occupied by some object
            clear_count = 0
            edge_step = None
        else:
            if edge_step is None:
                edge_step = step
            clear_count += 1
            if clear_count >= _MIN_CLEARANCE_PX:
                mid = _MIN_CLEARANCE_PX // 2
                descent_r = cr + dr * (edge_step + mid)
                descent_c = cc + dc * (edge_step + mid)
                break

    if descent_r is None:
        return None

    descent_r = int(np.clip(descent_r, 0, H - 1))
    descent_c = int(np.clip(descent_c, 0, W - 1))

    descent_world = _pixel_to_world(camera, cam_to_robot_base,
                                    descent_r, descent_c)

    # Push z — place the fingertip 3 cm above the table, OR just above the
    # object's top if the object is taller than 3 cm (so the closed
    # gripper sweeps *over* the object surface, not through its middle).
    #
    # Previously we used ``centroid_world[2] + _GRIPPER_DEPTH`` which put
    # the fingertip at the object's z-centroid — half-height for tall
    # objects, below-table for flat ones.  See FIXES_FOR_April18 P1-3.
    table_z = float(obj_pc[:, 2].min())           # lowest seen point ≈ table
    obj_top = float(obj_pc[:, 2].max())
    obj_height = max(0.0, obj_top - table_z)
    # Fingertip above table: 3 cm clearance, or object_top + small margin
    fingertip_z = table_z + max(0.03, obj_height + 0.005)
    z_push = fingertip_z + _GRIPPER_DEPTH

    return dict(
        descent_world=descent_world,
        z_push=z_push,
        centroid_world=centroid_world,
    )


def _fsm_push(env, descent_xy, z_push, push_direction, push_yaw,
              push_distance):
    """Close gripper → descend into empty space → sweep through object."""
    dx, dy = descent_xy
    z = np.clip(z_push, *env.ee_position_limit[2])
    orn = _make_topdown_quat(push_yaw)
    d = push_direction

    ex = dx + d[0] * push_distance
    ey = dy + d[1] * push_distance

    # Close gripper (paddle mode)
    env.move_gripper(0.0)
    for _ in range(30):
        env.step_simulation()

    # Move above descent point → descend → sweep → lift
    env.move_ee([dx, dy, env.GRIPPER_MOVING_HEIGHT, orn])
    env.move_ee([dx, dy, z, orn])
    env.move_ee([ex, ey, z, orn])
    for _ in range(40):
        env.step_simulation()
    env.move_ee([ex, ey, env.GRIPPER_MOVING_HEIGHT, orn])


# ── Full push pipeline ───────────────────────────────────────────────

def execute_push(
    demo,
    obj_id: int,
    direction: str,
    target_mask: np.ndarray,
    occupancy_mask: np.ndarray,
    distance_cm: float = None,
    verbose: bool = True,
) -> dict:
    """Push an object in a direction using the closed gripper.

    Uses SAM3 masks only — ``target_mask`` for the object being pushed,
    ``occupancy_mask`` (union of all perception masks) for collision-free
    descent point finding.

    Args:
        demo:            RobotEnvUI instance.
        obj_id:          PyBullet body ID of the object to push.
        direction:       One of "left", "right", "up", "down" (image-relative).
        target_mask:     (H, W) SAM3 mask of the object to push.
        occupancy_mask:  (H, W) combined SAM mask of all detected objects.
        distance_cm:     Push sweep distance in cm.  If None, uses default
                         (10 cm).  Clamped to [3, 20] cm.
        verbose:         Print diagnostics.

    Returns:
        dict: success (push executed), direction, distance_cm, n_attempts,
              attempt_details.
    """
    if direction not in _DIRECTION_VECTORS:
        return dict(success=False, direction=direction, n_attempts=0,
                    attempt_details=[],
                    note=f"Invalid direction '{direction}'. "
                         f"Use one of: left, right, up, down.")

    # Resolve push distance: VLM-provided (clamped) or default
    if distance_cm is not None:
        push_dist_m = np.clip(float(distance_cm), 3.0, 20.0) / 100.0
    else:
        push_dist_m = _PUSH_DISTANCE   # default 0.10 m (10 cm)
    actual_cm = push_dist_m * 100.0

    env = demo.env
    camera = env.camera
    cam2rb = env.cam_to_robot_base
    push_dir = _DIRECTION_VECTORS[direction]
    push_yaw = _PUSH_FINGER_YAW[direction]

    executed = False
    attempt_details = []

    for attempt in range(_MAX_PUSH_ATTEMPTS):
        params = _find_push_params(camera, cam2rb, target_mask,
                                   occupancy_mask, direction)

        detail = dict(attempt=attempt, direction=direction,
                      distance_cm=actual_cm,
                      exec_result="no_clear_space")

        if params is None:
            if verbose:
                print(f"    push attempt {attempt}: no clear space "
                      f"for {direction} push")
            attempt_details.append(detail)
            for _ in range(_SETTLE_STEPS):
                env.step_simulation()
            continue

        dw = params["descent_world"]
        if verbose:
            cw = params["centroid_world"]
            print(f"    push attempt {attempt}: {direction} {actual_cm:.0f}cm "
                  f"descent=({dw[0]:.3f},{dw[1]:.3f}) "
                  f"obj=({cw[0]:.3f},{cw[1]:.3f})")

        if attempt > 0:
            env.reset_robot()
            for _ in range(10):
                env.step_simulation()

        _fsm_push(env, dw[:2], params["z_push"], push_dir, push_yaw,
                  push_dist_m)

        for _ in range(_SETTLE_STEPS):
            env.step_simulation()

        detail["exec_result"] = "executed"
        executed = True
        attempt_details.append(detail)
        if verbose:
            print(f"    push executed ✓")
        break   # push executed — done

    return dict(
        success=executed,
        direction=direction,
        distance_cm=actual_cm,
        n_attempts=len(attempt_details),
        attempt_details=attempt_details,
    )


