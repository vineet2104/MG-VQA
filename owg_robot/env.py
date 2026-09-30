import os
from datetime import datetime
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import random
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch
from numpy.lib.npyio import save
from PIL import Image
from typing import *
from pprint import pprint

from .camera import Camera
from .robot import setup_sisbot

CAM_X, CAM_Y, CAM_Z = 0.05, -0.52, 1.9
IMG_SIZE = 224

TARGET_LOCATIONS = {
    'top left corner': (0.35, -0.85, 1.0),
    'top side': (0, -0.85, 1.0),
    'top right corner': (-0.3, -0.85, 1.0),
    'left side': (0.3, -0.5, 1.0),
    'middle': (0, -0.5, 1.0),
    'right side': (-0.3, -0.5, 1.0),
    'bottom left corner': (0.35, -0.15, 1.0),
    'bottom side': (0, -0.15, 1.0),
    'bottom right corner': (-0.3, -0.15, 1.0),
}
TARGET_ZONE_POS = [0.7, 0.0, 0.785]


class FailToReachTargetError(RuntimeError):
    pass


class Environment:
    OBJECT_INIT_HEIGHT = 1.05
    GRIPPER_MOVING_HEIGHT = 1.25
    GRIPPER_GRASPED_LIFT_HEIGHT = 1.4
    TARGET_ZONE_POS = TARGET_ZONE_POS
    SIMULATION_STEP_DELAY = 0.0005  #speed of simulator - the lower the fatser ## should be a param
    FINGER_LENGTH = 0.06
    Z_TABLE_TOP = 0.785
    GRIP_REDUCTION = 0.60
    DIST_BACKGROUND = 1.115
    IMG_SIZE = 224
    PIX_CONVERSION = 277
    IMG_ROTATION = -np.pi * 0.54
    CAM_ROTATION = 0
    DEPTH_RADIUS = 1
    TARGET_LOCATIONS = TARGET_LOCATIONS

    # ── Plane containment wall constants ─────────────────────────────────
    _WALL_HALF_X = 0.08          # ±8 cm in X
    _WALL_HALF_Y = 0.07          # ±7 cm in Y
    _WALL_HEIGHT = 0.30          # 30 cm tall
    _WALL_THICKNESS = 0.005      # 5 mm thin
    _WALL_ANGLE_RAD = math.pi / 20  # ~9° inward tilt
    _WALL_CEILING_Z = 0.35       # ceiling offset above table
    _WALL_COLOR = [0.82, 0.87, 0.93, 0.45]  # light translucent

    def __init__(self,
                 camera: Camera,
                 asset_root="./assets",
                 vis=False,
                 debug=False,
                 gripper_type='140',
                 finger_length=0.06,
                 n_grasp_attempts=3,
                 use_basket=False,
                 basket_scale=1.0,
                 basket_wall_height=0.12,
                 use_planes=False) -> None:
        self.vis = vis
        self.use_basket = use_basket
        self.basket_scale = basket_scale
        self.basket_wall_height = basket_wall_height
        self.use_planes = use_planes
        self.debug = debug
        self.camera = camera
        # Get rotation matrix
        self.IMG_SIZE = self.camera.width
        self.N_GRASP_ATTEMPTS = n_grasp_attempts
        img_center = self.IMG_SIZE / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(
            -img_center / self.PIX_CONVERSION,
            img_center / self.PIX_CONVERSION, 0, self.IMG_ROTATION)
        self.cam_to_robot_base = self.get_transform_matrix(
            camera.x, camera.y, camera.z, self.CAM_ROTATION)

        self.assets_root = asset_root

        self.obj_init_pos = (camera.x, camera.y)
        self.obj_ids = []
        self.obj_names = []
        self.obj_positions = []
        self.obj_orientations = []
        self.obj_grasps = []  # a list of possible grasps per object
        self.obj_grasp_rects = []

        if gripper_type not in ('85', '140', 'suction'):
            raise NotImplementedError('Gripper %s not implemented.' %
                                      gripper_type)
        self.gripper_type = gripper_type
        self.finger_length = finger_length

        # define environment
        # Always use DIRECT mode — PyBullet's GUI requires an X server and
        # segfaults on headless Mesa/llvmpipe, and we do not rely on it:
        # video recording goes through camera.capture_frame() in
        # step_simulation, and visualization goes through Blender (see
        # enable_blender_rendering).  The --vis flag controls *those*
        # features, not PyBullet's native window.
        # (Collaborator's 083e600 briefly enabled p.GUI when vis=True,
        # which breaks headless runs — reverted.)
        _ = vis   # kept for signature compatibility
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF(os.path.join('plane.urdf'))
        self.tableID = p.loadURDF(os.path.join(self.assets_root,
                                               'urdf/objects/table.urdf'),
                                  [0.0, -0.65, 0.76],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=True)
        self.target_table_id = p.loadURDF(os.path.join(
            self.assets_root, 'urdf/objects/target_table.urdf'),
                                          [0.7, 0.0, 0.76],
                                          p.getQuaternionFromEuler([0, 0, 0]),
                                          useFixedBase=True)
        self.target_id = p.loadURDF(os.path.join(self.assets_root,
                                                 'urdf/objects/traybox.urdf'),
                                    self.TARGET_ZONE_POS,
                                    p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    globalScaling=0.7)
        self.UR5Stand_id = p.loadURDF(os.path.join(
            self.assets_root, 'urdf/objects/ur5_stand.urdf'),
                                      [-0.7, -0.36, 0.0],
                                      p.getQuaternionFromEuler([0, 0, 0]),
                                      useFixedBase=True)

        # --- Optional spawn basket: keeps objects contained on the table ---
        # Parameters:
        #   basket_scale      — scales the floor footprint (width × depth)
        #   basket_wall_height — wall height in metres (independent of scale)
        self.basket_id = None
        if self.use_basket:
            basket_urdf = self._generate_basket_urdf(
                base_width=0.40 * self.basket_scale,
                base_depth=0.30 * self.basket_scale,
                wall_height=self.basket_wall_height,
                wall_thickness=0.01,
                wall_angle=0.087,  # ~5° outward tilt
            )
            # Place basket centered at the object spawn area, on top of table
            basket_pos = [self.obj_init_pos[0], self.obj_init_pos[1],
                          self.Z_TABLE_TOP + 0.025]  # half the bottom plate thickness
            self.basket_id = p.loadURDF(
                basket_urdf, basket_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True)
            # High friction on basket walls to stop objects sliding out
            p.changeDynamics(self.basket_id, -1,
                             lateralFriction=1.0,
                             restitution=0.1)
            # Store basket interior bounds for spawn constraining
            half_x = (0.40 * self.basket_scale) / 2.0 - 0.02
            half_y = (0.30 * self.basket_scale) / 2.0 - 0.02
            self._basket_bounds = {
                "cx": self.obj_init_pos[0],
                "cy": self.obj_init_pos[1],
                "half_x": half_x,
                "half_y": half_y,
                "z_bottom": self.Z_TABLE_TOP + 0.025 + 0.01,  # top of bottom plate
            }
            print(f"[env] Spawn basket loaded at {basket_pos} "
                  f"(scale={self.basket_scale}, walls={self.basket_wall_height:.2f}m, "
                  f"interior={half_x*2:.2f}x{half_y*2:.2f}m)")

        # --- Optional plane containment walls (replaces basket) ---
        self._wall_ids = []
        if self.use_planes:
            self._create_containment_walls()

        if gripper_type == 'suction':
            urdf_name = 'urdf/ur5_suction.urdf'
        else:
            urdf_name = 'urdf/ur5_robotiq_%s.urdf' % gripper_type
        self.robot_id = p.loadURDF(
            os.path.join(self.assets_root, urdf_name),
            [0, 0, 0.0],  # StartPosition
            p.getQuaternionFromEuler([0, 0, 0]),  # StartOrientation
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE)

        if gripper_type == 'suction':
            from .robot import setup_suction
            self.joints, self.controlGripper, self.controlJoints, self.mimicParentName =\
                setup_suction(p, self.robot_id)
        else:
            self.joints, self.controlGripper, self.controlJoints, self.mimicParentName =\
                setup_sisbot(p, self.robot_id, gripper_type)
        self.eef_id = 7  # ee_link

        # Teleport arm to home position BEFORE any physics stepping.
        # At URDF load all joints=0, which extends the arm straight into the
        # tray at (0.7, 0, 0.785).  resetJointState bypasses physics so the
        # arm never collides with the tray during initialisation.
        _home = (0, -1.5446774605904932, 1.54, -1.54,
                 -1.5707970583733368, 0.0009377758247187636)
        for _i, _jname in enumerate(self.controlJoints[:6]):
            p.resetJointState(self.robot_id, self.joints[_jname].id, _home[_i])

        # Add force sensors and friction (Robotiq grippers only)
        if gripper_type != 'suction':
            p.enableJointForceTorqueSensor(
                self.robot_id, self.joints['left_inner_finger_pad_joint'].id)
            p.enableJointForceTorqueSensor(
                self.robot_id, self.joints['right_inner_finger_pad_joint'].id)
            p.changeDynamics(self.robot_id,
                             self.joints['left_inner_finger_pad_joint'].id,
                             lateralFriction=1)
            p.changeDynamics(self.robot_id,
                             self.joints['right_inner_finger_pad_joint'].id,
                             lateralFriction=1)

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # Task space (Cartesian space)
        if debug:
            self.xin = p.addUserDebugParameter('x', -0.4, 0.4, 0.11)
            self.yin = p.addUserDebugParameter('y', -0.8, 0, -0.49)
            self.zin = p.addUserDebugParameter('z', 0.9, 1.3, 1.1)
            self.rollId = p.addUserDebugParameter('roll', -3.14, 3.14,
                                                  0)  # -1.57 yaw
            self.pitchId = p.addUserDebugParameter('pitch', -3.14, 3.14,
                                                   np.pi / 2)
            self.yawId = p.addUserDebugParameter('yaw', -np.pi / 2, np.pi / 2,
                                                 0)  # -3.14 pitch
            self.gripper_opening_length_control = p.addUserDebugParameter(
                'gripper_opening_length', 0, 0.1, 0.085)

        # Add debug lines for end effector and camera
        if vis:
            self.eef_debug_lineID = None
            # p.addUserDebugLine([camera.x, camera.y, camera.z - 0.15], [camera.x, camera.y, camera.z - 0.4], [0, 1, 0], lineWidth=5)
            dist = 1.5
            yaw = 30
            pitch = -50
            target = [0.1, -0.30, 0.95]
            p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

            # camera model
            alpha = 0.025  #m
            z_cam_l1 = camera.z  #m

            color = [0.3, 0, 0]
            p.addUserDebugText("RGB-D camera",
                               [camera.x - 0.1, camera.y, camera.z + 0.02],
                               color,
                               textSize=2)

            p.addUserDebugLine([camera.x + alpha, camera.y + alpha, z_cam_l1],
                               [camera.x + alpha, camera.y - alpha, z_cam_l1],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x + alpha, camera.y - alpha, z_cam_l1],
                               [camera.x - alpha, camera.y - alpha, z_cam_l1],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - alpha, camera.y - alpha, z_cam_l1],
                               [camera.x - alpha, camera.y + alpha, z_cam_l1],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - alpha, camera.y + alpha, z_cam_l1],
                               [camera.x + alpha, camera.y + alpha, z_cam_l1],
                               color,
                               lineWidth=4)

            z_cam_l2 = camera.z - 0.07  #m

            p.addUserDebugLine([camera.x + alpha, camera.y + alpha, z_cam_l2],
                               [camera.x + alpha, camera.y - alpha, z_cam_l2],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x + alpha, camera.y - alpha, z_cam_l2],
                               [camera.x - alpha, camera.y - alpha, z_cam_l2],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - alpha, camera.y - alpha, z_cam_l2],
                               [camera.x - alpha, camera.y + alpha, z_cam_l2],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - alpha, camera.y + alpha, z_cam_l2],
                               [camera.x + alpha, camera.y + alpha, z_cam_l2],
                               color,
                               lineWidth=4)

            ### body
            p.addUserDebugLine([camera.x + alpha, camera.y + alpha, z_cam_l1],
                               [camera.x + alpha, camera.y + alpha, z_cam_l2],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x + alpha, camera.y - alpha, z_cam_l1],
                               [camera.x + alpha, camera.y - alpha, z_cam_l2],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - alpha, camera.y - alpha, z_cam_l1],
                               [camera.x - alpha, camera.y - alpha, z_cam_l2],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - alpha, camera.y + alpha, z_cam_l1],
                               [camera.x - alpha, camera.y + alpha, z_cam_l2],
                               color,
                               lineWidth=4)

            ### Third rectangle
            z_cam_l3 = camera.z - 0.125  #m
            beta = alpha * 2
            p.addUserDebugLine([camera.x + beta, camera.y + beta, z_cam_l3],
                               [camera.x + beta, camera.y - beta, z_cam_l3],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x + beta, camera.y - beta, z_cam_l3],
                               [camera.x - beta, camera.y - beta, z_cam_l3],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - beta, camera.y - beta, z_cam_l3],
                               [camera.x - beta, camera.y + beta, z_cam_l3],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - beta, camera.y + beta, z_cam_l3],
                               [camera.x + beta, camera.y + beta, z_cam_l3],
                               color,
                               lineWidth=4)

            ### body
            p.addUserDebugLine([camera.x + alpha, camera.y + alpha, z_cam_l2],
                               [camera.x + beta, camera.y + beta, z_cam_l3],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x + alpha, camera.y - alpha, z_cam_l2],
                               [camera.x + beta, camera.y - beta, z_cam_l3],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - alpha, camera.y - alpha, z_cam_l2],
                               [camera.x - beta, camera.y - beta, z_cam_l3],
                               color,
                               lineWidth=4)
            p.addUserDebugLine([camera.x - alpha, camera.y + alpha, z_cam_l2],
                               [camera.x - beta, camera.y + beta, z_cam_l3],
                               color,
                               lineWidth=4)

            ### working area
            working_area = 0.79  #m
            beta = 0.4
            p.addUserDebugLine(
                [camera.x + beta, camera.y + beta, working_area],
                [camera.x + beta, camera.y - beta, working_area], [0, 1, 0],
                lineWidth=5)
            p.addUserDebugLine(
                [camera.x + beta, camera.y - beta, working_area],
                [camera.x - beta, camera.y - beta, working_area], [0, 1, 0],
                lineWidth=5)
            p.addUserDebugLine(
                [camera.x - beta, camera.y - beta, working_area],
                [camera.x - beta, camera.y + beta, working_area], [0, 1, 0],
                lineWidth=5)
            p.addUserDebugLine(
                [camera.x - beta, camera.y + beta, working_area],
                [camera.x + beta, camera.y + beta, working_area], [0, 1, 0],
                lineWidth=5)

        # --- Motion recording for Blender animation export ---
        self._motion_log: list = []          # list of frame snapshots
        self._recording_motion: bool = False # toggled by manipulation methods
        self._motion_step_counter: int = 0   # counts step_simulation calls while recording
        self._motion_frame_counter: int = 0  # sequential frame index in the log
        self._motion_record_interval: int = 8  # record every Nth step (240Hz / 8 = 30fps)
        self._motion_playback_fps: int = 30

        # Setup some Limit
        self.gripper_open_limit = (0.0, 0.1)
        self.ee_position_limit = ((-0.8, 0.8), (-0.8, 0.8), (0.785, 1.4))
        self.reset_robot()

    @staticmethod
    def get_transform_matrix(x, y, z, rot):
        return np.array([[np.cos(rot), -np.sin(rot), 0, x],
                         [np.sin(rot), np.cos(rot), 0, y], [0, 0, 1, z],
                         [0, 0, 0, 1]])

    # ------------------------------------------------------------------
    # Motion recording for Blender animation export
    # ------------------------------------------------------------------

    def start_motion_recording(self, playback_fps: int = 30):
        """Enable motion recording. Frames are captured during manipulation."""
        self._motion_log = []
        self._recording_motion = False
        self._motion_step_counter = 0
        self._motion_frame_counter = 0
        self._motion_playback_fps = playback_fps
        # PyBullet default is 240Hz; record every Nth step to hit target fps
        self._motion_record_interval = max(1, round(240 / playback_fps))

    def stop_motion_recording(self):
        """Disable motion recording and return the log."""
        self._recording_motion = False
        return list(self._motion_log)

    def get_motion_log(self) -> list:
        """Return the current motion log (list of frame snapshots)."""
        return list(self._motion_log)

    def clear_motion_log(self):
        """Clear the motion log."""
        self._motion_log = []
        self._motion_step_counter = 0
        self._motion_frame_counter = 0

    def set_recording_motion(self, active: bool):
        """Toggle motion capture on/off (called around manipulation actions)."""
        self._recording_motion = active
        if active:
            self._motion_step_counter = 0

    @staticmethod
    def _get_visual_frame(body_id: int, link_index: int, pc: int):
        """Get the visual-frame position and orientation for a body/link.

        For non-base links, getLinkState()[4:6] gives the URDF link frame.
        For the base link, getBasePositionAndOrientation() returns the
        *inertial* frame.  We undo the local inertial offset to recover
        the link (visual) frame — matching BlenderRenderer._get_visual_frame.
        """
        if link_index == -1:
            inertial_pos, inertial_orn = p.getBasePositionAndOrientation(
                body_id, physicsClientId=pc)
            dyn = p.getDynamicsInfo(body_id, -1, physicsClientId=pc)
            local_inertial_pos = dyn[3]
            local_inertial_orn = dyn[4]

            if (abs(local_inertial_pos[0]) < 1e-8 and
                abs(local_inertial_pos[1]) < 1e-8 and
                abs(local_inertial_pos[2]) < 1e-8):
                return list(inertial_pos), list(inertial_orn)

            inv_pos, inv_orn = p.invertTransform(
                local_inertial_pos, local_inertial_orn)
            vis_pos, vis_orn = p.multiplyTransforms(
                inertial_pos, inertial_orn, inv_pos, inv_orn)
            return list(vis_pos), list(vis_orn)
        else:
            state = p.getLinkState(body_id, link_index, physicsClientId=pc)
            return list(state[4]), list(state[5])

    def _get_visual_origin_offsets(self, body_id: int):
        """Return cached URDF <visual><origin> offsets for a body.

        Each entry is ``{link_index: (local_pos, local_orn)}`` where the
        offset is non-trivial.  These must be composed with the link
        frame to match Blender mesh placement (same logic as
        ``BlenderRenderer._load_body``).
        """
        if not hasattr(self, "_visual_offset_cache"):
            self._visual_offset_cache: dict = {}
        if body_id in self._visual_offset_cache:
            return self._visual_offset_cache[body_id]

        offsets = {}
        try:
            shapes = p.getVisualShapeData(
                body_id, physicsClientId=self.physicsClient)
            for shape in shapes:
                link_index = shape[1]
                local_pos = shape[5]
                local_orn = shape[6]
                has_offset = (any(abs(v) > 1e-8 for v in local_pos) or
                              any(abs(v) > 1e-8 for v in local_orn))
                if has_offset:
                    offsets[link_index] = (local_pos, local_orn)
        except Exception:
            pass
        self._visual_offset_cache[body_id] = offsets
        return offsets

    def _capture_motion_frame(self):
        """Snapshot all body transforms for one animation frame.

        Uses visual-frame transforms (same as the Blender renderer) so
        that keyframed positions match the mesh placement in the .blend.
        Also applies URDF <visual><origin> offsets — without these,
        objects whose mesh origin doesn't match the link origin appear
        at wrong positions in the .blend animation.
        """
        self._motion_frame_counter += 1
        transforms = []

        # Capture all PyBullet bodies (robot links + objects + furniture)
        num_bodies = p.getNumBodies(physicsClientId=self.physicsClient)
        for i in range(num_bodies):
            body_id = p.getBodyUniqueId(i, physicsClientId=self.physicsClient)
            n_joints = p.getNumJoints(body_id, physicsClientId=self.physicsClient)
            offsets = self._get_visual_origin_offsets(body_id)

            # Base link (visual frame, not inertial frame)
            pos, orn = self._get_visual_frame(body_id, -1, self.physicsClient)
            if -1 in offsets:
                lp, lo = offsets[-1]
                pos, orn = p.multiplyTransforms(pos, orn, lp, lo)
                pos, orn = list(pos), list(orn)
            transforms.append({
                "body_id": body_id,
                "link_index": -1,
                "pos": pos,
                "orn_xyzw": orn,
            })

            # Child links
            for link_idx in range(n_joints):
                pos, orn = self._get_visual_frame(
                    body_id, link_idx, self.physicsClient)
                if link_idx in offsets:
                    lp, lo = offsets[link_idx]
                    pos, orn = p.multiplyTransforms(pos, orn, lp, lo)
                    pos, orn = list(pos), list(orn)
                transforms.append({
                    "body_id": body_id,
                    "link_index": link_idx,
                    "pos": pos,
                    "orn_xyzw": orn,
                })

        self._motion_log.append({
            "frame": self._motion_frame_counter,
            "transforms": transforms,
        })

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()

        # Motion recording: capture frame at sub-sampled rate
        if self._recording_motion:
            self._motion_step_counter += 1
            if self._motion_step_counter % self._motion_record_interval == 0:
                self._capture_motion_frame()
        if self.vis:
            self.camera.capture_frame()
            if self.debug:
                if self.eef_debug_lineID is not None:
                    p.removeUserDebugItem(self.eef_debug_lineID)
                eef_xyz = p.getLinkState(self.robot_id, self.eef_id)[0:1]
                end = np.array(eef_xyz[0])
                end[2] -= 0.5
                self.eef_debug_lineID = p.addUserDebugLine(
                    np.array(eef_xyz[0]), end, [1, 0, 0])
            time.sleep(self.SIMULATION_STEP_DELAY)

    @staticmethod
    def is_still(handle):
        still_eps = 1e-3
        lin_vel, ang_vel = p.getBaseVelocity(handle)
        # print(np.abs(lin_vel).sum() + np.abs(ang_vel).sum())
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps

    def wait_until_still(self, objID, max_wait_epochs=100):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if self.is_still(objID):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def wait_until_all_still(self, max_wait_epochs=1000):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if np.all(list(self.is_still(obj_id) for obj_id in self.obj_ids)):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(
            self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def reset_robot(self):
        user_parameters = (0, -1.5446774605904932, 1.54, -1.54,
                           -1.5707970583733368, 0.0009377758247187636, 0.085)
        for _ in range(60):
            for i, name in enumerate(self.controlJoints):
                if name not in self.joints:
                    continue  # skip dummy entries (e.g. suction)

                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(self.robot_id,
                                        joint.id,
                                        p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i],
                                        force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                self.step_simulation()

            self.controlGripper(controlMode=p.POSITION_CONTROL,
                                targetPosition=0.085)
            self.step_simulation()

    def move_arm_away(self):
        joint = self.joints['shoulder_pan_joint']
        for _ in range(200):
            p.setJointMotorControl2(self.robot_id,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=0.,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

            self.step_simulation()

    def check_grasped(self, obj_id):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robot_id,
                                          linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id,
                                           linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left + contact_right
                          if item[2] in [obj_id])
        if len(contact_ids) == 1:
            return True
        return False

    def check_grasped_id(self):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robot_id,
                                          linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id,
                                           linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left + contact_right
                          if item[2] in self.obj_ids)
        if len(contact_ids) > 1:
            if self.debug:
                print('Warning: Multiple items in hand!')
        return list(item_id for item_id in contact_ids
                    if item_id in self.obj_ids)

    def check_contact(self, id_a, id_b):
        contact_a = p.getContactPoints(bodyA=id_a)
        contact_ids = set(item[2] for item in contact_a if item[2] in [id_b])
        if len(contact_ids) == 1:
            return True
        return False

    def check_target_reached(self, obj_id):
        aabb = p.getAABB(self.target_id, -1)
        x_min, x_max = aabb[0][0], aabb[1][0]
        y_min, y_max = aabb[0][1], aabb[1][1]
        pos = p.getBasePositionAndOrientation(obj_id)
        x, y = pos[0][0], pos[0][1]
        if x > x_min and x < x_max and y > y_min and y < y_max:
            return True
        return False

    def gripper_contact(self, bool_operator='and', force=250):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robot_id,
                                          linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id,
                                           linkIndexA=right_index)

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Check the force
        left_force = p.getJointState(
            self.robot_id, left_index)[2][:3]  # 6DOF, Torque is ignored
        right_force = p.getJointState(self.robot_id, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(
            right_force)

        # print(left_norm, right_norm)
        if bool_operator == 'and':
            return left_norm > force and right_norm > force
        else:
            return left_norm > force or right_norm > force

    def move_gripper(self, gripper_opening_length: float, step: int = 120):
        gripper_opening_length = np.clip(gripper_opening_length,
                                         *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin(
            (gripper_opening_length - 0.010) / 0.1143)  # angle calculation

        for _ in range(step):
            self.controlGripper(controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_opening_angle)

            self.step_simulation()

    def auto_close_gripper(self,
                           step: int = 120,
                           check_contact: bool = False) -> bool:
        # Get initial gripper open position
        initial_position = p.getJointState(
            self.robot_id, self.joints[self.mimicParentName].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            # time.sleep(1 / 120)
            if check_contact and self.gripper_contact():
                return True
        return False

    def calc_z_offset(self, gripper_opening_length: float):
        gripper_opening_length = np.clip(gripper_opening_length,
                                         *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - \
            math.asin((gripper_opening_length - 0.010) / 0.1143)
        if self.gripper_type == '140':
            gripper_length = 10.3613 * \
                np.sin(1.64534-0.24074 * (gripper_opening_angle / np.pi)) - 10.1219
        else:
            gripper_length = 1.231 - 1.1
        return gripper_length

    def remove_obj(self, obj_id):
        # Get index of obj in id list, then remove object from simulation
        idx = self.obj_ids.index(obj_id)
        self.obj_orientations.pop(idx)
        self.obj_positions.pop(idx)
        self.obj_ids.pop(idx)
        self.obj_names.pop(idx)
        self.obj_grasps.pop(idx)
        self.obj_grasp_rects.pop(idx)
        # Clear cached visual offsets for removed body
        if hasattr(self, "_visual_offset_cache"):
            self._visual_offset_cache.pop(obj_id, None)
        p.removeBody(obj_id)

    def remove_all_obj(self):
        for obj_id in self.obj_ids:
            p.removeBody(obj_id)
            for _ in range(10):
                self.step_simulation()
        self.obj_ids.clear()
        self.obj_positions.clear()
        self.obj_orientations.clear()
        self.obj_names.clear()
        self.obj_grasps.clear()
        self.obj_grasp_rects.clear()

    def reset_all_obj(self):
        for i, obj_id in enumerate(self.obj_ids):
            p.resetBasePositionAndOrientation(obj_id, self.obj_positions[i],
                                              self.obj_orientations[i])
            self.obj_grasps[i] = None
            self.obj_grasp_rects[i] = None
        self.wait_until_all_still()

    def set_obj_state(self, state):
        for i, item in enumerate(state):
            p.resetBasePositionAndOrientation(item['id'], item['pos'],
                                              item['orn'])
            self.obj_grasps[i] = None
            self.obj_grasp_rects[i] = None
        self.wait_until_all_still()

    def update_obj_states(self):
        for i, obj_id in enumerate(self.obj_ids):
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            self.obj_positions[i] = pos
            self.obj_orientations[i] = orn
            self.obj_grasps[i] = None
            self.obj_grasp_rects[i] = None

    def get_obj_pos(self, obj_id):
        index = self.obj_ids.index(obj_id)
        return self.obj_positions[index]

    def get_obj_orn(self, obj_id):
        index = self.obj_ids.index(obj_id)
        return self.obj_orientations[index]

    def get_obj_id_by_name(self, obj_name):
        index = self.obj_names.index(obj_name)
        return self.obj_ids[index]

    def get_ee_pos(self):
        ee = p.getLinkState(self.robot_id, self.eef_id)
        ee_xyz = np.float32(ee[0])
        ee_orn = np.float32(ee[1])
        return ee_xyz, ee_orn

    def get_obj_states(self):
        return [{
            'id': i,
            'name': n,
            'pos': p,
            'orn': o
        } for i, n, p, o in zip(self.obj_ids, self.obj_names,
                                self.obj_positions, self.obj_orientations)]

    @staticmethod
    def _generate_basket_urdf(
        base_width: float = 0.40,
        base_depth: float = 0.30,
        wall_height: float = 0.12,
        wall_thickness: float = 0.01,
        wall_angle: float = 0.087,
    ) -> str:
        """Generate a basket URDF with the given dimensions and write to a temp file.

        Each wall is a separate link connected by a fixed joint so that the
        Blender renderer (which maps one visual per link) can place them
        independently.

        Args:
            base_width:     Interior width along X (metres).
            base_depth:     Interior depth along Y (metres).
            wall_height:    Height of walls above the bottom plate (metres).
            wall_thickness: Thickness of each wall (metres).
            wall_angle:     Outward tilt of walls in radians (~5° = 0.087).

        Returns:
            Path to the generated URDF file (in /tmp).
        """
        import tempfile
        # Wall center Z relative to body origin
        wz = wall_height / 2.0 + 0.01  # 0.01 = bottom plate thickness
        # Wall X/Y offsets (half the base dimension + half wall thickness)
        wx = base_width / 2.0 + wall_thickness / 2.0
        wy = base_depth / 2.0 + wall_thickness / 2.0

        rgba = "0.45 0.42 0.38 1.0"

        def _wall_link_and_joint(name, link_id, rpy, xyz, size):
            """Generate a <link> + <joint> pair for one wall."""
            return f"""
  <!-- {name} -->
  <link name="{link_id}">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <geometry><box size="{size}"/></geometry>
      <material name="basket_mat"/>
    </visual>
    <collision>
      <geometry><box size="{size}"/></geometry>
    </collision>
  </link>
  <joint name="joint_{link_id}" type="fixed">
    <parent link="base_link"/>
    <child link="{link_id}"/>
    <origin rpy="{rpy}" xyz="{xyz}"/>
  </joint>"""

        walls = ""
        walls += _wall_link_and_joint(
            "Wall +X", "wall_px",
            f"0 {-wall_angle} 0", f"{wx:.4f} 0 {wz:.4f}",
            f"{wall_thickness} {base_depth} {wall_height}")
        walls += _wall_link_and_joint(
            "Wall -X", "wall_nx",
            f"0 {wall_angle} 0", f"{-wx:.4f} 0 {wz:.4f}",
            f"{wall_thickness} {base_depth} {wall_height}")
        walls += _wall_link_and_joint(
            "Wall +Y", "wall_py",
            f"{-wall_angle} 0 0", f"0 {wy:.4f} {wz:.4f}",
            f"{base_width} {wall_thickness} {wall_height}")
        walls += _wall_link_and_joint(
            "Wall -Y", "wall_ny",
            f"{wall_angle} 0 0", f"0 {-wy:.4f} {wz:.4f}",
            f"{base_width} {wall_thickness} {wall_height}")

        urdf = f"""<?xml version="1.0" ?>
<robot name="spawn_basket">
  <link name="base_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <!-- Bottom plate -->
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0.005"/>
      <geometry><box size="{base_width} {base_depth} 0.01"/></geometry>
      <material name="basket_mat">
        <color rgba="{rgba}"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0.005"/>
      <geometry><box size="{base_width} {base_depth} 0.01"/></geometry>
    </collision>
  </link>
{walls}
</robot>
"""
        fd, path = tempfile.mkstemp(suffix=".urdf", prefix="spawn_basket_")
        with os.fdopen(fd, "w") as f:
            f.write(urdf)
        return path

    def load_obj(self,
                 path,
                 name,
                 pos,
                 yaw,
                 mod_orn=False,
                 mod_stiffness=False):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        obj_id = p.loadURDF(path, pos, orn)
        # adjust position according to height
        aabb = p.getAABB(obj_id, -1)
        if mod_orn:
            minm, maxm = aabb[0][1], aabb[1][1]
            orn = p.getQuaternionFromEuler([0, np.pi * 0.5, yaw])
        else:
            minm, maxm = aabb[0][2], aabb[1][2]

        pos[2] += (maxm - minm) / 2
        p.resetBasePositionAndOrientation(obj_id, pos, orn)

        # ── Fix inertia: URDF ships identity inertia (1,1,1) which is
        # ~2400× too large for ~5cm objects.  Compute realistic box-
        # approximation inertia from the AABB so objects actually settle
        # under gravity instead of floating in precarious poses.
        aabb_min, aabb_max = np.array(aabb[0]), np.array(aabb[1])
        dims = aabb_max - aabb_min                       # dx, dy, dz
        obj_mass = 0.3  # kg — typical small household object
        ixx = obj_mass / 12.0 * (dims[1]**2 + dims[2]**2)
        iyy = obj_mass / 12.0 * (dims[0]**2 + dims[2]**2)
        izz = obj_mass / 12.0 * (dims[0]**2 + dims[1]**2)

        # change dynamics
        if mod_stiffness:
            p.changeDynamics(obj_id,
                             -1,
                             mass=obj_mass,
                             localInertiaDiagonal=[ixx, iyy, izz],
                             lateralFriction=0.8,
                             rollingFriction=0.005,
                             spinningFriction=0.005,
                             restitution=0.01,
                             contactStiffness=100000,
                             contactDamping=1.0)
        else:
            p.changeDynamics(obj_id,
                             -1,
                             mass=obj_mass,
                             localInertiaDiagonal=[ixx, iyy, izz],
                             lateralFriction=0.8,
                             rollingFriction=0.005,
                             spinningFriction=0.005,
                             restitution=0.01)
        self.obj_ids.append(obj_id)
        self.obj_positions.append(pos)
        self.obj_orientations.append(orn)
        self.obj_names.append(name)
        self.obj_grasps.append(None)
        self.obj_grasp_rects.append(None)
        return obj_id, name, pos, orn

    # ------------------------------------------------------------------
    # Plane containment walls (alternative to basket)
    # ------------------------------------------------------------------
    def _create_containment_walls(self):
        """Create 4 visible tilted box walls + 1 invisible ceiling plane.

        Walls constrain objects during spawning and are removed before the
        experiment begins.  See ``remove_containment_walls()``.
        """
        cx, cy = self.obj_init_pos
        wh = self._WALL_HEIGHT / 2.0
        wt = self._WALL_THICKNESS / 2.0
        wz = self.Z_TABLE_TOP + wh
        hx, hy = self._WALL_HALF_X, self._WALL_HALF_Y
        angle = self._WALL_ANGLE_RAD
        color = self._WALL_COLOR

        def _box_wall(half_extents, position, orientation):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                      rgbaColor=color)
            return p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis,
                                     basePosition=position,
                                     baseOrientation=orientation)

        # +X wall (right)
        self._wall_ids.append(_box_wall(
            [wt, hy + self._WALL_THICKNESS, wh],
            [cx + hx, cy, wz],
            p.getQuaternionFromEuler([0, -angle, 0])))
        # -X wall (left)
        self._wall_ids.append(_box_wall(
            [wt, hy + self._WALL_THICKNESS, wh],
            [cx - hx, cy, wz],
            p.getQuaternionFromEuler([0, angle, 0])))
        # +Y wall (back)
        self._wall_ids.append(_box_wall(
            [hx + self._WALL_THICKNESS, wt, wh],
            [cx, cy + hy, wz],
            p.getQuaternionFromEuler([angle, 0, 0])))
        # -Y wall (front)
        self._wall_ids.append(_box_wall(
            [hx + self._WALL_THICKNESS, wt, wh],
            [cx, cy - hy, wz],
            p.getQuaternionFromEuler([-angle, 0, 0])))
        # Ceiling (invisible — collision only, prevents bounce-out)
        ceil = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 0, -1])
        self._wall_ids.append(p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=ceil,
            basePosition=[cx, cy, self.Z_TABLE_TOP + self._WALL_CEILING_Z]))

        print(f"[env] Containment walls created: {len(self._wall_ids)} bodies "
              f"(region {hx*2:.2f}×{hy*2:.2f}m, height {self._WALL_HEIGHT:.2f}m)")

    def remove_containment_walls(self):
        """Remove spawn containment walls from the simulation.

        Should be called after objects have settled and before the
        experiment begins (i.e. before the VLM sees the scene).
        """
        if not self._wall_ids:
            return
        for wid in self._wall_ids:
            p.removeBody(wid)
        n = len(self._wall_ids)
        self._wall_ids.clear()
        print(f"[env] Removed {n} containment walls")

    def hide_containment_walls(self):
        """Make side walls invisible but keep collision; remove ceiling.

        Side walls become transparent (alpha=0) so they don't appear in
        PyBullet's segmentation buffer or RGB renders, but their
        collision shapes remain active — objects stay contained in the
        spawn region throughout the experiment.

        The ceiling plane is REMOVED entirely (not just hidden) because
        the robot arm needs to lift objects above the walls to place
        them on the tray.  The ceiling was only needed during spawn to
        prevent bounce-out.

        This is preferred over remove_containment_walls() because it
        prevents objects from drifting apart after settling while being
        invisible to both the VLM (Blender) and occlusion measurements
        (PyBullet seg).
        """
        if not self._wall_ids:
            return

        # The ceiling is always the LAST entry (appended after 4 side walls)
        # Remove it so the robot arm can lift objects above walls.
        if len(self._wall_ids) >= 5:
            ceiling_id = self._wall_ids.pop()  # remove last (ceiling)
            p.removeBody(ceiling_id)

        # Hide remaining side walls (make transparent, keep collision)
        for wid in self._wall_ids:
            vs_data = p.getVisualShapeData(wid)
            for entry in vs_data:
                link_idx = entry[1]
                p.changeVisualShape(wid, link_idx,
                                    rgbaColor=[0, 0, 0, 0])
        print(f"[env] Hid {len(self._wall_ids)} side walls "
              f"(collision active, ceiling removed)")

    def has_containment_walls(self):
        """Return True if containment walls are still present."""
        return bool(self._wall_ids)

    def load_isolated_obj(self,
                          path,
                          name,
                          mod_orn=False,
                          mod_stiffness=False,
                          spawn_radius=None,
                          drop_height=None,
                          drop_xy=None):
        """Load and drop an object onto the scene.

        Args:
            spawn_radius: If provided, overrides the default spawn region.
                Tuple (half_x, half_y) in meters for the XY spawn area.
            drop_height: If provided, overrides the default drop Z
                (meters above table top).
            drop_xy: If provided, (x, y) in world meters — the object is
                dropped over this exact XY (clamped to the wall region) with
                a small jitter, instead of a random spot.  Used by counting to
                drop the big-cover distractor right onto a target instance.
        """
        cx, cy = self.obj_init_pos

        if drop_xy is not None:
            # Drop over an explicit XY (e.g. onto a settled target instance).
            # A tiny jitter avoids a perfectly-centered balance; clamp to the
            # wall region if walls are present so it stays contained.
            jx = random.uniform(-0.015, 0.015)
            jy = random.uniform(-0.015, 0.015)
            r_x = drop_xy[0] + jx
            r_y = drop_xy[1] + jy
            if self.use_planes and self._wall_ids:
                margin = 0.02
                r_x = min(max(r_x, cx - self._WALL_HALF_X + margin),
                          cx + self._WALL_HALF_X - margin)
                r_y = min(max(r_y, cy - self._WALL_HALF_Y + margin),
                          cy + self._WALL_HALF_Y - margin)
            if drop_height is not None:
                drop_z = self.Z_TABLE_TOP + drop_height
            else:
                drop_z = self.Z_TABLE_TOP + 0.20 + random.uniform(0, 0.05)
        elif spawn_radius is not None:
            # Custom spawn radius (used by two-pass spawn for tight drops)
            half_x, half_y = spawn_radius
            r_x = random.uniform(cx - half_x, cx + half_x)
            r_y = random.uniform(cy - half_y, cy + half_y)
            if drop_height is not None:
                drop_z = self.Z_TABLE_TOP + drop_height
            else:
                drop_z = self.Z_TABLE_TOP + 0.20 + random.uniform(0, 0.05)
        elif self.use_planes and self._wall_ids:
            # Spawn within the plane-wall region.  Low drop height
            # (8-13cm) means objects land gently and don't bounce/scatter
            # away from the pile — creating natural horizontal overlap.
            # ``drop_height`` (if given) overrides the default gentle drop —
            # used by counting to layer targets (base pile) under distractors
            # while keeping wall containment so nothing scatters out.
            margin = 0.02
            r_x = random.uniform(cx - self._WALL_HALF_X + margin,
                                 cx + self._WALL_HALF_X - margin)
            r_y = random.uniform(cy - self._WALL_HALF_Y + margin,
                                 cy + self._WALL_HALF_Y - margin)
            if drop_height is not None:
                drop_z = self.Z_TABLE_TOP + drop_height
            else:
                drop_z = self.Z_TABLE_TOP + 0.08 + random.uniform(0, 0.05)
        elif self.use_basket and hasattr(self, '_basket_bounds'):
            # Spawn within the basket interior bounds
            bb = self._basket_bounds
            r_x = random.uniform(bb["cx"] - bb["half_x"],
                                 bb["cx"] + bb["half_x"])
            r_y = random.uniform(bb["cy"] - bb["half_y"],
                                 bb["cy"] + bb["half_y"])
            # Drop from above basket walls (wall height ~0.06 * scale)
            drop_z = bb["z_bottom"] + 0.15
        else:
            r_x = random.uniform(cx - 0.1, cx + 0.1)
            r_y = random.uniform(cy - 0.1, cy + 0.1)
            drop_z = self.Z_TABLE_TOP + 0.2
        yaw = random.uniform(0, np.pi)

        pos = [r_x, r_y, drop_z]
        _t0 = time.time()
        obj_id, _, _, _ = self.load_obj(path, name, pos, yaw, mod_orn,
                                        mod_stiffness)
        print(f"    [timing] loadURDF({name}): {time.time()-_t0:.2f}s", flush=True)

        _t0 = time.time()
        self.dummy_simulation_steps(100)
        print(f"    [timing] 100 settle steps: {time.time()-_t0:.2f}s", flush=True)

        _t0 = time.time()
        for _ in range(100):
            p.stepSimulation()
            if self.is_still(obj_id):
                break
        print(f"    [timing] wait_until_still: {time.time()-_t0:.2f}s", flush=True)

        self.update_obj_states()

    def set_obj_grasps(self, obj_id, grasps, grasp_rects):
        """
        Set list of possible grasps to object based on oracle / network
        """
        idx = self.obj_ids.index(obj_id)
        self.obj_grasps[idx] = grasps
        self.obj_grasp_rects[idx] = grasp_rects

    def get_obj_grasps(self, obj_id):
        idx = self.obj_ids.index(obj_id)
        return self.obj_grasps[idx]

    def get_obj_grasp_rects(self, obj_id):
        idx = self.obj_ids.index(obj_id)
        return self.obj_grasp_rects[idx]

    def get_obs(self, pointcloud=True):
        rgb, depth, seg = self.camera.get_cam_img()
        # remove background from seg
        state = self.get_obj_states()
        tmp = {int(x['id']): x for x in state}
        state_ids = [int(x['id']) for x in state]
        indices = np.unique(seg)
        for index in set(indices).difference(set(state_ids)):
            seg[seg == index] = 0
        result = {'image': rgb, 'depth': depth, 'seg': seg, 'obj_ids': indices}
        if pointcloud:
            pc = self.camera.get_pointcloud().reshape(-1, 3)
            # move to robot base frame
            pc[:, 2] = -pc[:, 2]
            pc[:, 1] = -pc[:, 1]  #convert to ref frame
            pc_hom = np.dot(self.cam_to_robot_base,
                            np.vstack([pc.T, np.ones((1, pc.shape[0]))]))
            pc_w = pc_hom[:3, :].T
            result['points'] = pc_w
        return result

    def create_temp_box(self, width, num):
        box_width = width
        box_height = 0.2
        box_z = self.Z_TABLE_TOP + (box_height / 2)
        id1 = p.loadURDF(f'owg_robot/assets/urdf/objects/slab{num}.urdf', [
            self.obj_init_pos[0] - box_width / 2, self.obj_init_pos[1], box_z
        ],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id2 = p.loadURDF(f'owg_robot/assets/urdf/objects/slab{num}.urdf', [
            self.obj_init_pos[0] + box_width / 2, self.obj_init_pos[1], box_z
        ],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id3 = p.loadURDF(f'owg_robot/assets/urdf/objects/slab{num}.urdf', [
            self.obj_init_pos[0], self.obj_init_pos[1] + box_width / 2, box_z
        ],
                         p.getQuaternionFromEuler([0, 0, np.pi * 0.5]),
                         useFixedBase=True)
        id4 = p.loadURDF(f'owg_robot/assets/urdf/objects/slab{num}.urdf', [
            self.obj_init_pos[0], self.obj_init_pos[1] - box_width / 2, box_z
        ],
                         p.getQuaternionFromEuler([0, 0, np.pi * 0.5]),
                         useFixedBase=True)
        return [id1, id2, id3, id4]

    def move_obj_along_axis(self, obj_id, axis, operator, step, stop):
        collison = False
        while not collison:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            new_pos = list(pos)
            if operator == '+':
                new_pos[axis] += step
                if new_pos[axis] > stop:
                    break
            else:
                new_pos[axis] -= step
                if new_pos[axis] < stop:
                    break
            # Move object towards center
            p.resetBasePositionAndOrientation(obj_id, new_pos, orn)
            p.stepSimulation()
            contact_a = p.getContactPoints(obj_id)
            # If object collides with any other object, stop
            contact_ids = set(item[2] for item in contact_a
                              if item[2] in self.obj_ids)
            if len(contact_ids) != 0:
                collison = True
        # Move one step back
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        new_pos = list(pos)
        if operator == '+':
            new_pos[axis] -= step
        else:
            new_pos[axis] += step
        p.resetBasePositionAndOrientation(obj_id, new_pos, orn)

    def move_to_init_pos(self):
        # move higher at place first
        ee_pos, ee_orn = self.get_ee_pos()
        self.move_ee([ee_pos[0], ee_pos[1], 1.25, ee_orn])
        for _ in range(30):
            self.step_simulation()
        # move init pos
        x, y, z = 0.49581263390917246, 0.10914993600132877, 1.21110183716901
        orn = (0.00032693392040361596, 0.7162828147217722,
               -0.00033611268582567366, 0.6978099379320486)
        action = (x, y, z, orn)
        return self.move_ee(action)

    @staticmethod
    def _quat_distance_deg(q1, q2):
        """Orientation error in degrees between two (x,y,z,w) quaternions.

        Uses the geodesic distance on SO(3): angle = 2 * arccos(|q1 · q2|).
        Handles the q ≡ −q equivalence (both represent the same rotation).
        Immune to gimbal lock — unlike the old Euler-angle comparison.
        """
        dot = abs(np.dot(q1, q2))
        dot = min(dot, 1.0)               # numerical safety
        return np.degrees(2.0 * np.arccos(dot))

    def move_ee(self,
                action,
                max_step=300,
                check_collision_config=None,
                custom_velocity=None,
                try_close_gripper=False,
                verbose=False,
                pos_tol=0.001,
                orn_tol_deg=1.0):
        x, y, z, orn = action
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        still_open_flag_ = True  # Hot fix

        real_xyz, real_xyzw = p.getLinkState(self.robot_id, self.eef_id)[0:2]
        alpha = 0.2  # this parameter can be tuned to make the movement smoother

        orn_arr = np.array(orn, dtype=np.float64)  # for quat distance calc

        for step in range(max_step):

            # apply IK
            x_tmp = alpha * x + (1 - alpha) * real_xyz[0]
            y_tmp = alpha * y + (1 - alpha) * real_xyz[1]
            z_tmp = alpha * z + (1 - alpha) * real_xyz[2]

            joint_poses = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=self.eef_id,
                targetPosition=[x_tmp, y_tmp, z_tmp],
                targetOrientation=orn,
                maxNumIterations=200)

            # Filter out the gripper
            for i, name in enumerate(self.controlJoints[:-1]):
                joint = self.joints[name]
                pose = joint_poses[i]
                # control robot end-effector
                p.setJointMotorControl2(
                    self.robot_id,
                    joint.id,
                    p.POSITION_CONTROL,
                    targetPosition=pose,
                    force=joint.maxForce,
                    maxVelocity=joint.maxVelocity
                    if custom_velocity is None else custom_velocity * (i + 1))

            self.step_simulation()
            if try_close_gripper and still_open_flag_ and not self.gripper_contact(
            ):
                still_open_flag_ = self.close_gripper(check_contact=True)

            # Check if contact with objects
            if check_collision_config and self.gripper_contact(
                    **check_collision_config):
                if self.debug:
                    print('Collision detected!', self.check_grasped_id())
                return False, p.getLinkState(self.robot_id, self.eef_id)[0:2]

            # Check convergence: position (Euclidean) + orientation (quaternion geodesic)
            real_xyz, real_xyzw = p.getLinkState(self.robot_id,
                                                 self.eef_id)[0:2]
            pos_err = np.linalg.norm(np.array((x, y, z)) - real_xyz)
            orn_err = self._quat_distance_deg(orn_arr, np.array(real_xyzw))

            if pos_err < pos_tol and orn_err < orn_tol_deg:
                if verbose:
                    print(f'Reach target in {step} steps '
                          f'(pos_err={pos_err:.4f}m, orn_err={orn_err:.2f}°)')
                return True, (real_xyz, real_xyzw)

        # Did not converge within max_step
        if self.debug:
            print(f'Failed to reach target after {max_step} steps '
                  f'(pos_err={pos_err:.4f}m, orn_err={orn_err:.1f}°)')
        return False, p.getLinkState(self.robot_id, self.eef_id)[0:2]

    def get_placement_points(self,
                             target_index: int,
                             num_samples: int = 40) -> list:
        """
        Finds points in table free space suitable for placing an object of the specified size.
        
        Parameters:
        - segmentation_mask: np.ndarray (np.int32), a mask where 0 represents table free space and unique positive integers represent objects.
        - target_index: int, the index of the target object in the segmentation mask.
        - num_samples: int, number of sample points to return.
        
        Returns:
        - List of tuples [(x1, y1), (x2, y2), ...] with coordinates in the free space where the target object can fit.
        """

        # Create a binary mask for the target object
        obs = self.get_obs()
        xyz = self.camera.get_pointcloud()
        segmentation_mask = obs['seg']
        target_mask = (segmentation_mask == target_index).astype(np.uint8)

        # Calculate bounding rectangle around the target object
        x, y, w, h = cv2.boundingRect(target_mask)
        object_size = (w, h)

        # Create a mask for the table free space
        table_free_space_mask = (segmentation_mask == 0).astype(np.uint8)

        # Find regions in the table free space that can fit the object
        kernel = np.ones(
            object_size,
            np.uint8)  # Structuring element to match the object size
        potential_placement_mask = cv2.erode(table_free_space_mask,
                                             kernel,
                                             iterations=1)

        # Get all possible placement points from the eroded mask
        placement_points = np.column_stack(
            np.where(potential_placement_mask == 1))

        # Randomly sample points, up to num_samples or the number of available points
        if len(placement_points) > 0:
            sampled_points = placement_points[np.random.choice(
                len(placement_points),
                min(num_samples, len(placement_points)),
                replace=False)]
            ys, xs = zip(*sampled_points)
            sampled_points = xyz[ys, xs, :]  # (N,3), camera frame
            sampled_points[:, 2] = -sampled_points[:, 2]
            sampled_points[:, 1] = -sampled_points[:, 1]  #convert to ref frame
            sampled_points_hom = np.dot(
                self.cam_to_robot_base,
                np.vstack(
                    [sampled_points.T,
                     np.ones((1, sampled_points.shape[0]))]))
            sampled_points = sampled_points_hom[:3, :].T

            # Return points as (x, y) tuples
            #return [(int(pt[1]), int(pt[0])) for pt in sampled_points]
            return sampled_points
        else:
            print('No available free space for placing the object.')
            # If no suitable placement points are found, return an empty list
            return []

    def grasp(self,
              pos: tuple,
              roll: float,
              gripper_opening_length: float,
              obj_height: float,
              debug: bool = False,
              vis: bool = False):
        """
        Method to perform grasp based on pose.
        pos [x y z]: The axis in real-world coordinate
        roll: float,   for grasp, it should be in [-pi/2, pi/2)
        """
        succes_grasp = False

        x, y, z = pos
        # Substracht gripper finger length from z
        z -= self.finger_length
        z = np.clip(z, *self.ee_position_limit[2])

        LID = []
        if vis:
            if len(LID) > 0:
                self.remove_drawing(LID)
                LID = []
            g = pos[0], pos[1], pos[
                2], roll, gripper_opening_length, obj_height
            LID = self.draw_predicted_grasp(g, color=[0, 1, 0], lineIDs=LID)
            self.dummy_simulation_steps(10)

        # Move above target
        #self.reset_robot()
        self.move_gripper(0.1)
        orn = p.getQuaternionFromEuler([roll, np.pi / 2, 0.0])
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # Reduce grip to get a tighter grip
        gripper_opening_length *= self.GRIP_REDUCTION

        # Grasp and lift object
        z_offset = self.calc_z_offset(gripper_opening_length)
        self.move_ee([x, y, z + z_offset, orn])
        # self.move_gripper(gripper_opening_length)
        self.auto_close_gripper(check_contact=True)
        for _ in range(40):
            self.step_simulation()
        self.move_ee([x, y, 1.25, orn])

        # check if the object has been grasped and lifted off the table
        grasped_id = self.check_grasped_id()
        grasped_obj_id = None
        if len(grasped_id) == 1:
            grasped_obj_id = grasped_id[0]
            succes_grasp = True

        if vis:
            self.remove_drawing(LID)
            self.dummy_simulation_steps(10)

        return succes_grasp, grasped_obj_id

    # ── GraspGen 6-DOF friction constants (see GraspGen_TUNING.MD) ──
    _GG_FINGER_LAT_FRIC  = 2.0
    _GG_FINGER_SPIN_FRIC = 0.5
    _GG_FINGER_ROLL_FRIC = 0.01
    _GG_OBJ_LAT_FRIC     = 1.5
    _GG_OBJ_SPIN_FRIC    = 0.3
    _GG_OBJ_ROLL_FRIC    = 0.01

    def boost_friction_for_graspgen(self, obj_id: int):
        """Increase finger-pad and object friction for GraspGen grasps.

        Called once per scene before any 6-DOF grasp attempts.
        """
        for pad in ('left_inner_finger_pad_joint',
                    'right_inner_finger_pad_joint'):
            link_id = self.joints[pad].id
            p.changeDynamics(self.robot_id, link_id,
                             lateralFriction=self._GG_FINGER_LAT_FRIC,
                             spinningFriction=self._GG_FINGER_SPIN_FRIC,
                             rollingFriction=self._GG_FINGER_ROLL_FRIC)
        p.changeDynamics(obj_id, -1,
                         lateralFriction=self._GG_OBJ_LAT_FRIC,
                         spinningFriction=self._GG_OBJ_SPIN_FRIC,
                         rollingFriction=self._GG_OBJ_ROLL_FRIC)

    def put_obj_in_tray_6dof(self,
                             obj_id: int,
                             grasps_6dof: list,
                             max_attempts: int = 5,
                             vis: bool = False):
        """Pick object with 6-DOF GraspGen grasps and place in tray.

        Uses the proven FSM from ``test_graspgen.py`` / GraspGen_TUNING.MD:
          - Approach from GRIPPER_MOVING_HEIGHT (1.25) with grasp orn.
          - Short lift (+20 cm) after close + 40-step settle.
          - Normal-speed transport keeping grasp orn throughout.
          - ``check_target_reached`` as the real success metric.
          - **No finger-length subtraction** (GraspGen base already
            accounts for full gripper depth).
          - **No orientation switch** during transport.

        Args:
            obj_id:       PyBullet body ID of the target object.
            grasps_6dof:  List of dicts with keys ``pos``, ``quat_ee``,
                          ``conf``, as returned by
                          ``RobotEnvUI.compute_grasps_graspgen()``.
            max_attempts: Number of grasps to try before giving up.

        Returns:
            (grasp_success: bool, target_success: bool)
        """
        if not grasps_6dof:
            print('[GraspGen] No grasps to try.')
            return False, False

        # Boost friction once before the attempt loop
        self.boost_friction_for_graspgen(obj_id)

        tz = self.TARGET_ZONE_POS

        for j, g in enumerate(grasps_6dof[:max_attempts]):
            bx, by, bz = g['pos']
            orn = g['quat_ee']
            z_grasp = np.clip(bz, *self.ee_position_limit[2])

            if j > 0:
                # Reset arm for a clean retry (gripper reopened)
                self.reset_robot()
                self.dummy_simulation_steps(10)
                # Re-boost friction (reset may reload dynamics)
                self.boost_friction_for_graspgen(obj_id)

            # ── GRASP (normal speed, approach from 1.25) ──
            self.move_gripper(0.1)
            self.move_ee([bx, by, self.GRIPPER_MOVING_HEIGHT, orn])
            self.move_ee([bx, by, z_grasp, orn])
            self.auto_close_gripper(check_contact=True)
            for _ in range(40):
                self.step_simulation()

            # ── SHORT LIFT (+20 cm) + settle ──
            z_over = z_grasp + 0.20
            self.move_ee([bx, by, z_over, orn])
            for _ in range(40):
                self.step_simulation()

            grasped = self.check_grasped_id()
            if not (len(grasped) == 1 and grasped[0] == obj_id):
                continue  # grasp failed, try next

            # ── TRANSPORT (normal speed, keep grasp orn) ──
            self.move_ee([tz[0], tz[1], 1.25, orn])
            self.move_ee([tz[0], tz[1], tz[2] + 0.25, orn])
            self.move_gripper(0.085)
            self.move_ee([tz[0], tz[1], self.GRIPPER_MOVING_HEIGHT, orn])

            for _ in range(20):
                self.step_simulation()

            if self.check_target_reached(obj_id):
                return True, True
            else:
                # Object grasped but didn't land in tray
                return True, False

        return False, False

    def pick_obj_by_id(self,
                       obj_id: int,
                       vis: bool = False,
                       grasp_indices: Optional[List[int]] = None):
        """
        Pick object by its state id, potentially with ranked grasp indices 
        """
        succes_grasp, grasped_obj_id = False, None
        idx = self.obj_ids.index(obj_id)
        grasps = self.obj_grasps[idx]
        assert grasps is not None, "First have to set grasps with env.set_obj_grasps()"
        if grasp_indices is not None:
            # re-ranking grasps
            grasps = [grasps[i] for i in grasp_indices]
        LID = []
        for j, g in enumerate(grasps):
            x, y, z, yaw, opening_len, obj_height = g

            if vis:
                if len(LID) > 0:
                    self.remove_drawing(LID)
                    LID = []
                LID = self.draw_predicted_grasp(g,
                                                color=[0, 1, 0],
                                                lineIDs=LID)
                self.dummy_simulation_steps(10)

            succes_grasp, grasped_obj_id = self.grasp((x, y, z), yaw,
                                                      opening_len, obj_height)
            if succes_grasp:
                #assert grasped_obj_id == obj_id, f'Grasped wrong object {self.obj_names[self.obj_ids.index(grasped_obj_id)]}'
                if grasped_obj_id != obj_id:
                    print(
                        f'Grasped wrong object {self.obj_names[self.obj_ids.index(grasped_obj_id)]}'
                    )
                    return False, obj_id, g
                if vis:
                    self.remove_drawing(LID)
                return True, obj_id, g

            if j == self.N_GRASP_ATTEMPTS:
                print(f'Exceeded {self.N_GRASP_ATTEMPTS} grasping attempts.s')
                return False, obj_id, None

            print('Grasping failed. Retrying...')

        if vis:
            self.remove_drawing(LID)
            self.dummy_simulation_steps(10)

        return succes_grasp, grasped_obj_id, None

    def place(self, target_loc: list, grasp: list, vis: bool = False):
        """
        Method to place object in exact location
        """
        try:
            grasped_obj_id = self.check_grasped_id()[0]
        except:
            print('No object currently in gripper. Exitting.')
            return False

        x, y, z, yaw, opening_len, obj_height = grasp
        # Move object to target location
        z_offset = self.calc_z_offset(opening_len)
        #y_drop = min(1.0, target_loc[2]) + z_offset + obj_height + 0.15
        y_final = min(1.0, target_loc[2]) + obj_height + z_offset
        y_orn = p.getQuaternionFromEuler([-np.pi * 0.25, np.pi / 2, 0.0])

        #self.move_arm_away()
        # self.move_ee([target_loc[0],
        #              target_loc[1], 1.25, y_orn])
        # self.move_ee([target_loc[0],
        #              target_loc[1], y_drop, y_orn])
        self.move_ee([target_loc[0], target_loc[1], y_final, y_orn])
        self.move_gripper(0.085)
        self.move_ee(
            [target_loc[0], target_loc[1], self.GRIPPER_MOVING_HEIGHT, y_orn])

        # Wait then check if object is in target zone
        for _ in range(20):
            self.step_simulation()

        succes_target = False
        if self.check_target_loc_reached(grasped_obj_id, target_loc):
            succes_target = True
            #self.remove_obj(grasped_obj_id)

        return succes_target

    def put_obj_in_loc(self,
                       obj_id: int,
                       target_loc: str,
                       vis: bool = False,
                       grasp_indices: Optional[List[int]] = None):
        """
        Method to pick object by id and place it in target location
        """
        succes_grasp, succes_target = False, False

        # pick object by id
        succes_grasp, grasped_obj_id, grasp = self.pick_obj_by_id(
            obj_id, vis=vis, grasp_indices=grasp_indices)

        if not succes_grasp:
            print('Grasping failed. Exitting')
            return False, False

        # Check if target is table region
        if isinstance(target_loc, str):
            assert target_loc in self.TARGET_LOCATIONS.keys(
            ), f'unknown location {target_loc}'
            target_loc = self.TARGET_LOCATIONS[target_loc]

        succes_target = self.place(target_loc, grasp, vis=vis)

        return succes_grasp, succes_target

    def check_target_loc_reached(self, obj_id, pos):
        if obj_id is None:
            print('Asking to check unknown object')
            return False
        (x, y, z), orn = p.getBasePositionAndOrientation(obj_id)
        if abs(pos[0] - x) < 0.1 and abs(pos[1] - y) < 0.1:
            return True
        else:
            return False

    def put_obj_in_tray(self,
                        obj_id: int,
                        debug: bool = False,
                        vis: bool = False,
                        grasp_indices: Optional[List[int]] = None):
        """
        Method to pick object by id and place it in tray location
        """
        succes_grasp, succes_target = False, False

        # pick object by id
        succes_grasp, grasped_obj_id, grasp = self.pick_obj_by_id(
            obj_id, vis=vis, grasp_indices=grasp_indices)

        if not succes_grasp:
            print('Grasping failed. Exitting')
            return False, False

        # Move object to target zone
        x, y, z, yaw, opening_len, obj_height = grasp
        z_offset = self.calc_z_offset(opening_len)
        y_drop = self.TARGET_ZONE_POS[2] + z_offset + obj_height + 0.15
        y_orn = p.getQuaternionFromEuler([-np.pi * 0.25, np.pi / 2, 0.0])

        #self.move_arm_away()
        self.move_ee(
            [self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1], 1.25, y_orn])
        self.move_ee(
            [self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1], y_drop, y_orn])
        self.move_gripper(0.085)
        self.move_ee([
            self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1],
            self.GRIPPER_MOVING_HEIGHT, y_orn
        ])

        # Wait then check if object is in target zone
        for _ in range(20):
            self.step_simulation()

        if self.check_target_reached(grasped_obj_id):
            succes_target = True
            #self.remove_obj(grasped_obj_id)

        return succes_grasp, succes_target

    def put_obj_in_free_space(self,
                              obj_id: int,
                              debug: bool = False,
                              vis: bool = False,
                              grasp_indices: Optional[List[int]] = None):
        """
        Method to pick object by id and place it in a free table location
        """
        target_pos_candidates = self.get_placement_points(obj_id)
        #target_pos_candidates = np.asarray(list(self.TARGET_LOCATIONS.values()))
        # select second closest point from object pos
        obj_pos = np.asarray(self.get_obj_pos(obj_id))
        chosen_idx = np.argsort(
            np.linalg.norm(target_pos_candidates - obj_pos, axis=1))[1]
        #chosen_idx = np.argmin(np.linalg.norm(target_pos_candidates - obj_pos, axis=1))
        target_pos = target_pos_candidates[chosen_idx]
        return self.put_obj_in_loc(obj_id,
                                   target_pos,
                                   grasp_indices=grasp_indices,
                                   vis=vis)

    def close(self):
        p.disconnect(self.physicsClient)

    def dummy_simulation_steps(self, n):
        for _ in range(n):
            p.stepSimulation()

    @staticmethod
    def draw_predicted_grasp(grasps, color=[0, 0, 1], lineIDs=[]):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(
            p.addUserDebugLine([x, y, z], [x, y, z + 0.15], color,
                               lineWidth=6))

        lineIDs.append(
            p.addUserDebugLine([
                x - gripper_size * math.sin(yaw),
                y - gripper_size * math.cos(yaw), z
            ], [
                x + gripper_size * math.sin(yaw),
                y + gripper_size * math.cos(yaw), z
            ],
                               color,
                               lineWidth=6))

        lineIDs.append(
            p.addUserDebugLine([
                x - gripper_size * math.sin(yaw),
                y - gripper_size * math.cos(yaw), z
            ], [
                x - gripper_size * math.sin(yaw),
                y - gripper_size * math.cos(yaw), z - finger_size
            ],
                               color,
                               lineWidth=6))
        lineIDs.append(
            p.addUserDebugLine([
                x + gripper_size * math.sin(yaw),
                y + gripper_size * math.cos(yaw), z
            ], [
                x + gripper_size * math.sin(yaw),
                y + gripper_size * math.cos(yaw), z - finger_size
            ],
                               color,
                               lineWidth=6))

        return lineIDs

    @staticmethod
    def remove_drawing(lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)

# Sungsu's help function
    def remove_out_of_bounds_objs(self, width=0.8, z_min=None):
        """
        Remove objects outside ROI or below table height.

        When a basket is active, uses the basket interior (with a small margin)
        as the ROI.  Otherwise uses a square of side ``width`` centered at
        ``self.obj_init_pos``.
        """
        if self.use_basket and hasattr(self, '_basket_bounds'):
            bb = self._basket_bounds
            # Allow a small margin outside the basket walls for objects
            # that are resting against a wall but protruding slightly
            margin = 0.05
            x_lo = bb["cx"] - bb["half_x"] - margin
            x_hi = bb["cx"] + bb["half_x"] + margin
            y_lo = bb["cy"] - bb["half_y"] - margin
            y_hi = bb["cy"] + bb["half_y"] + margin
        else:
            half_w = width / 2.0
            center_x, center_y = self.obj_init_pos[0], self.obj_init_pos[1]
            x_lo = center_x - half_w
            x_hi = center_x + half_w
            y_lo = center_y - half_w
            y_hi = center_y + half_w

        if z_min is None:
            z_min = self.Z_TABLE_TOP - 0.2  # slightly below table

        for obj_id in list(self.obj_ids):
            pos = self.get_obj_pos(obj_id)
            x, y, z = pos

            out_x = not (x_lo <= x <= x_hi)
            out_y = not (y_lo <= y <= y_hi)
            out_z = z < z_min

            if out_x or out_y or out_z:
                print(f"Removing object {obj_id} (out of bounds at {pos}).")
                self.remove_obj(obj_id)