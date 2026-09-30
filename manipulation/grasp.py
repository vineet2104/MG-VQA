"""
manipulation.grasp — 6-DOF grasping with GraspGen.

Functions:
  - execute_grasp_and_remove: 3-retry grasp → tray placement
  - set_finger_offset: Configure gripper finger rotation (deprecated)
  - set_num_pc_points: Point cloud subsample cap for GraspGen
"""

from .primitives import (
    execute_grasp_and_remove,
    set_finger_offset,
    set_num_pc_points,
)

__all__ = ["execute_grasp_and_remove", "set_finger_offset", "set_num_pc_points"]
