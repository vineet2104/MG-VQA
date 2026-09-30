"""Contact-based robot manipulation (no object-to-gripper attachment)."""
from .primitives import (
    execute_grasp_and_remove, execute_push, set_finger_offset, set_num_pc_points,
)
__all__ = ['execute_grasp_and_remove', 'execute_push', 'set_finger_offset', 'set_num_pc_points']
