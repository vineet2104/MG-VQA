import torch
import numpy as np
import trimesh
import os
from grasp_gen.robot import load_control_points_core, load_default_gripper_config
from pathlib import Path

class GripperModel(object):
    def __init__(self, data_root_dir=None):
        if data_root_dir is None:
            data_root_dir = f'{Path(__file__).parent.parent.parent}/assets/robotiq'
        fn_base = data_root_dir + "/robotiq_140_collision.obj"
        self.mesh = trimesh.load(fn_base)

    def get_gripper_collision_mesh(self):
        return self.mesh

    def get_gripper_visual_mesh(self):
        return self.mesh

def get_gripper_offset_bins():

    offset_bins = [
        0.0, 0.01360345836, 0.02720691672,
        0.04081037508, 0.05441383344, 0.0680172918,
        0.08162075016, 0.09522420852, 0.10882766688,
        0.12243112524000001, 0.1360345836
    ]

    offset_bin_weights = [
        0.16652107, 0.21488856, 0.37031708, 0.55618503, 0.75124664,
        0.93943357, 1.07824539, 1.19423112, 1.55731375, 3.17161779
    ]
    return offset_bins, offset_bin_weights

def load_control_points() -> torch.Tensor:
    """
    Load the control points for the gripper, used for training.
    Returns a tensor of shape (4, N) where N is the number of control points.
    """
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    control_points = torch.from_numpy(control_points).float()
    return control_points.T

def load_control_points_for_visualization():

    gripper_config = load_default_gripper_config(Path(__file__).stem)

    control_points = load_control_points_core(gripper_config)

    mid_point = (control_points[0] + control_points[1]) / 2

    control_points = [
        control_points[-2], control_points[0], mid_point,
        [0, 0, 0], mid_point, control_points[1], control_points[-1]
    ]
    return [control_points, ]