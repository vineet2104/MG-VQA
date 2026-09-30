# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Adithya Murali
"""
Utility functions for visualization using meshcat.
"""
from typing import Dict, List, Optional, Tuple, Union

import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import numpy as np
import trimesh
import trimesh.transformations as tra

from grasp_gen.robot import load_control_points_for_visualization
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


def is_rotation_matrix(M, tol=1e-4):
    tag = False
    I = np.identity(M.shape[0])

    if (np.linalg.norm((np.matmul(M, M.T) - I)) < tol) and (
        np.abs(np.linalg.det(M) - 1) < tol
    ):
        tag = True

    if tag is False:
        logger.info("M @ M.T:\n", np.matmul(M, M.T))
        logger.info("det:", np.linalg.det(M))

    return tag


def get_color_from_score(labels, use_255_scale=False):
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in [np.float32, float]:
        return scale * np.array([1 - labels, labels, 0])
    else:
        scale = 255.0 if use_255_scale else 1.0
        score = scale * np.stack(
            [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
            axis=1,
        )
        return score.astype(np.int32)


def trimesh_to_meshcat_geometry(
    mesh: trimesh.Trimesh,
) -> meshcat.geometry.TriangularMeshGeometry:
    """
    Args:
        mesh: trimesh.TriMesh object
    """

    return meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces)


def visualize_mesh(
    vis: meshcat.Visualizer,
    name: str,
    mesh: trimesh.Trimesh,
    color: Optional[List[int]] = None,
    transform: Optional[np.ndarray] = None,
):
    """Visualize a mesh in meshcat"""
    if vis is None:
        return

    if color is None:
        color = np.random.randint(low=0, high=256, size=3)

    mesh_vis = trimesh_to_meshcat_geometry(mesh)
    color_hex = rgb2hex(tuple(color))
    material = meshcat.geometry.MeshPhongMaterial(color=color_hex)
    vis[name].set_object(mesh_vis, material)

    if transform is not None:
        vis[name].set_transform(transform)


def rgb2hex(rgb: Tuple[int, int, int]) -> str:
    """
    Converts rgb color to hex

    Args:
        rgb: color in rgb, e.g. (255,0,0)
    """
    return "0x%02x%02x%02x" % (rgb)


def create_visualizer(clear=True):
    logger.info(
        "Waiting for meshcat server... have you started a server? Run `meshcat-server` in a separate terminal to start a server."
    )
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    if clear:
        vis.delete()
    logger.info("")
    return vis


def make_frame(
    vis: meshcat.Visualizer,
    name: str,
    h: float = 0.15,
    radius: float = 0.01,
    o: float = 1.0,
    T: Optional[np.ndarray] = None,
):
    """Add a red-green-blue triad to the Meschat visualizer.
    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]["x"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8, opacity=o),
    )
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]["x"].set_transform(rotate_x)

    vis[name]["y"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8, opacity=o),
    )
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]["y"].set_transform(rotate_y)

    vis[name]["z"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000FF, reflectivity=0.8, opacity=o),
    )
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]["z"].set_transform(rotate_z)

    if T is not None:
        is_valid = is_rotation_matrix(T[:3, :3])

        if not is_valid:
            raise ValueError("meshcat_utils:attempted to visualize invalid transform T")

        vis[name].set_transform(T)


def visualize_bbox(
    vis: meshcat.Visualizer,
    name: str,
    dims: np.ndarray,
    T: Optional[np.ndarray] = None,
    color: Optional[List[int]] = [255, 0, 0],
):
    """Visualize a bounding box using a wireframe.

    Args:
        vis (MeshCat Visualizer): the visualizer
        name (string): name for this frame (should be unique)
        dims (array-like): shape (3,), dimensions of the bounding box
        T (4x4 numpy.array): (optional) transform to apply to this geometry

    """
    if vis is None:
        return
    color_hex = rgb2hex(tuple(color))
    material = meshcat.geometry.MeshBasicMaterial(wireframe=True, color=color_hex)
    bbox = meshcat.geometry.Box(dims)
    vis[name].set_object(bbox, material)

    if T is not None:
        vis[name].set_transform(T)


def visualize_pointcloud(
    vis: meshcat.Visualizer,
    name: str,
    pc: np.ndarray,
    color: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    **kwargs,
):
    """
    Args:
        vis: meshcat visualizer object
        name: str
        pc: Nx3 or HxWx3
        color: (optional) same shape as pc[0 - 255] scale or just rgb tuple
        transform: (optional) 4x4 homogeneous transform
    """
    if vis is None:
        return
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)
        # Resize the color np array if needed.
        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])
        if color.ndim == 1:
            color = np.ones_like(pc) * np.array(color)

        # Divide it by 255 to make sure the range is between 0 and 1,
        color = color.astype(np.float32) / 255
    else:
        color = np.ones_like(pc)

    vis[name].set_object(
        meshcat.geometry.PointCloud(position=pc.T, color=color.T, **kwargs)
    )

    if transform is not None:
        vis[name].set_transform(transform)


def load_visualization_gripper_points(
    gripper_name: str = "franka_panda",
) -> List[np.ndarray]:
    """
    Need to return np.array of control points of shape [4, N], where is N is num points
    """

    ctrl_points = []

    for ctrl_pts in load_control_points_for_visualization(gripper_name):

        ctrl_pts = np.array(ctrl_pts, dtype=np.float32)
        ctrl_pts = np.hstack([ctrl_pts, np.ones([len(ctrl_pts), 1])])
        ctrl_pts = ctrl_pts.T

        ctrl_points.append(ctrl_pts)

    return ctrl_points


def visualize_grasp(
    vis: meshcat.Visualizer,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    gripper_name: str = "franka_panda",
    **kwargs,
):

    if vis is None:
        return
    grasp_vertices = load_visualization_gripper_points(gripper_name)
    for i, grasp_vertex in enumerate(grasp_vertices):
        vis[name + f"/{i}"].set_object(
            g.Line(
                g.PointsGeometry(grasp_vertex),
                g.MeshBasicMaterial(color=rgb2hex(tuple(color)), **kwargs),
            )
        )
        vis[name].set_transform(transform.astype(float))


def get_normals_from_mesh(
    mesh: trimesh.Trimesh, contact_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.neighbors import KDTree

    points_codebook, index = mesh.sample(16000, return_index=True)
    normals_codebook = mesh.face_normals[index]

    contact_radius = 0.005

    tree = KDTree(points_codebook)
    dist, idx = tree.query(contact_pts)
    matched = dist < contact_radius
    idx2 = idx[matched]
    normals = normals_codebook[idx2]
    mask = matched[:, 0]
    return normals, contact_pts[mask], mask
