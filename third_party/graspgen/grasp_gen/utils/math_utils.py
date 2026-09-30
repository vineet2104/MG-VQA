# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
import trimesh.transformations as tra
from scipy.optimize import linear_sum_assignment

import grasp_gen.utils.rotation_conversions as rotation_conversions
import grasp_gen.utils.so3 as so3


def matrix_to_rt(
    input: torch.Tensor, grasp_repr: str, kappa: Optional[float] = None
) -> torch.Tensor:
    grasps_r_mat = input[:, :3, :3]
    grasps_t = input[:, :3, 3]

    if grasp_repr == "r3_6d":
        grasps_r = matrix_to_rotation_6d(grasps_r_mat)
    elif grasp_repr == "r3_so3":
        grasps_r = so3.so3_log_map(grasps_r_mat)
        noise_scale_rot = 1 / torch.pi
        grasps_r = grasps_r * noise_scale_rot  # Scale range to [-1, 1]
    elif grasp_repr == "r3_euler":
        grasps_r = rotation_conversions.matrix_to_euler_angles(
            grasps_r_mat, convention="XYZ"
        )
        noise_scale_rot = 1 / torch.pi
        grasps_r = grasps_r * noise_scale_rot  # Scale range to [-1, 1]
    else:
        raise NotImplementedError(f"Representation for rotation {grasp_repr} unknown!")

    if kappa is not None:
        grasps_t = kappa * grasps_t
    grasps_gt = torch.hstack([grasps_t, grasps_r]).float()
    return grasps_gt


def rt_to_matrix(
    input: torch.Tensor, grasp_repr: str, kappa: Optional[float] = None
) -> torch.Tensor:
    batch_size = input.shape[0]
    mat = torch.zeros([batch_size, 4, 4]).to(device=input.device)

    if grasp_repr == "r3_6d":
        mat[:, :3, :3] = rotation_6d_to_matrix(input[:, 3:])
    elif grasp_repr == "r3_so3":
        mat[:, :3, :3] = so3.so3_exp_map(input[:, 3:] * torch.pi)
    elif grasp_repr == "r3_euler":
        mat[:, :3, :3] = rotation_conversions.euler_angles_to_matrix(
            input[:, 3:] * torch.pi, convention="XYZ"
        )
    else:
        raise NotImplementedError(
            f"Rotation representation {grasp_repr} is not implemented!"
        )

    mat[:, :3, 3] = input[:, :3]
    if kappa is not None:
        inv_kappa = 1.0 / kappa
        mat[:, :3, 3] *= inv_kappa

    return mat


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Original implementation from https://github.com/facebookresearch/pytorch3d

    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Original implementation from https://github.com/facebookresearch/pytorch3d

    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def rotation_from_vectors(v1, v2):
    """
    Uses Rodrigues' formula to compute the [3x3] rotation matrix
    to align vector v1 onto v2.
    """
    u = v1 / np.linalg.norm(v1)
    Ru = v2 / np.linalg.norm(v2)
    dim = u.size
    I = np.identity(dim)
    c = np.dot(u, Ru)
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        # same direction
        return I
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        return -I
    else:
        K = np.outer(Ru, u) - np.outer(u, Ru)
        return I + K + (K @ K) / (1 + c)


def construct_suction_grasp_from_point_and_vector(
    contact_point, approach_vector, normal_vector, offset=-0.005
):
    """
    Computes grasp pose given contact point, normal vector and approach vector.
    The offset is needed to avoid collisions with the collision checker
    """

    # Align the gripper approach vector with the contact point surface normal
    T_rotate_gripper_to_surface_normal = np.eye(4)
    T_rotate_gripper_to_surface_normal[:3, :3] = rotation_from_vectors(
        approach_vector, normal_vector
    )

    T_rotate_gripper_to_surface_normal[:3, :3] = rotation_matrix_from_vectors(
        normal_vector, approach_vector
    )
    T_rotate_gripper_to_surface_normal = (
        T_rotate_gripper_to_surface_normal @ tra.euler_matrix(np.pi, 0, 0)
    )

    # NOTE: This is to prevent collisions (as in the default state the gripper collides with the object)
    T_approach_offset = tra.translation_matrix([0, 0, offset])

    T_translate_gripper_to_contact_point = tra.translation_matrix(contact_point)
    grasp_pose = (
        T_translate_gripper_to_contact_point
        @ T_rotate_gripper_to_surface_normal
        @ T_approach_offset
    )
    return grasp_pose


def rotation_matrix_from_vectors(
    v1: Union[torch.Tensor, np.ndarray, List], v2: Union[torch.Tensor, np.ndarray, List]
) -> torch.Tensor:
    """
    Compute the rotation matrix that aligns vector v1 onto vector v2.

    This function uses the axis-angle representation to compute the rotation matrix
    that rotates vector v1 to align with vector v2. The rotation is computed using
    the cross product to find the rotation axis and the dot product to find the angle.

    Args:
        v1: First vector to align from. Can be torch.Tensor, np.ndarray, or List.
        v2: Second vector to align to. Can be torch.Tensor, np.ndarray, or List.

    Returns:
        torch.Tensor: 3x3 rotation matrix that rotates v1 to align with v2.
                     The matrix is computed using axis-angle representation.

    Note:
        Both input vectors are normalized to unit vectors before computing the rotation.
        The function handles edge cases where vectors are parallel or anti-parallel.
    """
    if type(v1) in [np.ndarray, list]:
        v1 = torch.tensor(v1)
    if type(v2) in [np.ndarray, list]:
        v2 = torch.tensor(v2)

    v1 = v1.float()
    v2 = v2.float()

    # Ensure v1 and v2 are unit vectors
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)

    # Compute the cross product (axis of rotation)
    axis = torch.cross(v1, v2)

    # Compute the dot product (cosine of the angle)
    cos_theta = torch.sum(v1 * v2, dim=-1, keepdim=True)
    sin_theta = axis.norm(dim=-1, keepdim=True)

    # Normalize the axis to get the unit vector for the axis
    axis = axis / sin_theta

    # Compute the angle of rotation
    angle = torch.atan2(sin_theta, cos_theta)

    # Convert the axis-angle to a rotation matrix
    rotation_matrix = axis_angle_to_matrix(axis * angle)

    return rotation_matrix


def compute_pose_distance_batch(
    poses1: torch.Tensor, poses2: torch.Tensor
) -> torch.Tensor:
    """
    Compute distances between two sets of poses in a batched manner.

    Args:
        poses1: First set of poses [N1, 4, 4]
        poses2: Second set of poses [N2, 4, 4]

    Returns:
        torch.Tensor: Distance matrix [N1, N2]
    """
    # Extract positions [N1, 3] and [N2, 3]
    pos1 = poses1[:, :3, 3]  # [N1, 3]
    pos2 = poses2[:, :3, 3]  # [N2, 3]

    # Compute pairwise position distances
    pos1_expanded = pos1.unsqueeze(1)  # [N1, 1, 3]
    pos2_expanded = pos2.unsqueeze(0)  # [1, N2, 3]
    pos_dist = torch.norm(pos1_expanded - pos2_expanded, dim=2)  # [N1, N2]

    # Extract rotation matrices [N1, 3, 3] and [N2, 3, 3]
    R1 = poses1[:, :3, :3]  # [N1, 3, 3]
    R2 = poses2[:, :3, :3]  # [N2, 3, 3]

    # Compute relative rotations
    R1_expanded = R1.unsqueeze(1)  # [N1, 1, 3, 3]
    R2_expanded = R2.unsqueeze(0)  # [1, N2, 3, 3]
    R_diff = torch.matmul(R1_expanded, R2_expanded.transpose(-2, -1))  # [N1, N2, 3, 3]

    # Reshape for so3_log_map
    N1, N2 = R_diff.shape[:2]
    R_diff_flat = R_diff.reshape(-1, 3, 3)

    # Apply so3_log_map and compute rotation distances
    log_maps = so3.so3_log_map(R_diff_flat)  # [-1, 3]
    rot_dist = torch.norm(log_maps, dim=1).reshape(N1, N2) / torch.pi

    # Combine distances with equal weights
    return pos_dist + rot_dist


def compute_pose_emd(poses1: torch.Tensor, poses2: torch.Tensor) -> float:
    """
    Compute EMD between two sets of poses with equal weighting of position and rotation.

    Args:
        poses1: First set of poses [N1, 4, 4]
        poses2: Second set of poses [N2, 4, 4]

    Returns:
        float: EMD distance considering both position and rotation equally
    """
    # Ensure input is torch tensor
    if isinstance(poses1, np.ndarray):
        poses1 = torch.from_numpy(poses1).float()
    if isinstance(poses2, np.ndarray):
        poses2 = torch.from_numpy(poses2).float()

    # Compute cost matrix using vectorized operations
    M = compute_pose_distance_batch(poses1, poses2)

    # Convert to numpy for linear_sum_assignment
    M_np = M.cpu().numpy()

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(M_np)

    # Compute the total cost and normalize
    emd = M_np[row_ind, col_ind].sum() / len(row_ind)

    return float(emd)
