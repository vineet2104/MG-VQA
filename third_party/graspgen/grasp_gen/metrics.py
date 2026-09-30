# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

# Third Party
import numpy as np
import torch
import trimesh.transformations as tra
from scipy.spatial import KDTree
from torch import Tensor, nn
from torch.autograd import Function

from grasp_gen.robot import GripperInfo
from grasp_gen.utils.rotation_conversions import matrix_to_quaternion


def compute_recall(
    pose_set_a: np.ndarray, pose_set_b: np.ndarray, radius: float = 0.02
) -> float:

    # Recompute for tighter tolerance
    tree = KDTree(pose_set_a[:, :3, 3])
    visited = set()

    for i, grasp in enumerate(pose_set_b):
        close_indexes = tree.query_ball_point(grasp[:3, 3], radius)
        visited.update([(close_index) for close_index in close_indexes])

    recall = len(visited) / len(pose_set_a)
    return recall


def compute_metrics_given_two_sets_of_poses(
    poses_A: torch.Tensor,
    poses_B: torch.Tensor,
    gripper_info: GripperInfo,
    consider_symmetry: bool = False,
) -> Dict[str, float]:
    """
    Note that poses_A and poses_B have the same batch size and shape. e.g. [N, 4, 4]
    """

    actual_noise_pts_quat = normalize_quaternion(
        matrix_to_quaternion(poses_A[:, :3, :3])
    )
    pred_noise_pts_quat = normalize_quaternion(matrix_to_quaternion(poses_B[:, :3, :3]))

    phi3 = angular_distance_phi3(actual_noise_pts_quat, pred_noise_pts_quat)
    criterion = GeodesicLoss(reduction="none")
    geodesic_dist = criterion(poses_A[:, :3, :3], poses_B[:, :3, :3])
    device = geodesic_dist.device

    if gripper_info.symmetric and consider_symmetry:

        poses_A_mirror = np.array(
            [
                g @ tra.euler_matrix(0, 0, np.pi)
                for g in poses_A.clone().detach().cpu().numpy()
            ]
        )
        poses_A_mirror = torch.from_numpy(poses_A_mirror).to(geodesic_dist.device)
        geodesic_dist_mirror = criterion(poses_A_mirror[:, :3, :3], poses_B[:, :3, :3])

        geodesic_dist = torch.vstack([geodesic_dist_mirror, geodesic_dist])
        geodesic_dist = geodesic_dist.min(axis=0)[0]

        actual_noise_pts_quat = normalize_quaternion(
            matrix_to_quaternion(poses_A_mirror[:, :3, :3])
        )
        phi3_mirror = angular_distance_phi3(actual_noise_pts_quat, pred_noise_pts_quat)
        phi3 = torch.vstack([phi3_mirror, phi3])
        phi3 = phi3.min(axis=0)[0]

    phi3 = phi3.mean()
    geodesic_dist = geodesic_dist.mean()

    depth = gripper_info.depth
    # TODO - read this depth value from cfg file
    poses_A_shifted = np.array(
        [
            g @ tra.translation_matrix([0, 0, depth])
            for g in poses_A.clone().detach().cpu().numpy()
        ]
    )
    poses_B_shifted = np.array(
        [
            g @ tra.translation_matrix([0, 0, depth])
            for g in poses_B.clone().detach().cpu().numpy()
        ]
    )

    actual_noise_pts_t = poses_A_shifted[:, :3, 3]
    pred_noise_pts_t = poses_B_shifted[:, :3, 3]
    error = torch.tensor(actual_noise_pts_t - pred_noise_pts_t)
    translation_error = torch.linalg.norm(error, dim=1).mean().to(geodesic_dist.device)
    stats = {
        "error_trans_l2": translation_error,
        "error_rot_geodesic": geodesic_dist,
        "error_rot_phi3": phi3,
    }
    return stats


def angular_distance_phi3(
    goal_quat: torch.Tensor, current_quat: torch.Tensor
) -> torch.Tensor:
    """This function computes the angular distance phi_3.

    See Huynh, Du Q. "Metrics for 3D rotations: Comparison and analysis." Journal of Mathematical
    Imaging and Vision 35 (2009): 155-164.

    Args:
        goal_quat: _description_
        current_quat: _description_

    Returns:
        Angular distance in range [0,1]
    """
    dot_prod = (
        goal_quat[..., 0] * current_quat[..., 0]
        + goal_quat[..., 1] * current_quat[..., 1]
        + goal_quat[..., 2] * current_quat[..., 2]
        + goal_quat[..., 3] * current_quat[..., 3]
    )

    dot_prod = torch.abs(dot_prod)
    distance = dot_prod
    distance = torch.arccos(dot_prod) / (torch.pi * 0.5)
    return distance


def quat_multiply(
    q1: torch.Tensor, q2: torch.Tensor, q_res: torch.Tensor
) -> torch.Tensor:
    a_w = q1[..., 0]
    a_x = q1[..., 1]
    a_y = q1[..., 2]
    a_z = q1[..., 3]
    b_w = q2[..., 0]
    b_x = q2[..., 1]
    b_y = q2[..., 2]
    b_z = q2[..., 3]

    q_res[..., 0] = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z

    q_res[..., 1] = a_w * b_x + b_w * a_x + a_y * b_z - b_y * a_z
    q_res[..., 2] = a_w * b_y + b_w * a_y + a_z * b_x - b_z * a_x
    q_res[..., 3] = a_w * b_z + b_w * a_z + a_x * b_y - b_x * a_y
    return q_res


class OrientationError(Function):
    @staticmethod
    def geodesic_distance(goal_quat, current_quat, quat_res):
        conjugate_quat = current_quat.clone()
        conjugate_quat[..., 1:] *= -1.0
        quat_res = quat_multiply(goal_quat, conjugate_quat, quat_res)

        quat_res = -1.0 * quat_res * torch.sign(quat_res[..., 0]).unsqueeze(-1)
        quat_res[..., 0] = 0.0
        # quat_res = conjugate_quat * 0.0
        return quat_res

    @staticmethod
    def forward(ctx, goal_quat, current_quat, quat_res):
        quat_res = OrientationError.geodesic_distance(goal_quat, current_quat, quat_res)
        rot_error = torch.norm(quat_res, dim=-1, keepdim=True)
        ctx.save_for_backward(quat_res, rot_error)
        return rot_error

    @staticmethod
    def backward(ctx, grad_out):
        grad_mul = None
        if ctx.needs_input_grad[1]:
            (quat_error, r_err) = ctx.saved_tensors
            scale = 1 / r_err
            scale = torch.nan_to_num(scale, 0, 0, 0)

            grad_mul = grad_out * scale * quat_error
            # print(grad_out.shape)
            # if grad_out.shape[0] == 6:
            #    #print(grad_out.view(-1))
            #    #print(grad_mul.view(-1)[-6:])
            #    #exit()
        return None, grad_mul, None


def normalize_quaternion(in_quaternion: torch.Tensor) -> torch.Tensor:
    k = torch.sign(in_quaternion[..., 0:1])
    # NOTE: torch sign returns 0 as sign value when value is 0.0
    k = torch.where(k == 0, 1.0, k)
    k2 = k / torch.linalg.norm(in_quaternion, dim=-1, keepdim=True)
    # normalize quaternion
    in_q = k2 * in_quaternion
    return in_q


class GeodesicLoss(nn.Module):
    r"""Creates a criterion that measures the distance between rotation matrices, which is
    useful for pose estimation problems.
    The distance ranges from 0 to :math:`pi`.
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices and:
    "Metrics for 3D Rotations: Comparison and Analysis" (https://link.springer.com/article/10.1007/s10851-009-0161-2).

    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.

    The loss can be described as:

    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)

    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.

        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: Shape :math:`(N, 3, 3)`.
        - Target: Shape :math:`(N, 3, 3)`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`. Otherwise, scalar.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.double()
        target = target.double()
        R_diffs = input @ target.permute(0, 2, 1)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()
