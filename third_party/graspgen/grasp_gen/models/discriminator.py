#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grasp_gen.dataset.dataset import MAPPING_ID2NAME
from grasp_gen.utils.math_utils import matrix_to_rt
from grasp_gen.models.model_utils import (
    PointNetPlusPlus,
    SinusoidalPosEmb,
    convert_to_ptv3_pc_format,
    load_pretrained_checkpoint_to_dict,
    offset2batch,
)
from grasp_gen.models.ptv3.ptv3 import PointTransformerV3
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.logging_config import get_logger
from grasp_gen.models.model_utils import load_pretrained_checkpoint_to_dict

logger = get_logger(__name__)


class GraspGenDiscriminator(nn.Module):
    """Neural network module for discriminating between good and bad grasps.

    This class implements a discriminator that evaluates the quality of generated grasps
    based on object geometry and grasp pose.

    Args:
        num_obs_dim (int): Dimension of observation features. Default: 512
        obs_backbone (str): Type of observation encoder backbone. Default: 'vit'
        grasp_repr (str): Grasp representation type. Default: 'r3_6d'
        grid_size (float): Grid size for point cloud processing. Default: 0.01
        sample_embed_dim (int): Dimension for grasp embeddings. Default: 512
        pose_repr (str): Type of pose representation. Default: False
        topk_ratio (float): Ratio of top grasps to consider. Default: 0.40
        checkpoint_object_encoder_pretrained (str): Path to pretrained encoder. Default: None
        kappa (float): Scale factor for noise. See calculate_dataset_kappa function in dataset.py for more details. Compute for each grasp dataset.
    """

    def __init__(
        self,
        num_obs_dim: int = 512,
        obs_backbone: str = "vit",
        grasp_repr: str = "r3_6d",
        grid_size: float = 0.01,
        sample_embed_dim: int = 512,
        pose_repr: str = False,
        topk_ratio: float = 0.40,
        checkpoint_object_encoder_pretrained: str = None,
        kappa: float = 3.30,
        gripper_name: str = "franka_panda",
    ):
        super().__init__()

        self.num_obs_dim = num_obs_dim
        self.obs_backbone = obs_backbone
        self.grasp_repr = grasp_repr
        self.grid_size = grid_size
        self.pose_repr = pose_repr
        self.topk_ratio = topk_ratio
        self.checkpoint_object_encoder_pretrained = checkpoint_object_encoder_pretrained
        self.kappa = None if kappa <= 0 else kappa
        self.gripper_name = gripper_name

        if self.grasp_repr == "r3_6d":
            self.output_dim = 9
        elif self.grasp_repr in ["r3_so3", "r3_euler"]:
            self.output_dim = 6
        else:
            raise NotImplementedError(
                f"Rotation representation {grasp_repr} is not implemented!"
            )

        if self.obs_backbone == "pointnet":
            self.object_encoder = PointNetPlusPlus(
                output_embedding_dim=self.num_obs_dim,
                feature_dim=1 if self.pose_repr == "pc_feature" else -1,
            )
        elif self.obs_backbone == "ptv3":
            self.object_encoder = PointTransformerV3(
                in_channels=3,
                enable_flash=False,
                cls_mode=True,
            )
        else:
            raise NotImplementedError()

        gripper_info = get_gripper_info(self.gripper_name)
        self.ctr_pts = gripper_info.control_points

        logger.info(f"Pose representation is {self.pose_repr}")
        total_input_dim = sample_embed_dim + num_obs_dim
        if self.pose_repr == "mlp":
            self.sample_encoder = nn.Sequential(
                nn.Linear(self.output_dim, sample_embed_dim),
                nn.ReLU(),
                nn.Linear(sample_embed_dim, sample_embed_dim),
            )
        elif self.pose_repr == "grasp_cloud":
            num_input_dim = 3 * self.ctr_pts.shape[1]
            self.sample_encoder = nn.Sequential(
                nn.Linear(num_input_dim, sample_embed_dim),
                nn.ReLU(),
                nn.Linear(sample_embed_dim, sample_embed_dim),
            )
        elif self.pose_repr == "grasp_cloud_pe":
            num_input_dim = 3 * self.ctr_pts.shape[1]
            num_embed_dim_per_dim = int(sample_embed_dim / num_input_dim)
            self.sample_encoder = nn.Sequential(
                SinusoidalPosEmb(num_embed_dim_per_dim),
                nn.Linear(num_embed_dim_per_dim * num_input_dim, sample_embed_dim),
                nn.ReLU(),
                nn.Linear(sample_embed_dim, sample_embed_dim),
                nn.ReLU(),
                nn.Linear(sample_embed_dim, sample_embed_dim),
            )
        elif self.pose_repr == "pe":
            num_embed_dim_per_dim = int(sample_embed_dim / self.output_dim)
            self.sample_encoder = nn.Sequential(
                SinusoidalPosEmb(num_embed_dim_per_dim),
                nn.Linear(num_embed_dim_per_dim * self.output_dim, sample_embed_dim),
                nn.ReLU(),
                nn.Linear(sample_embed_dim, sample_embed_dim),
                nn.ReLU(),
                nn.Linear(sample_embed_dim, sample_embed_dim),
            )
        elif self.pose_repr == "pc_feature":
            total_input_dim = num_obs_dim
        else:
            raise NotImplementedError(
                f"Pose input representation {self.pose_repr} is not implemented"
            )

        self.prediction_head = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Linear(total_input_dim // 2, total_input_dim // 4),
            nn.ReLU(),
            nn.Linear(total_input_dim // 4, 1),
        )

        if self.checkpoint_object_encoder_pretrained is not None:
            if os.path.exists(self.checkpoint_object_encoder_pretrained):
                model_state_dict_object_encoder = load_pretrained_checkpoint_to_dict(
                    self.checkpoint_object_encoder_pretrained, "object_encoder"
                )
                self.object_encoder.load_state_dict(model_state_dict_object_encoder)
                for param in self.object_encoder.parameters():
                    param.requires_grad = False
                logger.info("Using pretrained object encoder!")
            else:
                logger.info(
                    f"Object encoder checkpoints not found at location {self.checkpoint_object_encoder_pretrained}"
                )

    @classmethod
    def from_config(cls, cfg):
        """Creates a GraspGenDiscriminator instance from a configuration object.

        Args:
            cfg: Configuration object containing model parameters

        Returns:
            GraspGenDiscriminator: Instantiated model
        """
        args = {
            "num_obs_dim": cfg.num_obs_dim,
            "obs_backbone": cfg.obs_backbone,
            "grasp_repr": cfg.grasp_repr,
            "grid_size": cfg.ptv3.grid_size,
            "sample_embed_dim": cfg.num_obs_dim,
            "pose_repr": cfg.pose_repr,
            "topk_ratio": cfg.topk_ratio,
            "checkpoint_object_encoder_pretrained": cfg.checkpoint_object_encoder_pretrained,
            "kappa": cfg.kappa,
            "gripper_name": cfg.gripper_name,
        }
        return cls(**args)

    def forward(self, data, cfg=None, eval=False):
        """Forward pass of the discriminator.

        Args:
            data: Input data dictionary containing point clouds and grasps
            cfg: Optional configuration object
            eval (bool): Whether to run in evaluation mode

        Returns:
            tuple: (outputs, losses, stats) containing grasp scores, losses and metrics
        """
        device = data["points"].device

        if "grasp_key" not in data:
            grasp_key = "grasps"
        else:
            grasp_key = data["grasp_key"]
        grasps = data[grasp_key]

        num_objects_in_batch = len(data["points"])
        num_grasps_per_batch = data[grasp_key][0].shape[0]
        batch_size = num_objects_in_batch * num_grasps_per_batch
        depth = data["points"]

        num_points = depth.shape[-2]
        depth = depth.reshape([-1, num_points, 3])

        if type(grasps) == list:
            grasps = torch.cat(grasps)

        grasps = grasps.reshape([-1, 4, 4])

        if self.kappa is not None:
            depth = self.kappa * depth

        if self.obs_backbone == "ptv3":
            depth = convert_to_ptv3_pc_format(depth, grid_size=self.grid_size)

        grasps_input = matrix_to_rt(grasps, self.grasp_repr, kappa=self.kappa)

        offset = (
            torch.tensor([num_grasps_per_batch])
            .repeat(num_objects_in_batch)
            .cumsum(dim=0)
            .to(device)
        )
        mask_batch = offset2batch(offset)

        if self.pose_repr in ["grasp_cloud", "grasp_cloud_pe", "pc_feature"]:
            ctrl_pts = self.ctr_pts.to(device=device)
            grasp_pc = (grasps @ ctrl_pts).transpose(-2, -1)[..., :3]
            grasps_input = grasp_pc.reshape([batch_size, -1])

        if self.pose_repr == "pc_feature":
            depth_full = depth[mask_batch]
            depth_full = torch.cat([depth_full, grasp_pc], dim=1)
            pc_feature = torch.cat(
                [
                    torch.zeros(
                        [num_grasps_per_batch * num_objects_in_batch, num_points, 1]
                    ),
                    torch.ones(
                        [
                            num_grasps_per_batch * num_objects_in_batch,
                            grasp_pc.shape[1],
                            1,
                        ]
                    ),
                ],
                dim=1,
            ).to(device=device)

            total_embedding = torch.cat([depth_full, pc_feature], dim=-1)
            total_embedding = self.object_encoder(total_embedding)
        else:
            sample_embedding = self.sample_encoder(grasps_input)
            object_embedding = self.object_encoder(
                depth
            )  # object_embedding size is [num_objects_in_batch, self.num_obs_dim]
            object_embedding = object_embedding[
                mask_batch
            ]  # Redistribute object embeddings to full batch, result is [batch_size, self.num_obs_dim]

            total_embedding = torch.cat([sample_embedding, object_embedding], axis=-1)

        logits = self.prediction_head(total_embedding)

        losses, outputs, stats = {}, {}, {}
        outputs["logits"] = logits.reshape(
            [num_objects_in_batch, num_grasps_per_batch, 1]
        )
        outputs["grasp_confidence"] = outputs["logits"].sigmoid()

        if "labels" in data:
            labels = data["labels"]
            if type(labels) == list:
                labels = torch.cat(labels)
            bce_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels, reduction="none"
            )
            ratio_topk = self.topk_ratio
            num_top_k = int(batch_size * ratio_topk)
            bce_topk, mask = bce_loss.topk(num_top_k, dim=0)
            losses["bce_topk"] = (1.0, bce_topk.mean())

            pred = logits.sigmoid()
            from sklearn.metrics import average_precision_score

            labels = labels.squeeze(1).cpu().numpy()
            score = pred.squeeze(1).detach().cpu().numpy()
            ap = average_precision_score(labels, score)
            stats["ap"] = torch.tensor(ap).to(device)

            if "grasp_ids" in data:
                grasp_ids = data["grasp_ids"]
                if type(grasp_ids) == list:
                    grasp_ids = torch.cat(grasp_ids)
                    grasp_ids = grasp_ids.cpu().squeeze(1).numpy()

                mask = mask.cpu().numpy()
                grasp_ids_topk = grasp_ids[mask]

                for grasp_id, grasp_name in MAPPING_ID2NAME.items():

                    mask = grasp_ids_topk == grasp_id
                    val = mask.sum() / num_top_k
                    key = f"topk_ratio_{grasp_name}"
                    stats[key] = torch.tensor(val).to(device)

                    mask = grasp_ids == grasp_id
                    # stats[f"ap_{grasp_name}"] = average_precision_score(labels[mask], score[mask])
                    key = f"loss_{grasp_name}"
                    val = bce_loss[mask].detach().cpu().numpy().mean()
                    stats[key] = torch.tensor(val).to(device)

        return outputs, losses, stats

    def infer(self, data):
        """Inference method for evaluating grasps.

        Args:
            data: Input data dictionary containing point clouds and grasps

        Returns:
            tuple: (outputs, losses, stats) containing grasp scores and metrics
        """
        outputs, losses, stats = self.forward(data)
        data.update(outputs)
        return data, losses, stats
