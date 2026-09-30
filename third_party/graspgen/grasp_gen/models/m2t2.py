# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
"""
Top-level M2T2 network.
"""
import torch
import torch.nn as nn

from grasp_gen.models.action_decoder import ActionDecoder, infer_placements
from grasp_gen.models.contact_decoder import ContactDecoder
from grasp_gen.models.criterion import GraspCriterion, PlaceCriterion, SetCriterion
from grasp_gen.models.matcher import HungarianMatcher
from grasp_gen.models.model_utils import repeat_new_axis
from grasp_gen.models.pointnet.pointnet2 import PointNet2MSG, PointNet2MSGCls


class M2T2(nn.Module):
    def __init__(
        self,
        scene_encoder: nn.Module,
        contact_decoder: nn.Module,
        object_encoder: nn.Module = None,
        grasp_mlp: nn.Module = None,
        set_criterion: nn.Module = None,
        grasp_criterion: nn.Module = None,
        place_criterion: nn.Module = None,
    ):
        super(M2T2, self).__init__()
        self.scene_encoder = scene_encoder
        self.object_encoder = object_encoder
        self.contact_decoder = contact_decoder
        self.grasp_mlp = grasp_mlp
        self.set_criterion = set_criterion
        self.grasp_criterion = grasp_criterion
        self.place_criterion = place_criterion

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args["scene_encoder"] = PointNet2MSG.from_config(cfg.scene_encoder)
        channels = args["scene_encoder"].out_channels
        obj_channels = None
        if cfg.contact_decoder.num_place_queries > 0:
            args["object_encoder"] = PointNet2MSGCls.from_config(cfg.object_encoder)
            obj_channels = args["object_encoder"].out_channels
            args["place_criterion"] = PlaceCriterion.from_config(cfg.place_loss)
        args["contact_decoder"] = ContactDecoder.from_config(
            cfg.contact_decoder, channels, obj_channels
        )
        if cfg.contact_decoder.num_grasp_queries > 0:

            args["grasp_mlp"] = ActionDecoder.from_config(
                cfg.action_decoder, args["contact_decoder"]
            )
            matcher = HungarianMatcher.from_config(cfg.matcher)
            args["set_criterion"] = SetCriterion.from_config(cfg.grasp_loss, matcher)

            args["grasp_criterion"] = GraspCriterion.from_config(cfg.grasp_loss)
        return cls(**args)

    def forward(self, data, cfg):
        scene_feat = self.scene_encoder(data["inputs"])
        object_feat = {"features": {}}
        if self.object_encoder is not None:
            object_feat = self.object_encoder(data["object_inputs"])
        if "task_is_place" in data:
            for key, val in object_feat["features"].items():
                object_feat["features"][key] = val * data["task_is_place"].view(
                    data["task_is_place"].shape[0], 1, 1
                )
        embedding, outputs = self.contact_decoder(scene_feat, object_feat)

        losses = {}
        if self.place_criterion is not None:
            losses, stats = self.place_criterion(outputs, data)
            outputs[-1].update(stats)

        if self.set_criterion is not None:
            set_losses, outputs = self.set_criterion(outputs, data)
            losses.update(set_losses)
        else:
            outputs = outputs[-1]

        mask_features = scene_feat["features"][self.contact_decoder.mask_feature]

        if self.set_criterion is not None:
            obj_embedding = [
                emb[idx] for emb, idx in zip(embedding["grasp"], outputs["matched_idx"])
            ]
        else:
            # There is only one token
            obj_embedding = [emb[0] for emb in embedding["grasp"]]

        confidence = [mask.sigmoid() for mask in outputs["matched_grasping_masks"]]

        if self.grasp_criterion is not None:

            grasp_outputs = self.grasp_mlp(
                data["points"],
                mask_features,
                confidence,
                cfg.mask_thresh,
                obj_embedding,
                data["grasping_masks"],
            )

            outputs.update(grasp_outputs)

            contact_losses = self.grasp_criterion(outputs, data)
            losses.update(contact_losses)
        return outputs, losses

    def infer(self, data, cfg):
        scene_feat = self.scene_encoder(data["inputs"])
        object_feat = {"features": {}}
        if self.object_encoder is not None:
            object_feat = self.object_encoder(data["object_inputs"])
        embedding, outputs = self.contact_decoder(scene_feat, object_feat)
        outputs = outputs[-1]

        if cfg.task == "place":
            if cfg.cam_coord:
                cam_pose = data["cam_pose"]
            else:
                cam_pose = repeat_new_axis(
                    torch.eye(4).to(data["inputs"].device),
                    data["inputs"].shape[0],
                    dim=0,
                )
            placement_outputs = infer_placements(
                data["points"],
                outputs["placement_masks"],
                data["bottom_center"],
                data["ee_pose"],
                cam_pose,
                cfg.mask_thresh,
                cfg.placement_height,
            )
            outputs.update(placement_outputs)
            outputs["placement_masks"] = (
                outputs["placement_masks"].sigmoid() > cfg.mask_thresh
            )

        if cfg.task == "pick":
            objectness = outputs["objectness"].sigmoid()
            confidence = outputs["grasping_masks"].sigmoid()
            masks = confidence > cfg.mask_thresh
            object_ids = [
                torch.where((score > cfg.object_thresh) & (mask.sum(dim=1) > 0))[0]
                for score, mask in zip(objectness, masks)
            ]
            outputs["objectness"] = [
                score[idx] for score, idx in zip(objectness, object_ids)
            ]
            outputs["grasping_masks"] = [
                mask[idx] for mask, idx in zip(masks, object_ids)
            ]
            confidence = [conf[idx] for conf, idx in zip(confidence, object_ids)]
            mask_features = scene_feat["features"][self.contact_decoder.mask_feature]
            obj_embedding = [
                emb[idx] for emb, idx in zip(embedding["grasp"], object_ids)
            ]

            if self.grasp_mlp is not None:
                grasp_outputs = self.grasp_mlp(
                    data["points"],
                    mask_features,
                    confidence,
                    cfg.mask_thresh,
                    obj_embedding,
                )
                outputs.update(grasp_outputs)

            else:
                xyz = data["points"]
                mask_feats = mask_features.moveaxis(1, -1)  # [B, H, W, mask_dim]
                contacts, conf_all, inputs, num_grasps = [], [], [], []
                total_grasps, num_objs = 0, 0
                mask_thresh = cfg.mask_thresh
                for i, (pts, feat, emb, conf_scene) in enumerate(
                    zip(xyz, mask_feats, embedding, confidence)
                ):
                    masks = conf_scene > mask_thresh
                    conf_list, num = [], []
                    for e, m, conf in zip(emb, masks, conf_scene):
                        f, p, c = feat[m], pts[m], conf[m]
                        inputs.append(f)
                        conf_list.append(c)
                        contacts.append(p)
                        num.append(f.shape[0])
                        total_grasps += f.shape[0]
                    conf_all.append(conf_list)
                contacts = torch.cat(contacts)
                outputs["grasp_contacts"] = contacts
                outputs["grasp_confidence"] = conf_all

        return outputs
