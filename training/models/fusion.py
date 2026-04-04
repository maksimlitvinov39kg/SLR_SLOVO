"""
Dual-stream fusion model: RGB + Keypoints.
Includes sign boundary regression head (Krivov et al. 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rgb_branch import RGBBranch
from .keypoint_branch import KeypointBranch, RMSNorm


class GatedFusion(nn.Module):
    def __init__(self, rgb_dim, kp_dim, fused_dim):
        super().__init__()
        total_dim = rgb_dim + kp_dim
        self.rgb_proj = nn.Linear(rgb_dim, fused_dim)
        self.kp_proj = nn.Linear(kp_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.norm = RMSNorm(fused_dim)

    def forward(self, rgb_feat, kp_feat):
        rgb_proj = self.rgb_proj(rgb_feat)
        kp_proj = self.kp_proj(kp_feat)
        gate_input = torch.cat([rgb_feat, kp_feat], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * rgb_proj + (1 - gate) * kp_proj
        return self.norm(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(self, rgb_dim, kp_dim, fused_dim, num_heads=8):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, fused_dim)
        self.kp_proj = nn.Linear(kp_dim, fused_dim)
        self.rgb_cross = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.kp_cross = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm = RMSNorm(fused_dim * 2)
        self.out_proj = nn.Linear(fused_dim * 2, fused_dim)

    def forward(self, rgb_feat, kp_feat):
        rgb = self.rgb_proj(rgb_feat).unsqueeze(1)
        kp = self.kp_proj(kp_feat).unsqueeze(1)
        rgb_enhanced, _ = self.rgb_cross(rgb, kp, kp)
        kp_enhanced, _ = self.kp_cross(kp, rgb, rgb)
        fused = torch.cat([rgb_enhanced.squeeze(1), kp_enhanced.squeeze(1)], dim=-1)
        fused = self.norm(fused)
        return self.out_proj(fused)


class DualStreamModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        rgb_backbone: str = "mvit_v2_s",
        rgb_pretrained: bool = True,
        rgb_freeze_stages: int = 0,
        num_frames_rgb: int = 32,
        drop_path_rate: float = 0.2,
        kp_d_model: int = 256,
        kp_num_heads: int = 8,
        kp_num_layers: int = 6,
        kp_dropout: float = 0.2,
        kp_drop_path: float = 0.15,
        fusion_type: str = "gated",
        fused_dim: int = 512,
        mode: str = "both",
        classifier_dropout: float = 0.0,
        label_smooth_eps: float = 0.1,
        use_regression_head: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.use_regression_head = use_regression_head

        # Build branches
        if mode in ("rgb", "both"):
            self.rgb_branch = RGBBranch(
                backbone=rgb_backbone,
                pretrained=rgb_pretrained,
                freeze_stages=rgb_freeze_stages,
                dropout=0.0,
                drop_path_rate=drop_path_rate,
            )
            rgb_dim = self.rgb_branch.get_feature_dim()
        else:
            rgb_dim = 0

        if mode in ("keypoints", "both"):
            self.kp_branch = KeypointBranch(
                d_model=kp_d_model,
                num_heads=kp_num_heads,
                num_layers=kp_num_layers,
                dropout=kp_dropout,
                drop_path=kp_drop_path,
            )
            kp_dim = self.kp_branch.get_feature_dim()
        else:
            kp_dim = 0

        # Fusion
        if mode == "both":
            if fusion_type == "concat":
                self.fusion = nn.Sequential(
                    nn.Linear(rgb_dim + kp_dim, fused_dim),
                    RMSNorm(fused_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            elif fusion_type == "gated":
                self.fusion = GatedFusion(rgb_dim, kp_dim, fused_dim)
            elif fusion_type == "cross_attention":
                self.fusion = CrossAttentionFusion(rgb_dim, kp_dim, fused_dim)
            classifier_input = fused_dim
        elif mode == "rgb":
            classifier_input = rgb_dim
        else:
            classifier_input = kp_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_input, num_classes),
        )

        # Regression head: predicts normalized (begin, end) of sign
        if use_regression_head:
            self.regression_head = nn.Linear(classifier_input, 2)
            # Small init for regression head (like Krivov init_scale=0.001)
            nn.init.normal_(self.regression_head.weight, std=0.001)
            nn.init.zeros_(self.regression_head.bias)

    def forward(self, batch):
        """
        Returns:
            dict with 'logits', optionally 'bounds_pred'
        """
        # Extract features
        if self.mode == "rgb":
            feat = self.rgb_branch(batch["rgb"])
        elif self.mode == "keypoints":
            feat = self.kp_branch(batch["keypoints"], batch["non_empty_frame_idxs"])
        else:
            rgb_feat = self.rgb_branch(batch["rgb"])
            kp_feat = self.kp_branch(batch["keypoints"], batch["non_empty_frame_idxs"])
            if isinstance(self.fusion, (GatedFusion, CrossAttentionFusion)):
                feat = self.fusion(rgb_feat, kp_feat)
            else:
                feat = self.fusion(torch.cat([rgb_feat, kp_feat], dim=-1))

        logits = self.classifier(feat)

        output = {"logits": logits}

        if self.use_regression_head:
            bounds_pred = torch.sigmoid(self.regression_head(feat))  # (B, 2) in [0, 1]
            output["bounds_pred"] = bounds_pred

        return output

    def get_rgb_params(self):
        if hasattr(self, "rgb_branch"):
            return self.rgb_branch.parameters()
        return []

    def get_kp_params(self):
        if hasattr(self, "kp_branch"):
            return self.kp_branch.parameters()
        return []

    def get_fusion_params(self):
        params = list(self.classifier.parameters())
        if hasattr(self, "fusion"):
            params += list(self.fusion.parameters())
        if hasattr(self, "regression_head"):
            params += list(self.regression_head.parameters())
        return params
