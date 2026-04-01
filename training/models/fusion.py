"""
Dual-stream fusion model: RGB + Keypoints.

Supports three fusion strategies:
1. "concat" - simple concatenation + MLP
2. "gated" - gated fusion with learned weighting
3. "cross_attention" - bidirectional cross-attention fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rgb_branch import RGBBranch
from .keypoint_branch import KeypointBranch, RMSNorm


class GatedFusion(nn.Module):
    """Gated fusion: learns how much to weight each modality."""

    def __init__(self, rgb_dim, kp_dim, fused_dim):
        super().__init__()
        total_dim = rgb_dim + kp_dim

        # Project both to same dim
        self.rgb_proj = nn.Linear(rgb_dim, fused_dim)
        self.kp_proj = nn.Linear(kp_dim, fused_dim)

        # Gate: sigmoid over concatenated features
        self.gate = nn.Sequential(
            nn.Linear(total_dim, fused_dim),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(fused_dim)

    def forward(self, rgb_feat, kp_feat):
        rgb_proj = self.rgb_proj(rgb_feat)
        kp_proj = self.kp_proj(kp_feat)

        gate_input = torch.cat([rgb_feat, kp_feat], dim=-1)
        gate = self.gate(gate_input)  # (B, fused_dim), values in [0, 1]

        fused = gate * rgb_proj + (1 - gate) * kp_proj
        return self.norm(fused)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between RGB and keypoint features."""

    def __init__(self, rgb_dim, kp_dim, fused_dim, num_heads=8):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, fused_dim)
        self.kp_proj = nn.Linear(kp_dim, fused_dim)

        # RGB attends to keypoints
        self.rgb_cross = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=0.1, batch_first=True
        )
        # Keypoints attend to RGB
        self.kp_cross = nn.MultiheadAttention(
            fused_dim, num_heads, dropout=0.1, batch_first=True
        )

        self.norm = RMSNorm(fused_dim * 2)
        self.out_proj = nn.Linear(fused_dim * 2, fused_dim)

    def forward(self, rgb_feat, kp_feat):
        # Project to same dim, add sequence dim for attention
        rgb = self.rgb_proj(rgb_feat).unsqueeze(1)  # (B, 1, D)
        kp = self.kp_proj(kp_feat).unsqueeze(1)     # (B, 1, D)

        rgb_enhanced, _ = self.rgb_cross(rgb, kp, kp)  # (B, 1, D)
        kp_enhanced, _ = self.kp_cross(kp, rgb, rgb)    # (B, 1, D)

        fused = torch.cat([rgb_enhanced.squeeze(1), kp_enhanced.squeeze(1)], dim=-1)
        fused = self.norm(fused)
        return self.out_proj(fused)


class DualStreamModel(nn.Module):
    """
    Full dual-stream model: RGB branch + Keypoint branch + Fusion + Classifier.

    Can also run in single-stream mode (rgb_only or kp_only).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        # RGB branch config
        rgb_backbone: str = "mvit_v2_s",
        rgb_pretrained: bool = True,
        rgb_freeze_stages: int = 0,
        # Keypoint branch config
        kp_d_model: int = 256,
        kp_num_heads: int = 8,
        kp_num_layers: int = 6,
        kp_dropout: float = 0.2,
        kp_drop_path: float = 0.15,
        # Fusion config
        fusion_type: str = "gated",  # "concat", "gated", "cross_attention"
        fused_dim: int = 512,
        # Mode
        mode: str = "both",  # "rgb", "keypoints", "both"
        # Classifier
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        # Build branches
        if mode in ("rgb", "both"):
            self.rgb_branch = RGBBranch(
                backbone=rgb_backbone,
                pretrained=rgb_pretrained,
                freeze_stages=rgb_freeze_stages,
                dropout=0.1,
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

        # Build fusion
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
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
            classifier_input = fused_dim
        elif mode == "rgb":
            classifier_input = rgb_dim
        else:
            classifier_input = kp_dim

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, classifier_input // 2),
            RMSNorm(classifier_input // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_input // 2, num_classes),
        )

    def forward(self, batch):
        """
        Args:
            batch: dict with keys depending on mode:
                - "rgb": (B, 3, T, H, W)
                - "keypoints": (B, T, 87, 3)
                - "non_empty_frame_idxs": (B, T)
        Returns:
            logits: (B, num_classes)
        """
        if self.mode == "rgb":
            feat = self.rgb_branch(batch["rgb"])
            return self.classifier(feat)

        elif self.mode == "keypoints":
            feat = self.kp_branch(batch["keypoints"], batch["non_empty_frame_idxs"])
            return self.classifier(feat)

        else:  # both
            rgb_feat = self.rgb_branch(batch["rgb"])
            kp_feat = self.kp_branch(batch["keypoints"], batch["non_empty_frame_idxs"])

            if isinstance(self.fusion, (GatedFusion, CrossAttentionFusion)):
                fused = self.fusion(rgb_feat, kp_feat)
            else:
                # concat fusion
                fused = self.fusion(torch.cat([rgb_feat, kp_feat], dim=-1))

            return self.classifier(fused)

    def get_rgb_params(self):
        """Get RGB branch parameters (for differential LR)."""
        if hasattr(self, "rgb_branch"):
            return self.rgb_branch.parameters()
        return []

    def get_kp_params(self):
        """Get keypoint branch parameters."""
        if hasattr(self, "kp_branch"):
            return self.kp_branch.parameters()
        return []

    def get_fusion_params(self):
        """Get fusion + classifier parameters."""
        params = list(self.classifier.parameters())
        if hasattr(self, "fusion"):
            params += list(self.fusion.parameters())
        return params
