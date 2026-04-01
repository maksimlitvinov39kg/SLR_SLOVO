"""
RGB Branch: MViTv2-S pretrained on Kinetics-400.
Uses pytorchvideo / torchvision implementation.
"""

import torch
import torch.nn as nn
from functools import partial


class RGBBranch(nn.Module):
    """
    MViTv2 backbone for RGB video.

    Input: (B, C=3, T, H, W) - video tensor
    Output: (B, feature_dim) - pooled features
    """

    def __init__(
        self,
        backbone: str = "mvit_v2_s",
        pretrained: bool = True,
        freeze_stages: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "mvit_v2_s":
            self._build_mvit(pretrained, freeze_stages, dropout)
        elif backbone == "video_swin_t":
            self._build_swin(pretrained, freeze_stages, dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _build_mvit(self, pretrained, freeze_stages, dropout):
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        model = mvit_v2_s(weights=weights)

        # Feature dim from MViTv2-S
        self.feature_dim = 768

        # Remove the classification head
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        # The actual MViT architecture in torchvision is a bit different,
        # so let's use the full model and replace the head
        self.model = model
        self.model.head = nn.Identity()

        # Freeze early stages if requested
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

        self.dropout = nn.Dropout(dropout)

    def _build_swin(self, pretrained, freeze_stages, dropout):
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights

        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        model = swin3d_t(weights=weights)

        self.feature_dim = 768
        model.head = nn.Identity()
        self.model = model

        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

        self.dropout = nn.Dropout(dropout)

    def _freeze_stages(self, n_stages):
        """Freeze first n stages of the backbone."""
        ct = 0
        for name, param in self.model.named_parameters():
            if ct < n_stages:
                param.requires_grad = False
            # Count "blocks" or "layers" transitions as stages
            if "block" in name or "layers" in name:
                if ".0." in name and ct < n_stages:
                    ct += 1

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) video tensor
        Returns:
            features: (B, feature_dim)
        """
        features = self.model(x)  # (B, feature_dim)
        return self.dropout(features)

    def get_feature_dim(self):
        return self.feature_dim
