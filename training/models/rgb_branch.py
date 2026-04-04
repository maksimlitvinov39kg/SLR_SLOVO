"""
RGB Branch: MViTv2-S pretrained on Kinetics-400.
MViTv2 uses relative positional encoding, so it handles
different temporal sizes (16, 32, etc.) natively.
"""

import torch
import torch.nn as nn


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
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "mvit_v2_s":
            self._build_mvit(pretrained, freeze_stages, dropout, drop_path_rate)
        elif backbone == "video_swin_t":
            self._build_swin(pretrained, freeze_stages, dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _build_mvit(self, pretrained, freeze_stages, dropout, drop_path_rate):
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        model = mvit_v2_s(weights=weights)

        self.feature_dim = 768

        # Replace classification head with identity
        model.head = nn.Identity()

        # Set drop path rate if specified
        if drop_path_rate > 0:
            self._set_drop_path(model, drop_path_rate)

        # MViTv2 uses relative positional encoding (not absolute),
        # so it handles different temporal sizes natively — no interpolation needed

        self.model = model

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

    def _set_drop_path(self, model, rate):
        """Set stochastic depth rate across all blocks."""
        blocks = [b for b in model.blocks if hasattr(b, 'stochastic_depth')]
        n = len(blocks)
        for i, block in enumerate(blocks):
            block.stochastic_depth.p = rate * i / max(n - 1, 1)

    def _freeze_stages(self, n_stages):
        """Freeze first n stages of the backbone."""
        ct = 0
        for name, param in self.model.named_parameters():
            if ct < n_stages:
                param.requires_grad = False
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
        features = self.model(x)
        return self.dropout(features)

    def get_feature_dim(self):
        return self.feature_dim
