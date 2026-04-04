"""
RGB Branch: MViTv2-S pretrained on Kinetics-400.

Supports 32 frames by interpolating temporal position embeddings
from the pretrained 16-frame model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interpolate_mvit_pos_embed(model, new_temporal_size):
    """
    Interpolate MViTv2 positional embeddings to support a different
    number of input frames. Pretrained model expects 16 frames
    (8 temporal tokens after patch embed with stride 2).

    For 32 frames -> 16 temporal tokens. Interpolate from 8 -> 16.
    """
    # MViTv2 in torchvision stores pos_embedding as a nn.Parameter
    # Shape: (1, T*H*W, C) where T=8, H=56, W=56 for 16 input frames
    old_pos = model.pos_encoding.pos_embedding  # (1, 8*56*56, C)

    old_T = 8   # 16 frames / temporal_stride=2
    new_T = new_temporal_size // 2  # 32 frames / temporal_stride=2 = 16
    H, W = 56, 56
    C = old_pos.shape[-1]

    if old_T == new_T:
        return  # no change needed

    # Reshape to (1, T, H, W, C)
    pos_4d = old_pos.data.reshape(1, old_T, H, W, C)

    # Permute to (1, C, T, H, W) for interpolation
    pos_5d = pos_4d.permute(0, 4, 1, 2, 3)

    # Interpolate only temporal dimension
    pos_5d_interp = F.interpolate(
        pos_5d, size=(new_T, H, W), mode='trilinear', align_corners=False
    )

    # Back to (1, T*H*W, C)
    new_pos = pos_5d_interp.permute(0, 2, 3, 4, 1).reshape(1, new_T * H * W, C)

    model.pos_encoding.pos_embedding = nn.Parameter(new_pos)
    print(f"Interpolated pos_embed: temporal {old_T} -> {new_T} "
          f"(total tokens {old_pos.shape[1]} -> {new_pos.shape[1]})")


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
        num_frames: int = 16,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_frames = num_frames

        if backbone == "mvit_v2_s":
            self._build_mvit(pretrained, freeze_stages, dropout, num_frames, drop_path_rate)
        elif backbone == "video_swin_t":
            self._build_swin(pretrained, freeze_stages, dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _build_mvit(self, pretrained, freeze_stages, dropout, num_frames, drop_path_rate):
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        model = mvit_v2_s(weights=weights)

        self.feature_dim = 768

        # Replace classification head with identity
        model.head = nn.Identity()

        # Set drop path rate if specified
        if drop_path_rate > 0:
            self._set_drop_path(model, drop_path_rate)

        # Interpolate position embeddings for 32 frames
        if num_frames > 16 and pretrained:
            _interpolate_mvit_pos_embed(model, num_frames)

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
