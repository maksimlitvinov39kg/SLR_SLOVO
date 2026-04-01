"""
Keypoint Branch: improved version of the original hierarchical transformer.
Key improvements over the bachelor's version:
  - Stochastic depth for regularization
  - Better embedding with separate spatial attention per body part
  - Dilated temporal convolutions
  - Proper pre-norm transformer blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(x, cos, sin):
    """Apply RoPE to queries/keys. x: (B, H, T, D)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:2*d]
    cos = cos[:x.shape[2], :d].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :d].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if x.shape[-1] > 2 * d:
        rotated = torch.cat([rotated, x[..., 2*d:]], dim=-1)
    return rotated


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(self.d_k)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = dim * expansion
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.wo = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.wo(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)

        # Stochastic depth
        self.drop_path_rate = drop_path
        if drop_path > 0:
            self.drop_path = StochasticDepth(drop_path)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.dropout(self.attn(self.norm1(x), mask)))
        x = x + self.drop_path(self.dropout(self.mlp(self.norm2(x))))
        return x


class StochasticDepth(nn.Module):
    """Drop entire residual branch with probability p during training."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.p
        return x * keep / (1 - self.p)


class TemporalConvBlock(nn.Module):
    """Multi-scale dilated temporal convolutions."""
    def __init__(self, d_model):
        super().__init__()
        # Different kernel sizes AND dilations
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model // 4, kernel_size=3, padding=1, dilation=1),
            nn.Conv1d(d_model, d_model // 4, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(d_model, d_model // 4, kernel_size=3, padding=4, dilation=4),
            nn.Conv1d(d_model, d_model // 4, kernel_size=3, padding=8, dilation=8),
        ])
        self.norm = RMSNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x_t = x.transpose(1, 2)  # (B, D, T)
        outs = [conv(x_t) for conv in self.convs]
        x_t = torch.cat(outs, dim=1)  # (B, D, T)
        x = x_t.transpose(1, 2)  # (B, T, D)
        return self.norm(self.act(x) + residual)


class BodyPartEmbedding(nn.Module):
    """Embed a single body part's keypoints -> temporal sequence."""
    def __init__(self, n_landmarks, d_model, n_dims=2):
        super().__init__()
        self.proj = nn.Linear(n_landmarks * n_dims, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # x: (B, T, N_landmarks, 2) or (B, T, N_landmarks, 3)
        B, T, N, D = x.shape
        x_flat = x.reshape(B, T, N * D)
        return self.norm(self.proj(x_flat))


class CrossModalAttention(nn.Module):
    """Cross-attention between two modality streams."""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        out, _ = self.attn(query, context, context)
        return self.norm(query + self.dropout(out))


class KeypointBranch(nn.Module):
    """
    Improved keypoint transformer.

    Input: keypoints (B, T, 78, 2), non_empty_frame_idxs (B, T)
    Output: features (B, feature_dim)

    RTMW COCO-WholeBody SLR subset (78 keypoints):
        [0:30]   Face — eyebrows (10) + lips (20)
        [30:51]  Left hand (21)
        [51:72]  Right hand (21)
        [72:78]  Upper body pose (6) — shoulders, elbows, wrists
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.2,
        drop_path: float = 0.15,
        n_dims: int = 2,  # RTMW=2 (x,y), MediaPipe legacy=3 (x,y,z)
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = d_model
        self.n_dims = n_dims

        # Body part embeddings: face(30), left hand(21), right hand(21), pose(6)
        self.face_embed = BodyPartEmbedding(30, d_model, n_dims)
        self.left_hand_embed = BodyPartEmbedding(21, d_model, n_dims)
        self.right_hand_embed = BodyPartEmbedding(21, d_model, n_dims)
        self.pose_embed = BodyPartEmbedding(6, d_model, n_dims)

        # Cross-modal attention between hands
        self.hand_cross_attn = CrossModalAttention(d_model, num_heads=4, dropout=dropout)
        # Cross-modal attention between face and pose
        self.face_pose_cross_attn = CrossModalAttention(d_model, num_heads=4, dropout=dropout)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal position embedding
        self.temporal_embed = nn.Embedding(512, d_model)

        # Transformer blocks with linearly increasing drop path
        drop_rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]

        # Local blocks (first half)
        n_local = num_layers // 2
        self.local_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout, drop_rates[i])
            for i in range(n_local)
        ])

        # Temporal conv
        self.temporal_conv = TemporalConvBlock(d_model)

        # Global blocks (second half)
        self.global_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout, drop_rates[n_local + i])
            for i in range(num_layers - n_local)
        ])

        self.final_norm = RMSNorm(d_model)

        # Pooling: attention-based
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, keypoints, non_empty_frame_idxs):
        """
        Args:
            keypoints: (B, T, 78, 2)
            non_empty_frame_idxs: (B, T)
        Returns:
            features: (B, d_model)
        """
        B, T = keypoints.shape[:2]
        attention_mask = (non_empty_frame_idxs != -1).float()  # (B, T)

        # Split body parts (RTMW SLR subset layout)
        face = keypoints[:, :, 0:30, :]       # eyebrows + lips
        left_hand = keypoints[:, :, 30:51, :]
        right_hand = keypoints[:, :, 51:72, :]
        pose = keypoints[:, :, 72:78, :]

        # Embed each part
        face_emb = self.face_embed(face)
        lh_emb = self.left_hand_embed(left_hand)
        rh_emb = self.right_hand_embed(right_hand)
        pose_emb = self.pose_embed(pose)

        # Cross-modal attention
        lh_enhanced = self.hand_cross_attn(lh_emb, rh_emb)
        rh_enhanced = self.hand_cross_attn(rh_emb, lh_emb)
        face_enhanced = self.face_pose_cross_attn(face_emb, pose_emb)

        # Fuse all streams
        concat = torch.cat([face_enhanced, lh_enhanced, rh_enhanced, pose_emb], dim=-1)
        x = self.fusion(concat)  # (B, T, d_model)

        # Add temporal position
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.temporal_embed(pos)

        # Local transformer blocks
        for block in self.local_blocks:
            x = block(x, attention_mask)

        # Temporal convolution
        x = self.temporal_conv(x)

        # Global transformer blocks
        for block in self.global_blocks:
            x = block(x, attention_mask)

        x = self.final_norm(x)

        # Attention pooling
        query = self.pool_query.expand(B, -1, -1)
        key_padding_mask = attention_mask == 0
        pooled, _ = self.pool_attn(query, x, x, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)  # (B, d_model)

        return pooled

    def get_feature_dim(self):
        return self.feature_dim
