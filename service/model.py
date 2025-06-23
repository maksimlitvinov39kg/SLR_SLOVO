import torch
from torch import nn
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
device = 'cpu'
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x):
        seq_len = x.shape[1] 
        batch_size = x.shape[0] 
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        
        if emb.shape[-1] != self.dim:
            if emb.shape[-1] > self.dim:
                emb = emb[..., :self.dim]
            else:
                pad_size = self.dim - emb.shape[-1]
                emb = F.pad(emb, (0, pad_size))
        
        cos_emb = emb.cos()[None, :, None, :]  # [1, T, 1, D]
        sin_emb = emb.sin()[None, :, None, :]  # [1, T, 1, D]
        return cos_emb, sin_emb

def apply_rotary_pos_emb(x, cos, sin):
    x_rot = x.clone()
    half_dim = x.shape[-1] // 2
    
    x1, x2 = x_rot[..., :half_dim], x_rot[..., half_dim:2*half_dim]
    cos = cos[..., :half_dim]
    sin = sin[..., :half_dim]
    
    rotated = torch.cat(
        [x1 * cos - x2 * sin, 
         x1 * sin + x2 * cos],
        dim=-1
    )
    
    if x.shape[-1] > 2 * half_dim:
        rotated = torch.cat([rotated, x_rot[..., 2*half_dim:]], dim=-1)
        
    return rotated


class AdvancedScaledDotProductAttention(nn.Module):
    
    def __init__(self, d_model, temperature=1.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
       
        self.rope = RotaryPositionalEmbedding(d_model)
        
    def forward(self, q, k, v, mask=None):
        B, H, T, D = q.shape
        
        q_reshaped = q.reshape(B*H, T, D)
        k_reshaped = k.reshape(B*H, T, D)
        
        cos, sin = self.rope(q_reshaped)
        
        cos = cos.expand(B*H, -1, -1, -1).reshape(B, H, T, D)
        sin = sin.expand(B*H, -1, -1, -1).reshape(B, H, T, D)
        
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(D) * self.temperature)
        
        if mask is not None:
                if mask.dim() == 2:  # [B, T]
                    mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
                elif mask.dim() == 3:  # [B, T, T]
                    mask = mask.unsqueeze(1)  # [B, 1, T, T]
                
                try:
                    scores = scores.masked_fill(mask == 0, -1e4)
                except RuntimeError:
                    causal_mask = torch.tril(torch.ones(T, T, device=device))
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                    scores = scores.masked_fill(causal_mask == 0, -1e4)
        
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores = scores - scores_max
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        return out, attn_weights


class MultiHeadAttentionWithRoPE(nn.Module):
    """Fixed Multi-Head Attention with RoPE"""
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = AdvancedScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, D = x.shape
        
        Q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        

        attn_out, _ = self.attention(Q, K, V, mask)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        
        out = self.w_o(attn_out)
        return self.dropout(out)
        

class SwiGLU(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2, bias=False)
        self.w2 = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight

class AdvancedTransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1, use_swiglu=True):
        super().__init__()
        self.attention = MultiHeadAttentionWithRoPE(d_model, num_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        if use_swiglu:
            self.mlp = SwiGLU(d_model)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable skip connection weights
        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.alpha_mlp = nn.Parameter(torch.ones(1))
        
    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.alpha_attn * self.dropout(attn_out)
        
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.alpha_mlp * self.dropout(mlp_out)
        
        return x

class TemporalConvolution(nn.Module):
    def __init__(self, d_model, kernel_sizes=[3, 5, 7]):
        super().__init__()
        
        self.d_model = d_model
        self.num_kernels = len(kernel_sizes)
        
        dims_per_conv = []
        remaining_dim = d_model
        
        for i in range(self.num_kernels - 1):
            dim = remaining_dim // (self.num_kernels - i)
            dims_per_conv.append(dim)
            remaining_dim -= dim
        dims_per_conv.append(remaining_dim) 
        
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, dims_per_conv[i], 
                     kernel_size=k, padding=k//2)
            for i, k in enumerate(kernel_sizes)
        ])
        
        total_out_dim = sum(dims_per_conv)
        assert total_out_dim == d_model, f"Dimension mismatch: {total_out_dim} != {d_model}"
        
        self.norm = RMSNorm(d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: [B, T, D]
        residual = x
        x = x.transpose(1, 2)  # [B, D, T]
        
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        
        x = torch.cat(conv_outs, dim=1)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]
        
        assert x.shape == residual.shape, f"Shape mismatch: {x.shape} vs {residual.shape}"
        
        return self.norm(x + residual)

class PatchEmbedding(nn.Module):
    def __init__(self, num_landmarks, d_model, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_landmarks = num_landmarks
        self.d_model = d_model
        
        self.proj = nn.Linear(patch_size * 2, d_model)
        self.norm = RMSNorm(d_model)
        
        max_patches = (num_landmarks + patch_size - 1) // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        
    def forward(self, x):
       
        B, T, N, _ = x.shape
        
        patches = []
        for i in range(0, N, self.patch_size):
            end_idx = min(i + self.patch_size, N)
            patch = x[:, :, i:end_idx, :].flatten(-2) 
            
            if patch.shape[-1] < self.patch_size * 2:
                pad_size = self.patch_size * 2 - patch.shape[-1]
                patch = F.pad(patch, (0, pad_size))
            
            patches.append(patch)
        
        patches = torch.stack(patches, dim=2)  # [B, T, num_patches, patch_size*2]
        
        patches = self.proj(patches)  # [B, T, num_patches, d_model]
        
        num_patches = patches.shape[2]
        patches = patches + self.pos_embed[:, :num_patches, :]
        
        return self.norm(patches)

class AdvancedLandmarkEmbedding(nn.Module):
    def __init__(self, num_landmarks, d_model, name, patch_size=4):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.d_model = d_model
        self.name = name
        
        self.patch_embedding = PatchEmbedding(num_landmarks, d_model, patch_size)
        
        self.self_attention = MultiHeadAttentionWithRoPE(d_model, num_heads=4)
        
        self.missing_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [B, T, N, 2]
        B, T, N, _ = x.shape
        
        frame_missing = (x.sum(dim=(2, 3)) == 0)  # [B, T]
        
        patches = self.patch_embedding(x)  # [B, T, num_patches, d_model]
        
        B, T, P, D = patches.shape
        patches_flat = patches.view(B * T, P, D)
        attended_patches = self.self_attention(patches_flat)
        attended_patches = attended_patches.view(B, T, P, D)
        
        temporal_repr = attended_patches.mean(dim=2)  # [B, T, d_model]
        
        missing_expanded = self.missing_embed.expand(B, T, D)
        temporal_repr = torch.where(
            frame_missing.unsqueeze(-1).expand_as(temporal_repr),
            missing_expanded,
            temporal_repr
        )
        
        return temporal_repr

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(0.1)
        self.norm = RMSNorm(d_model)
        
    def forward(self, query, key_value):
        B, T, D = query.shape
        
        Q = self.w_q(query).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key_value).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(key_value).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        
        attended = self.w_o(attn_out)
        attended = self.dropout(attended)
        
        return self.norm(query + attended)

class AdvancedEmbeddingBothHands(nn.Module):
    
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model
        
        self.lips_embedding = AdvancedLandmarkEmbedding(40, d_model, 'lips')
        self.left_hand_embedding = AdvancedLandmarkEmbedding(21, d_model, 'left_hand')
        self.right_hand_embedding = AdvancedLandmarkEmbedding(21, d_model, 'right_hand')
        self.pose_embedding = AdvancedLandmarkEmbedding(5, d_model, 'pose')
        
        self.cross_attention_hands = CrossModalAttention(d_model)
        self.cross_attention_face_pose = CrossModalAttention(d_model)
        
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4)
        self.fusion_proj = nn.Linear(d_model * 4, d_model)
        
        self.temporal_embed = nn.Embedding(1000, d_model)
        
    def forward(self, lips, left_hand, right_hand, pose, non_empty_frame_idxs):
        B, T = lips.shape[:2]
        
        lips_emb = self.lips_embedding(lips)        # [B, T, d_model]
        left_hand_emb = self.left_hand_embedding(left_hand)
        right_hand_emb = self.right_hand_embedding(right_hand)
        pose_emb = self.pose_embedding(pose)
        
        left_hand_enhanced = self.cross_attention_hands(left_hand_emb, right_hand_emb)
        right_hand_enhanced = self.cross_attention_hands(right_hand_emb, left_hand_emb)
        
        lips_enhanced = self.cross_attention_face_pose(lips_emb, pose_emb)
        pose_enhanced = self.cross_attention_face_pose(pose_emb, lips_emb)
        
        all_features = torch.stack([
            lips_enhanced, left_hand_enhanced, right_hand_enhanced, pose_enhanced
        ], dim=-1)  # [B, T, d_model, 4]
        
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = (all_features * weights).sum(dim=-1)  # [B, T, d_model]
        
        concat_features = torch.cat([
            lips_enhanced, left_hand_enhanced, right_hand_enhanced, pose_enhanced
        ], dim=-1)  # [B, T, d_model*4]
        
        projected = self.fusion_proj(concat_features)  # [B, T, d_model]
    
        x = fused + projected
        
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.temporal_embed(positions)
        
        return x

class HierarchicalTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.local_blocks = nn.ModuleList([
            AdvancedTransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers // 2)
        ])
        
        self.temporal_conv = TemporalConvolution(d_model)
        
        self.global_blocks = nn.ModuleList([
            AdvancedTransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers - num_layers // 2)
        ])
        
        self.norm = RMSNorm(d_model)
        
    def forward(self, x, attention_mask=None):
        for block in self.local_blocks:
            x = block(x, attention_mask)
        
        x = self.temporal_conv(x)
        
        for block in self.global_blocks:
            x = block(x, attention_mask)
        
        return self.norm(x)

class MultiScalePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool1d(scale) for scale in [1, 2, 4, 8]
        ])
        
        self.attention_pooling = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=0.1, batch_first=True
        )
        
        total_features = d_model * 7
        
        self.combiner = nn.Sequential(
            nn.Linear(total_features, d_model * 2),
            RMSNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x, attention_mask):
        B, T, D = x.shape
        
        device = x.device
        pooling_query = torch.randn(1, 1, D, device=device) * 0.02
        
        expanded_mask = attention_mask.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        
        masked_x = x * expanded_mask
        mean_pooled = masked_x.sum(dim=1) / torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        
        masked_x_for_max = x.masked_fill(expanded_mask == 0, -float('inf'))
        max_pooled, _ = masked_x_for_max.max(dim=1)
        max_pooled = torch.where(torch.isinf(max_pooled), torch.zeros_like(max_pooled), max_pooled)
        
        x_transposed = masked_x.transpose(1, 2)  # [B, D, T]
        multiscale_features = []
        
        for pool_layer in self.pooling_layers:
            pooled = pool_layer(x_transposed)  # [B, D, scale]
            pooled = pooled.mean(dim=-1)  # [B, D]
            multiscale_features.append(pooled)
        
        query = pooling_query.expand(B, -1, -1)
        key_padding_mask = attention_mask.squeeze(-1) == 0
        attended, _ = self.attention_pooling(query, x, x, key_padding_mask=key_padding_mask)
        attended = attended.squeeze(1)
        
        all_features = torch.cat([
            mean_pooled,        # [B, D]
            max_pooled,         # [B, D]
            attended,           # [B, D]
            *multiscale_features  # 4 x [B, D]
        ], dim=-1)  # [B, 7*D]
        
        return self.combiner(all_features)


class SOTAClassifier(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        self.hidden_dim1 = d_model // 2  # 256 if d_model=512
        self.hidden_dim2 = d_model // 4  # 128 if d_model=512
        
        self.layer1 = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim1),
            RMSNorm(self.hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            RMSNorm(self.hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.final_classifier = nn.Linear(self.hidden_dim2, num_classes)
        

        self.aux_classifiers = nn.ModuleList([
            nn.Linear(d_model, num_classes),          
            nn.Linear(self.hidden_dim1, num_classes), 
            nn.Linear(self.hidden_dim2, num_classes)  
        ])
        
        self.confidence_head = nn.Linear(self.hidden_dim2, 1)
        
    def forward(self, x, return_aux=False, return_confidence=False):
        intermediate_features = []
        
        intermediate_features.append(x)  # [B, d_model]
        
        x1 = self.layer1(x)
        intermediate_features.append(x1)  # [B, hidden_dim1]
        
        x2 = self.layer2(x1)
        intermediate_features.append(x2)  # [B, hidden_dim2]
        
        main_output = self.final_classifier(x2)
        
        results = [main_output]
        
        if return_aux:
            aux_outputs = []
            for i, aux_classifier in enumerate(self.aux_classifiers):
                aux_output = aux_classifier(intermediate_features[i])
                aux_outputs.append(aux_output)
            results.append(aux_outputs)
        
        if return_confidence:
            confidence = torch.sigmoid(self.confidence_head(x2))
            results.append(confidence)
        
        return results[0] if len(results) == 1 else tuple(results)

class SOTASignLanguageModel(nn.Module):

    def __init__(self, lips_mean, lips_std, left_hands_mean, left_hands_std,
                 right_hands_mean, right_hands_std, pose_mean, pose_std,
                 d_model=512, num_heads=8, num_layers=8, num_classes=250):
        super().__init__()
        
        self.register_buffer("LIPS_MEAN", torch.tensor(lips_mean))
        self.register_buffer("LIPS_STD", torch.tensor(lips_std))
        self.register_buffer("LEFT_HANDS_MEAN", torch.tensor(left_hands_mean))
        self.register_buffer("LEFT_HANDS_STD", torch.tensor(left_hands_std))
        self.register_buffer("RIGHT_HANDS_MEAN", torch.tensor(right_hands_mean))
        self.register_buffer("RIGHT_HANDS_STD", torch.tensor(right_hands_std))
        self.register_buffer("POSE_MEAN", torch.tensor(pose_mean))
        self.register_buffer("POSE_STD", torch.tensor(pose_std))
        
        self.d_model = d_model
        
        self.embedding = AdvancedEmbeddingBothHands(d_model)
        
        self.transformer = HierarchicalTransformer(d_model, num_heads, num_layers)
        
        self.pooling = MultiScalePooling(d_model)
        
        self.classifier = SOTAClassifier(d_model, num_classes)
        
        self.label_smoothing = 0.1
        self.focal_alpha = 1.0
        self.focal_gamma = 2.0
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(self, frames, non_empty_frame_idxs, training=True, return_aux=False):
        attention_mask = (non_empty_frame_idxs != -1).float()
        
        if training:

            random_keep_prob = 0.8
            random_mask = torch.rand_like(attention_mask) < random_keep_prob
            attention_mask = attention_mask * random_mask.float()
            batch_size,seq_len = attention_mask.size()[0],attention_mask.size()[1]
            min_frames = max(1, int(seq_len * 0.2))
            for b in range(batch_size):
                if random_mask[b].sum() < min_frames:
                    valid_indices = (attention_mask[b, :, 0] == 1).nonzero(as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        keep_indices = valid_indices[torch.randperm(len(valid_indices))[:min_frames]]
                        random_mask[b, keep_indices, 0] = True
            
            attention_mask = attention_mask * random_mask.float()
        
        x = frames[:, :, :, :2]
        
        LIPS_START = 0
        LEFT_HAND_START = 40
        RIGHT_HAND_START = 61
        POSE_START = 82
        
        lips = x[:, :, LIPS_START:LIPS_START+40, :]
        left_hand = x[:, :, LEFT_HAND_START:LEFT_HAND_START+21, :]
        right_hand = x[:, :, RIGHT_HAND_START:RIGHT_HAND_START+21, :]
        pose = x[:, :, POSE_START:POSE_START+5, :]
        

        lips = self._normalize_landmarks(lips, self.LIPS_MEAN, self.LIPS_STD)
        left_hand = self._normalize_landmarks(left_hand, self.LEFT_HANDS_MEAN, self.LEFT_HANDS_STD)
        right_hand = self._normalize_landmarks(right_hand, self.RIGHT_HANDS_MEAN, self.RIGHT_HANDS_STD)
        pose = self._normalize_landmarks(pose, self.POSE_MEAN, self.POSE_STD)
        
        x = self.embedding(lips, left_hand, right_hand, pose, non_empty_frame_idxs)
        
        x = self.transformer(x, attention_mask)
        
        x = self.pooling(x, attention_mask)
        
        if return_aux and training:
            return self.classifier(x, return_aux=True)
        else:
            return self.classifier(x)
    
    def _normalize_landmarks(self, landmarks, mean, std):

        missing_mask = (landmarks == 0.0)
        
        normalized = torch.where(
            missing_mask,
            torch.tensor(0.0, device=landmarks.device),
            (landmarks - mean) / (std + 1e-8)
        )
        
        normalized = torch.clamp(normalized, -5.0, 5.0)
        
        return normalized
    
    def compute_loss(self, logits, targets, aux_logits=None, confidence=None):

        main_loss = self.focal_loss_with_smoothing(logits, targets)
        
        total_loss = main_loss
        
        if aux_logits is not None:
            aux_weight = 0.2 
            for i, aux_logit in enumerate(aux_logits):
                layer_weight = aux_weight * (0.5 ** (len(aux_logits) - i - 1))
                aux_loss = self.focal_loss_with_smoothing(aux_logit, targets)
                total_loss += layer_weight * aux_loss
        
        if confidence is not None:
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = math.log(logits.shape[1])
            normalized_entropy = entropy / max_entropy
            
            predictions = torch.argmax(logits, dim=1)
            is_correct = (predictions == targets).float()
            target_confidence = 1.0 - normalized_entropy * (1.0 - is_correct)
            
            confidence_loss = F.mse_loss(confidence.squeeze(), target_confidence)
            total_loss += 0.1 * confidence_loss
        
        return total_loss
    
    def focal_loss_with_smoothing(self, logits, targets):
        
        num_classes = logits.shape[1]

        targets_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        focal_weights = self.focal_alpha * (1 - probs) ** self.focal_gamma
        focal_loss = -(focal_weights * targets_smooth * log_probs).sum(dim=1)
        
        return focal_loss.mean()

class CyclicLRWithWarmup:
    def __init__(self, optimizer, base_lr=1e-5, max_lr=3e-4, warmup_epochs=5, 
                 cycle_epochs=20, decay_factor=0.8):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.cycle_epochs = cycle_epochs
        self.decay_factor = decay_factor
        self.epoch = 0
        
    def step(self):
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.epoch / self.warmup_epochs)
        else:
            cycle_epoch = (self.epoch - self.warmup_epochs) % self.cycle_epochs
            cycle_num = (self.epoch - self.warmup_epochs) // self.cycle_epochs
            
            current_max_lr = self.max_lr * (self.decay_factor ** cycle_num)
            
            lr = self.base_lr + (current_max_lr - self.base_lr) * 0.5 * \
                 (1 + math.cos(math.pi * cycle_epoch / self.cycle_epochs))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.epoch += 1
        return lr

class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, x, y, non_empty_frame_idxs):
        if self.alpha <= 0:
            return x, y, non_empty_frame_idxs
        
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_indices = lam * non_empty_frame_idxs + (1 - lam) * non_empty_frame_idxs[index]
        
        return mixed_x, y, mixed_indices, index, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class ImprovedSignLanguageDataset(Dataset):
    def __init__(self, X, y, non_empty_frame_idxs, training=True, augment_prob=0.5):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.non_empty_frame_idxs = torch.tensor(non_empty_frame_idxs, dtype=torch.float32)
        self.training = training
        self.augment_prob = augment_prob
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        frame_idxs = self.non_empty_frame_idxs[idx].clone()
        
        if self.training and torch.rand(1).item() < self.augment_prob:
            x = self._augment_sequence(x, frame_idxs)
        
        return x, y, frame_idxs
    
    def _augment_sequence(self, x, frame_idxs):

        if torch.rand(1).item() < 0.3:
            x = self._time_warp(x, frame_idxs)
        
        if torch.rand(1).item() < 0.2:
            noise_std = 0.01
            noise = torch.randn_like(x) * noise_std
            mask = (x != 0.0).float()
            x = x + noise * mask
        
        if torch.rand(1).item() < 0.3:
            scale_factor = torch.rand(1).item() * 0.2 + 0.9
            x = x * scale_factor
        
        return x
        
    def _time_warp(self, x, frame_idxs):
        seq_len = x.shape[0]
        valid_frames = (frame_idxs != -1).sum().item()
        if valid_frames > 10: 
            speed_factor = 0.8 + torch.rand(1).item() * (1.2 - 0.8)
            new_len = int(valid_frames * speed_factor)
            new_len = max(5, min(new_len, seq_len))
            
            if new_len != valid_frames:
                valid_mask = frame_idxs != -1
                valid_x = x[valid_mask]
                
                old_indices = torch.linspace(0, len(valid_x) - 1, len(valid_x))
                new_indices = torch.linspace(0, len(valid_x) - 1, new_len)
                

                interp_x = torch.zeros(new_len, x.shape[1], x.shape[2])
                for i in range(new_len):
                    ratio = new_indices[i] / (len(valid_x) - 1) if len(valid_x) > 1 else 0
                    idx = int(ratio * (len(valid_x) - 1))
                    idx = min(idx, len(valid_x) - 1)
                    interp_x[i] = valid_x[idx]
                
                x[:new_len] = interp_x
                x[new_len:] = 0
                frame_idxs[:new_len] = torch.arange(new_len, dtype=frame_idxs.dtype)
                frame_idxs[new_len:] = -1
        
        return x