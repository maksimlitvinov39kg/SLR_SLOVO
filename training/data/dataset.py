"""
SLOVO Dataset: supports RGB frames, pre-extracted keypoints, or both.

Expects directory layout:
    data_root/
    ├── annotations.csv          # tab-separated: attachment_id, text, train/test, begin, end
    ├── slovo/
    │   ├── train/*.mp4
    │   └── test/*.mp4
    └── keypoints/               # optional, from extract_keypoints.py
        ├── train/*.npy
        └── test/*.npy
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from typing import Optional, Tuple


class SLOVODataset(Dataset):
    """
    Dual-stream SLOVO dataset.

    Modes:
        - "rgb": return video frames only
        - "keypoints": return pre-extracted keypoints only
        - "both": return (frames, keypoints)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        mode: str = "both",            # "rgb", "keypoints", "both"
        num_frames_rgb: int = 32,       # frames to sample for RGB branch
        num_frames_kp: int = 128,       # frames for keypoint branch
        rgb_size: int = 224,            # spatial size for RGB
        rgb_transform=None,             # torchvision transforms for RGB
        kp_augment: bool = False,       # whether to augment keypoints
        label_map: Optional[dict] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.num_frames_rgb = num_frames_rgb
        self.num_frames_kp = num_frames_kp
        self.rgb_size = rgb_size
        self.rgb_transform = rgb_transform
        self.kp_augment = kp_augment and (split == "train")

        # Load annotations
        ann_path = self.data_root / "annotations.csv"
        ann = pd.read_csv(ann_path, sep="\t")
        self.df = ann[ann["split"] == split].reset_index(drop=True)

        # Build label map: text -> int
        if label_map is not None:
            self.label_map = label_map
        else:
            all_labels = sorted(ann["text"].unique())
            self.label_map = {label: i for i, label in enumerate(all_labels)}

        self.num_classes = len(self.label_map)

        # Paths
        self.video_dir = self.data_root / "slovo" / split
        self.kp_dir = self.data_root / "keypoints" / split

        # Verify data exists
        if mode in ("rgb", "both"):
            assert self.video_dir.exists(), f"Video dir not found: {self.video_dir}"
        if mode in ("keypoints", "both"):
            if not self.kp_dir.exists():
                print(f"WARNING: Keypoint dir not found: {self.kp_dir}. "
                      f"Run extract_keypoints.py first or switch to mode='rgb'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        attachment_id = row["attachment_id"]
        label = self.label_map[row["text"]]

        result = {"label": torch.tensor(label, dtype=torch.long)}

        if self.mode in ("rgb", "both"):
            frames = self._load_rgb(attachment_id)
            result["rgb"] = frames

        if self.mode in ("keypoints", "both"):
            kp, non_empty = self._load_keypoints(attachment_id)
            result["keypoints"] = kp
            result["non_empty_frame_idxs"] = non_empty

        return result

    # ---- RGB loading ----
    def _load_rgb(self, attachment_id: str) -> torch.Tensor:
        video_path = self.video_dir / f"{attachment_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return torch.zeros(3, self.num_frames_rgb, self.rgb_size, self.rgb_size)

        # Uniform temporal sampling
        indices = np.linspace(0, total - 1, self.num_frames_rgb, dtype=int)
        frames = []

        for target_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.rgb_size, self.rgb_size))
                frames.append(frame)
            else:
                frames.append(np.zeros((self.rgb_size, self.rgb_size, 3), dtype=np.uint8))

        cap.release()

        # (T, H, W, C) -> (C, T, H, W) for video models
        video = np.stack(frames, axis=0)  # (T, H, W, 3)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        video = (video - mean) / std

        if self.rgb_transform is not None:
            video = self.rgb_transform(video)

        return video

    # ---- Keypoint loading ----
    def _load_keypoints(self, attachment_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        kp_path = self.kp_dir / f"{attachment_id}.npy"

        if kp_path.exists():
            kp = np.load(str(kp_path))  # (N_frames, 87, 3)
        else:
            # Fallback: extract on-the-fly (slow but works)
            from .extract_keypoints import extract_keypoints_from_video
            video_path = self.video_dir.parent / self.split / f"{attachment_id}.mp4"
            kp = extract_keypoints_from_video(str(video_path), max_frames=self.num_frames_kp)

        # Filter empty frames (where both hands are zero)
        hand_sum = np.abs(kp[:, 40:82, :]).sum(axis=(1, 2))  # hands region
        non_empty_mask = hand_sum > 0

        if non_empty_mask.sum() == 0:
            non_empty_mask[:] = True  # fallback: keep all

        kp_filtered = kp[non_empty_mask]
        n_valid = len(kp_filtered)

        # Pad/resample to fixed length
        target_len = self.num_frames_kp
        if n_valid < target_len:
            pad = np.zeros((target_len - n_valid, 87, 3), dtype=np.float32)
            kp_out = np.concatenate([kp_filtered, pad], axis=0)
            non_empty = np.concatenate([
                np.arange(n_valid, dtype=np.float32),
                np.full(target_len - n_valid, -1.0, dtype=np.float32)
            ])
        elif n_valid > target_len:
            indices = np.linspace(0, n_valid - 1, target_len, dtype=int)
            kp_out = kp_filtered[indices]
            non_empty = np.arange(target_len, dtype=np.float32)
        else:
            kp_out = kp_filtered
            non_empty = np.arange(target_len, dtype=np.float32)

        kp_tensor = torch.from_numpy(kp_out).float()     # (128, 87, 3)
        ne_tensor = torch.from_numpy(non_empty).float()   # (128,)

        if self.kp_augment:
            kp_tensor, ne_tensor = self._augment_keypoints(kp_tensor, ne_tensor)

        return kp_tensor, ne_tensor

    # ---- Keypoint augmentations ----
    def _augment_keypoints(self, kp, non_empty):
        """Apply augmentations to keypoint sequences."""

        # Mirror: swap left/right hand + flip x-coordinate
        if torch.rand(1).item() < 0.3:
            kp = self._mirror_hands(kp)

        # Time warp: random speed change
        if torch.rand(1).item() < 0.3:
            kp, non_empty = self._time_warp(kp, non_empty)

        # Gaussian noise on non-zero keypoints
        if torch.rand(1).item() < 0.3:
            mask = (kp != 0).float()
            noise = torch.randn_like(kp) * 0.015
            kp = kp + noise * mask

        # Random scale
        if torch.rand(1).item() < 0.3:
            scale = 0.85 + torch.rand(1).item() * 0.3  # [0.85, 1.15]
            kp = kp * scale

        # Coordinate dropout: zero out random landmarks
        if torch.rand(1).item() < 0.2:
            T, N, D = kp.shape
            drop_mask = torch.rand(N) > 0.15  # keep 85% of landmarks
            drop_mask = drop_mask.unsqueeze(0).unsqueeze(-1).expand(T, N, D)
            kp = kp * drop_mask.float()

        # 2D rotation (small angle)
        if torch.rand(1).item() < 0.2:
            angle = (torch.rand(1).item() - 0.5) * 0.3  # ±0.15 rad (~8.5°)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x, y = kp[..., 0].clone(), kp[..., 1].clone()
            kp[..., 0] = cos_a * x - sin_a * y
            kp[..., 1] = sin_a * x + cos_a * y

        return kp, non_empty

    def _mirror_hands(self, kp):
        """Swap left/right hand landmarks and flip x-coordinate."""
        mirrored = kp.clone()
        # Swap left hand (40:61) with right hand (61:82)
        mirrored[:, 40:61, :] = kp[:, 61:82, :]
        mirrored[:, 61:82, :] = kp[:, 40:61, :]
        # Flip x-coordinate (mirror horizontally)
        mask = (mirrored != 0).float()
        mirrored[..., 0] = (1.0 - mirrored[..., 0]) * mask[..., 0]
        return mirrored

    def _time_warp(self, kp, non_empty):
        T = kp.shape[0]
        valid = (non_empty != -1).sum().int().item()
        if valid < 10:
            return kp, non_empty

        speed = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
        new_len = max(5, min(int(valid * speed), T))

        if new_len == valid:
            return kp, non_empty

        old_indices = torch.linspace(0, valid - 1, new_len).long()
        kp_valid = kp[:valid]
        warped = kp_valid[old_indices]

        result = torch.zeros_like(kp)
        result[:new_len] = warped

        ne = torch.full_like(non_empty, -1.0)
        ne[:new_len] = torch.arange(new_len, dtype=torch.float32)

        return result, ne


def build_dataloaders(cfg):
    """Build train/val dataloaders from config dict."""
    from torch.utils.data import DataLoader

    # Build label map from full dataset
    ann = pd.read_csv(os.path.join(cfg["data_root"], "annotations.csv"), sep="\t")
    all_labels = sorted(ann["text"].unique())
    label_map = {label: i for i, label in enumerate(all_labels)}

    train_ds = SLOVODataset(
        data_root=cfg["data_root"],
        split="train",
        mode=cfg.get("mode", "both"),
        num_frames_rgb=cfg.get("num_frames_rgb", 32),
        num_frames_kp=cfg.get("num_frames_kp", 128),
        rgb_size=cfg.get("rgb_size", 224),
        kp_augment=True,
        label_map=label_map,
    )

    val_ds = SLOVODataset(
        data_root=cfg["data_root"],
        split="test",
        mode=cfg.get("mode", "both"),
        num_frames_rgb=cfg.get("num_frames_rgb", 32),
        num_frames_kp=cfg.get("num_frames_kp", 128),
        rgb_size=cfg.get("rgb_size", 224),
        kp_augment=False,
        label_map=label_map,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if cfg.get("num_workers", 4) > 0 else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True if cfg.get("num_workers", 4) > 0 else False,
    )

    return train_loader, val_loader, label_map
