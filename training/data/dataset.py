"""
SLOVO Dataset with Krivov et al. 2024 training strategies.

Supports RGB frames, pre-extracted keypoints, or both.
Includes video augmentations, image augmentations, and sign boundary annotations.
"""

import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from typing import Optional, Tuple

# Non-mirrored signs in SLOVO (these should NOT be flipped)
FLIP_BLACKLIST = {739, 635, 148, 636}


class SLOVODataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        mode: str = "both",
        num_frames_rgb: int = 32,
        frame_interval: int = 2,
        num_frames_kp: int = 128,
        rgb_size: int = 224,
        rgb_transform=None,
        kp_augment: bool = False,
        label_map: Optional[dict] = None,
        video_aug: bool = False,
        image_aug: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.num_frames_rgb = num_frames_rgb
        self.frame_interval = frame_interval
        self.num_frames_kp = num_frames_kp
        self.rgb_size = rgb_size
        self.rgb_transform = rgb_transform
        self.kp_augment = kp_augment and (split == "train")
        self.video_aug = video_aug and (split == "train")
        self.image_aug = image_aug and (split == "train")
        self.is_train = (split == "train")

        # Load annotations
        ann_path = self.data_root / "annotations.csv"
        ann = pd.read_csv(ann_path, sep="\t")
        is_train = ann["train"].astype(str).str.strip().str.lower() == "true"
        if split == "train":
            self.df = ann[is_train].reset_index(drop=True)
        else:
            self.df = ann[~is_train].reset_index(drop=True)

        # Build label map
        if label_map is not None:
            self.label_map = label_map
        else:
            all_labels = sorted(ann["text"].unique())
            self.label_map = {label: i for i, label in enumerate(all_labels)}

        self.num_classes = len(self.label_map)

        # Paths
        self.video_dir = self.data_root / "slovo" / split
        kp_split_dir = self.data_root / "keypoints" / split
        kp_flat_dir = self.data_root / "keypoints"
        self.kp_dir = kp_split_dir if kp_split_dir.exists() else kp_flat_dir

        if mode in ("rgb", "both"):
            assert self.video_dir.exists(), f"Video dir not found: {self.video_dir}"
        if mode in ("keypoints", "both"):
            if not self.kp_dir.exists():
                print(f"WARNING: Keypoint dir not found: {self.kp_dir}.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        attachment_id = row["attachment_id"]
        label = self.label_map[row["text"]]

        # Sign boundaries from annotations (begin/end frame)
        begin = int(row["begin"]) if pd.notna(row.get("begin")) else 0
        end = int(row["end"]) if pd.notna(row.get("end")) else -1
        total_len = int(row["length"]) if pd.notna(row.get("length")) else 0

        result = {
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.mode in ("rgb", "both"):
            frames, iou_score, norm_begin, norm_end = self._load_rgb(
                attachment_id, begin, end, total_len, label
            )
            result["rgb"] = frames
            result["iou_score"] = torch.tensor(iou_score, dtype=torch.float32)
            result["sign_bounds"] = torch.tensor([norm_begin, norm_end], dtype=torch.float32)

        if self.mode in ("keypoints", "both"):
            kp, non_empty = self._load_keypoints(attachment_id)
            result["keypoints"] = kp
            result["non_empty_frame_idxs"] = non_empty

        return result

    # ---- RGB loading with Krivov-style sampling ----
    def _load_rgb(self, attachment_id, begin, end, total_len, label):
        video_path = self.video_dir / f"{attachment_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            dummy = torch.zeros(3, self.num_frames_rgb, self.rgb_size, self.rgb_size)
            return dummy, 1.0, 0.0, 1.0

        if end <= 0 or end > total_frames:
            end = total_frames

        # Random boundary shift (training only)
        if self.video_aug:
            shift_left = random.randint(-5, 5)
            shift_right = random.randint(-5, 5)
            begin = max(0, begin + shift_left)
            end = min(total_frames, end + shift_right)
            if end <= begin:
                end = min(begin + 1, total_frames)

        # Sample frames with interval (like Krivov: clip_len=32, interval=2)
        clip_len = self.num_frames_rgb
        interval = self.frame_interval
        needed = clip_len * interval  # frames of video we need to cover

        # Center the sampling window around the sign
        sign_center = (begin + end) // 2
        win_start = max(0, sign_center - needed // 2)
        win_end = win_start + needed

        if win_end > total_frames:
            win_end = total_frames
            win_start = max(0, win_end - needed)

        # Generate frame indices with interval
        indices = []
        for i in range(clip_len):
            idx = win_start + i * interval
            idx = min(idx, total_frames - 1)
            indices.append(idx)

        # If not enough frames, repeat last
        while len(indices) < clip_len:
            indices.append(indices[-1])

        # Video augmentation: speed change, random drop/add
        if self.video_aug:
            indices = self._video_augment(indices, total_frames)

        # Read frames
        frames = []
        for target_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((int(cap.get(4)), int(cap.get(3)), 3), dtype=np.uint8)
                              if len(frames) == 0
                              else frames[-1].copy())
        cap.release()

        # Resize to 300 first (like Krivov), then crop to 224
        resized_frames = []
        for f in frames:
            f = cv2.resize(f, (300, 300))
            resized_frames.append(f)

        # Image augmentations (per-frame)
        if self.image_aug:
            resized_frames = self._image_augment(resized_frames, label)

        # Random crop (train) or center crop (val) from 300 to 224
        if self.is_train:
            crop_y = random.randint(0, 300 - self.rgb_size)
            crop_x = random.randint(0, 300 - self.rgb_size)
        else:
            crop_y = (300 - self.rgb_size) // 2
            crop_x = (300 - self.rgb_size) // 2

        cropped = []
        for f in resized_frames:
            cropped.append(f[crop_y:crop_y + self.rgb_size, crop_x:crop_x + self.rgb_size])

        # To tensor: (T, H, W, C) -> (C, T, H, W)
        video = np.stack(cropped, axis=0)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0

        # Normalize (Krivov uses custom mean/std, we use ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        video = (video - mean) / std

        # Compute IoU score: overlap of sign with sampling window
        win_s = indices[0]
        win_e = indices[-1]
        overlap_start = max(win_s, begin)
        overlap_end = min(win_e, end)
        overlap = max(0, overlap_end - overlap_start)
        window_size = max(win_e - win_s, 1)
        iou_score = overlap / window_size

        # "no event" class gets IoU=1 (index 1000 in SLOVO)
        no_event_idx = self.num_classes - 1
        if label == no_event_idx:
            iou_score = 1.0

        # Normalized sign boundaries relative to sampling window
        norm_begin = (begin - win_s) / max(window_size, 1)
        norm_end = (end - win_s) / max(window_size, 1)
        norm_begin = max(0.0, min(1.0, norm_begin))
        norm_end = max(0.0, min(1.0, norm_end))

        if label == no_event_idx:
            norm_begin, norm_end = 0.0, 0.0

        return video, iou_score, norm_begin, norm_end

    # ---- Video augmentations (Krivov Section 3.1) ----
    def _video_augment(self, indices, total_frames):
        """Apply one random video augmentation."""
        r = random.random()

        if r < 0.25:
            # Speed up 2x: take every 2nd frame
            indices = indices[::2]
            # Pad to original length by extending
            while len(indices) < self.num_frames_rgb:
                next_idx = min(indices[-1] + self.frame_interval, total_frames - 1)
                indices.append(next_idx)
        elif r < 0.5:
            # Slow down 2x: duplicate each frame
            slowed = []
            for idx in indices:
                slowed.extend([idx, idx])
            indices = slowed[:self.num_frames_rgb]
        elif r < 0.75:
            # Random drop 10%
            n_drop = max(1, int(len(indices) * 0.1))
            drop_idxs = sorted(random.sample(range(len(indices)), n_drop), reverse=True)
            remaining = [idx for i, idx in enumerate(indices) if i not in drop_idxs]
            # Extend with frames after the end to maintain length
            while len(remaining) < self.num_frames_rgb:
                next_idx = min(remaining[-1] + self.frame_interval, total_frames - 1)
                remaining.append(next_idx)
            indices = remaining[:self.num_frames_rgb]
        else:
            # Random add 30%: duplicate random frames
            n_add = max(1, int(len(indices) * 0.3))
            add_positions = sorted(random.sample(range(len(indices)), min(n_add, len(indices))))
            expanded = []
            add_set = set(add_positions)
            for i, idx in enumerate(indices):
                expanded.append(idx)
                if i in add_set:
                    expanded.append(idx)  # duplicate
            indices = expanded[:self.num_frames_rgb]

        # Ensure exact length
        while len(indices) < self.num_frames_rgb:
            indices.append(indices[-1])
        indices = indices[:self.num_frames_rgb]

        return indices

    # ---- Image augmentations (Krivov Section 3.2) ----
    def _image_augment(self, frames, label):
        """Apply image augmentations to a list of numpy frames (H, W, 3) uint8."""
        augmented = []

        # Horizontal flip (with blacklist for non-mirrored signs)
        do_flip = random.random() < 0.5 and label not in FLIP_BLACKLIST
        # Color jitter params (same for all frames in clip)
        do_color = random.random() < 0.5
        brightness = 1.0 + random.uniform(-0.1, 0.1) if do_color else 1.0
        contrast = 1.0 + random.uniform(-0.005, 0.005) if do_color else 1.0
        # Salt and pepper noise
        do_noise = random.random() < 0.5
        noise_amount = random.uniform(0.001, 0.005) if do_noise else 0
        # Sharpness
        do_sharp = random.random() < 0.35
        # Random erasing (same region for all frames)
        do_erase = random.random() < 0.25
        if do_erase:
            h, w = frames[0].shape[:2]
            area_ratio = random.uniform(0.02, 0.33)
            eh = int(h * area_ratio ** 0.5)
            ew = int(w * area_ratio ** 0.5)
            ey = random.randint(0, max(h - eh, 0))
            ex = random.randint(0, max(w - ew, 0))

        for f in frames:
            if do_flip:
                f = np.fliplr(f).copy()

            if do_color:
                f = np.clip(f.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
                mean = f.mean()
                f = np.clip((f.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

            if do_noise and noise_amount > 0:
                mask = np.random.random(f.shape[:2])
                f = f.copy()
                f[mask < noise_amount / 2] = 0         # pepper
                f[mask > 1 - noise_amount / 2] = 255    # salt

            if do_sharp:
                factor = random.uniform(0.5, 2.0)
                kernel = np.array([[-1, -1, -1], [-1, 8 + factor, -1], [-1, -1, -1]]) / (factor + 1)
                f = cv2.filter2D(f, -1, kernel.astype(np.float32))
                f = np.clip(f, 0, 255).astype(np.uint8)

            if do_erase:
                f = f.copy()
                f[ey:ey + eh, ex:ex + ew] = np.random.randint(0, 255, (eh, ew, 3), dtype=np.uint8)

            augmented.append(f)

        return augmented

    # ---- Keypoint loading ----
    def _load_keypoints(self, attachment_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        from .extract_keypoints import N_SLR_LANDMARKS, SLR_LEFT_HAND_RANGE, SLR_RIGHT_HAND_RANGE

        N_KP = N_SLR_LANDMARKS
        N_DIMS = 2

        npz_path = self.kp_dir / f"{attachment_id}.npz"
        npy_path = self.kp_dir / f"{attachment_id}.npy"

        if npz_path.exists():
            data = np.load(str(npz_path))
            kp = data["keypoints"]
            scores = data["scores"]
        elif npy_path.exists():
            kp_old = np.load(str(npy_path))
            kp = kp_old[:, :, :2]
            if kp.shape[1] > N_KP:
                kp = kp[:, :N_KP, :]
            elif kp.shape[1] < N_KP:
                pad = np.zeros((kp.shape[0], N_KP - kp.shape[1], 2), dtype=np.float32)
                kp = np.concatenate([kp, pad], axis=1)
            scores = (np.abs(kp).sum(axis=-1) > 0).astype(np.float32)
        else:
            kp = np.zeros((1, N_KP, N_DIMS), dtype=np.float32)
            scores = np.zeros((1, N_KP), dtype=np.float32)

        lh_start, lh_end = SLR_LEFT_HAND_RANGE
        rh_start, rh_end = SLR_RIGHT_HAND_RANGE
        hand_score = scores[:, lh_start:rh_end].sum(axis=1)
        non_empty_mask = hand_score > 0

        if non_empty_mask.sum() == 0:
            non_empty_mask[:] = True

        kp_filtered = kp[non_empty_mask]
        n_valid = len(kp_filtered)

        target_len = self.num_frames_kp
        if n_valid < target_len:
            pad = np.zeros((target_len - n_valid, N_KP, N_DIMS), dtype=np.float32)
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

        kp_tensor = torch.from_numpy(kp_out).float()
        ne_tensor = torch.from_numpy(non_empty).float()

        if self.kp_augment:
            kp_tensor, ne_tensor = self._augment_keypoints(kp_tensor, ne_tensor)

        return kp_tensor, ne_tensor

    # ---- Keypoint augmentations ----
    def _augment_keypoints(self, kp, non_empty):
        if torch.rand(1).item() < 0.3:
            kp = self._mirror_hands(kp)
        if torch.rand(1).item() < 0.3:
            kp, non_empty = self._time_warp(kp, non_empty)
        if torch.rand(1).item() < 0.3:
            mask = (kp != 0).float()
            noise = torch.randn_like(kp) * 0.015
            kp = kp + noise * mask
        if torch.rand(1).item() < 0.3:
            scale = 0.85 + torch.rand(1).item() * 0.3
            kp = kp * scale
        if torch.rand(1).item() < 0.2:
            T, N, D = kp.shape
            drop_mask = torch.rand(N) > 0.15
            drop_mask = drop_mask.unsqueeze(0).unsqueeze(-1).expand(T, N, D)
            kp = kp * drop_mask.float()
        if torch.rand(1).item() < 0.2:
            angle = (torch.rand(1).item() - 0.5) * 0.3
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x, y = kp[..., 0].clone(), kp[..., 1].clone()
            kp[..., 0] = cos_a * x - sin_a * y
            kp[..., 1] = sin_a * x + cos_a * y
        return kp, non_empty

    def _mirror_hands(self, kp):
        from .extract_keypoints import SLR_LEFT_HAND_RANGE, SLR_RIGHT_HAND_RANGE
        lh_s, lh_e = SLR_LEFT_HAND_RANGE
        rh_s, rh_e = SLR_RIGHT_HAND_RANGE
        mirrored = kp.clone()
        mirrored[:, lh_s:lh_e, :] = kp[:, rh_s:rh_e, :]
        mirrored[:, rh_s:rh_e, :] = kp[:, lh_s:lh_e, :]
        mask = (mirrored != 0).float()
        mirrored[..., 0] = (1.0 - mirrored[..., 0]) * mask[..., 0]
        return mirrored

    def _time_warp(self, kp, non_empty):
        T = kp.shape[0]
        valid = (non_empty != -1).sum().int().item()
        if valid < 10:
            return kp, non_empty
        speed = 0.8 + torch.rand(1).item() * 0.4
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

    ann = pd.read_csv(os.path.join(cfg["data_root"], "annotations.csv"), sep="\t")
    all_labels = sorted(ann["text"].unique())
    label_map = {label: i for i, label in enumerate(all_labels)}

    train_ds = SLOVODataset(
        data_root=cfg["data_root"],
        split="train",
        mode=cfg.get("mode", "both"),
        num_frames_rgb=cfg.get("num_frames_rgb", 32),
        frame_interval=cfg.get("frame_interval", 2),
        num_frames_kp=cfg.get("num_frames_kp", 128),
        rgb_size=cfg.get("rgb_size", 224),
        kp_augment=True,
        label_map=label_map,
        video_aug=cfg.get("video_aug", True),
        image_aug=cfg.get("image_aug", True),
    )

    val_ds = SLOVODataset(
        data_root=cfg["data_root"],
        split="test",
        mode=cfg.get("mode", "both"),
        num_frames_rgb=cfg.get("num_frames_rgb", 32),
        frame_interval=cfg.get("frame_interval", 2),
        num_frames_kp=cfg.get("num_frames_kp", 128),
        rgb_size=cfg.get("rgb_size", 224),
        kp_augment=False,
        label_map=label_map,
        video_aug=False,
        image_aug=False,
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
