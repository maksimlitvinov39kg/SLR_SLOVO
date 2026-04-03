"""
Training script for SLOVO Sign Language Recognition.

Usage:
    python train.py --config configs/fusion.yaml
    python train.py --config configs/rgb_only.yaml
    python train.py --config configs/keypoint_only.yaml

Supports:
    - Single-stream (RGB or Keypoints) and Dual-stream (fusion) training
    - Differential learning rates for pretrained vs. new modules
    - MixUp augmentation
    - Cosine annealing with warm restarts
    - AMP (mixed precision) for H100
    - Gradient accumulation
    - TensorBoard logging
"""

import os
import sys
import yaml
import argparse
import time
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data.dataset import build_dataloaders
from models.fusion import DualStreamModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---- Loss functions ----

class FocalLossWithSmoothing(nn.Module):
    """Focal loss with label smoothing. Better for imbalanced 1000-class setting."""

    def __init__(self, num_classes, alpha=1.0, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Label smoothing
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        smooth_targets = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        focal_weights = self.alpha * (1 - probs) ** self.gamma
        loss = -(focal_weights * smooth_targets * log_probs).sum(dim=1)
        return loss.mean()


def mixup_data(batch, alpha=0.4):
    """Apply MixUp to a batch. Returns mixed batch + targets for mixup criterion."""
    if alpha <= 0:
        return batch, None, None, 1.0

    lam = np.random.beta(alpha, alpha)
    B = batch["label"].size(0)
    index = torch.randperm(B, device=batch["label"].device)

    mixed_batch = {}
    mixed_batch["label"] = batch["label"]

    if "rgb" in batch:
        mixed_batch["rgb"] = lam * batch["rgb"] + (1 - lam) * batch["rgb"][index]
    if "keypoints" in batch:
        mixed_batch["keypoints"] = lam * batch["keypoints"] + (1 - lam) * batch["keypoints"][index]
    if "non_empty_frame_idxs" in batch:
        mixed_batch["non_empty_frame_idxs"] = batch["non_empty_frame_idxs"]  # don't mix masks

    return mixed_batch, batch["label"][index], index, lam


# ---- Training loop ----

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_accum = cfg.get("gradient_accumulation", 1)
    use_mixup = cfg.get("mixup_alpha", 0) > 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # Move to device
        batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # MixUp
        if use_mixup and np.random.rand() < 0.5:
            batch, targets_b, _, lam = mixup_data(batch, cfg["mixup_alpha"])
        else:
            targets_b = None
            lam = 1.0

        # Forward
        with autocast(dtype=torch.bfloat16 if cfg.get("bf16", False) else torch.float16):
            logits = model(batch)

            if targets_b is not None:
                loss = lam * criterion(logits, batch["label"]) + \
                       (1 - lam) * criterion(logits, targets_b)
            else:
                loss = criterion(logits, batch["label"])

            loss = loss / grad_accum

        # Backward
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and cfg.get("scheduler_step_per_batch", False):
                scheduler.step()

        total_loss += loss.item() * grad_accum
        preds = logits.argmax(dim=1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)

        if step % cfg.get("log_every", 50) == 0:
            lr = optimizer.param_groups[0]["lr"]
            acc = 100.0 * correct / max(total, 1)
            print(f"  [Step {step}/{len(loader)}] loss={total_loss/(step+1):.4f} "
                  f"acc={acc:.2f}% lr={lr:.2e}")

    if scheduler is not None and not cfg.get("scheduler_step_per_batch", False):
        scheduler.step()

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    model.eval()
    total_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0

    for batch in loader:
        batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        with autocast(dtype=torch.bfloat16 if cfg.get("bf16", False) else torch.float16):
            logits = model(batch)
            loss = criterion(logits, batch["label"])

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == batch["label"]).sum().item()

        # Top-5
        _, top5_preds = logits.topk(5, dim=1)
        correct_top5 += (top5_preds == batch["label"].unsqueeze(1)).any(dim=1).sum().item()

        total += batch["label"].size(0)

    acc = 100.0 * correct / total
    top5_acc = 100.0 * correct_top5 / total
    return total_loss / len(loader), acc, top5_acc


def build_optimizer(model, cfg):
    """Build optimizer with differential learning rates."""
    mode = cfg.get("mode", "both")
    base_lr = float(cfg.get("lr", 1e-4))

    if mode == "both":
        param_groups = [
            {
                "params": list(model.get_rgb_params()),
                "lr": base_lr * cfg.get("rgb_lr_mult", 0.1),  # lower LR for pretrained
                "weight_decay": cfg.get("weight_decay", 0.05),
            },
            {
                "params": list(model.get_kp_params()),
                "lr": base_lr,
                "weight_decay": cfg.get("weight_decay", 0.05),
            },
            {
                "params": list(model.get_fusion_params()),
                "lr": base_lr,
                "weight_decay": cfg.get("weight_decay", 0.05),
            },
        ]
        # Filter empty groups
        param_groups = [g for g in param_groups if len(list(g["params"])) > 0]
    else:
        param_groups = [
            {
                "params": model.parameters(),
                "lr": base_lr,
                "weight_decay": cfg.get("weight_decay", 0.05),
            }
        ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    return optimizer


def build_scheduler(optimizer, cfg, steps_per_epoch):
    sched_type = cfg.get("scheduler", "cosine_warm_restarts")

    if sched_type == "cosine_warm_restarts":
        T_0 = cfg.get("cosine_T0", 10)
        T_mult = cfg.get("cosine_T_mult", 2)
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)

    elif sched_type == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            epochs=cfg["epochs"],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )

    return None


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print(f"Config: {cfg}")
    print(f"Mode: {cfg.get('mode', 'both')}")

    # Data
    train_loader, val_loader, label_map = build_dataloaders(cfg)
    num_classes = len(label_map)
    print(f"Num classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = DualStreamModel(
        num_classes=num_classes,
        rgb_backbone=cfg.get("rgb_backbone", "mvit_v2_s"),
        rgb_pretrained=cfg.get("rgb_pretrained", True),
        rgb_freeze_stages=cfg.get("rgb_freeze_stages", 0),
        kp_d_model=cfg.get("kp_d_model", 256),
        kp_num_heads=cfg.get("kp_num_heads", 8),
        kp_num_layers=cfg.get("kp_num_layers", 6),
        kp_dropout=cfg.get("kp_dropout", 0.2),
        kp_drop_path=cfg.get("kp_drop_path", 0.15),
        fusion_type=cfg.get("fusion_type", "gated"),
        fused_dim=cfg.get("fused_dim", 512),
        mode=cfg.get("mode", "both"),
        classifier_dropout=cfg.get("classifier_dropout", 0.3),
    ).cuda()

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss
    criterion = FocalLossWithSmoothing(
        num_classes=num_classes,
        alpha=cfg.get("focal_alpha", 1.0),
        gamma=cfg.get("focal_gamma", 2.0),
        smoothing=cfg.get("label_smoothing", 0.05),
    )

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # AMP
    scaler = GradScaler()

    # Resume
    start_epoch = 0
    best_acc = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # Output dir
    output_dir = Path(cfg.get("output_dir", "checkpoints")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Training loop
    epochs = cfg.get("epochs", 50)
    patience = cfg.get("patience", 15)
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, epoch
        )

        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, cfg)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% val_top5={val_top5:.2f}% | "
              f"time={elapsed:.0f}s")

        # TensorBoard logging
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch + 1)
        writer.add_scalar("accuracy/val_top5", val_top5, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # Save best
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
                "label_map": label_map,
            }, output_dir / "best.pt")
            print(f"  -> New best: {best_acc:.2f}%")
        else:
            no_improve += 1

        # Save periodic
        if (epoch + 1) % cfg.get("save_every", 10) == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
            }, output_dir / f"epoch_{epoch+1}.pt")

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
