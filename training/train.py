"""
Training script for SLOVO Sign Language Recognition.
Implements Krivov et al. 2024 training strategies + keypoint fusion.

Usage:
    python train.py --config configs/rgb_only.yaml
    python train.py --config configs/fusion.yaml
"""

import os
import yaml
import argparse
import time
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data.dataset import build_dataloaders
from models.fusion import DualStreamModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---- Loss: IoU-balanced CrossEntropy (Krivov Section 3.3) ----

class IoUBalancedCELoss(nn.Module):
    """
    CrossEntropy where logits are scaled by IoU scores.
    IoU score = overlap of sign with sampling window / window size.
    """
    def __init__(self, num_classes, smoothing=0.1, no_event_idx=1000):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.no_event_idx = no_event_idx

    def forward(self, logits, targets, iou_scores=None):
        # Label smoothing
        with torch.no_grad():
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            smooth = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes

        log_probs = F.log_softmax(logits, dim=1)

        # Scale by IoU scores if provided
        if iou_scores is not None:
            # iou_scores: (B,) in [0, 1]
            # Scale the log probs for the correct class by IoU
            iou_weight = iou_scores.unsqueeze(1)  # (B, 1)
            loss = -(smooth * log_probs * iou_weight).sum(dim=1)
        else:
            loss = -(smooth * log_probs).sum(dim=1)

        return loss.mean()


# ---- MixUp / CutMix (Krivov uses alpha=0.8/1.0) ----

def mixup_cutmix(batch, mixup_alpha=0.8, cutmix_alpha=1.0):
    """Apply either MixUp or CutMix with 50/50 probability."""
    if np.random.rand() < 0.5:
        return _mixup(batch, mixup_alpha)
    else:
        return _cutmix(batch, cutmix_alpha)


def _mixup(batch, alpha):
    if alpha <= 0:
        return batch, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    B = batch["label"].size(0)
    index = torch.randperm(B, device=batch["label"].device)

    mixed = {}
    mixed["label"] = batch["label"]
    if "rgb" in batch:
        mixed["rgb"] = lam * batch["rgb"] + (1 - lam) * batch["rgb"][index]
    if "keypoints" in batch:
        mixed["keypoints"] = lam * batch["keypoints"] + (1 - lam) * batch["keypoints"][index]
    if "non_empty_frame_idxs" in batch:
        mixed["non_empty_frame_idxs"] = batch["non_empty_frame_idxs"]
    if "iou_score" in batch:
        mixed["iou_score"] = lam * batch["iou_score"] + (1 - lam) * batch["iou_score"][index]
    if "sign_bounds" in batch:
        mixed["sign_bounds"] = lam * batch["sign_bounds"] + (1 - lam) * batch["sign_bounds"][index]

    return mixed, batch["label"][index], index, lam


def _cutmix(batch, alpha):
    if alpha <= 0 or "rgb" not in batch:
        return batch, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    B = batch["label"].size(0)
    index = torch.randperm(B, device=batch["label"].device)

    mixed = {k: v for k, v in batch.items()}

    # CutMix on RGB only
    rgb = batch["rgb"]  # (B, C, T, H, W)
    _, _, _, H, W = rgb.shape
    cut_h = int(H * (1 - lam) ** 0.5)
    cut_w = int(W * (1 - lam) ** 0.5)
    cy = np.random.randint(0, H)
    cx = np.random.randint(0, W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    rgb_mixed = rgb.clone()
    rgb_mixed[:, :, :, y1:y2, x1:x2] = rgb[index, :, :, y1:y2, x1:x2]
    mixed["rgb"] = rgb_mixed

    # Adjust lambda to actual cut ratio
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)

    return mixed, batch["label"][index], index, lam


# ---- Training loop ----

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_accum = cfg.get("gradient_accumulation", 1)
    use_mixup = cfg.get("mixup_alpha", 0) > 0
    use_regression = cfg.get("use_regression_head", True)
    huber_weight = cfg.get("huber_loss_weight", 5.0)

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # MixUp / CutMix
        if use_mixup and np.random.rand() < 0.5:
            batch, targets_b, _, lam = mixup_cutmix(
                batch,
                mixup_alpha=cfg.get("mixup_alpha", 0.8),
                cutmix_alpha=cfg.get("cutmix_alpha", 1.0),
            )
        else:
            targets_b = None
            lam = 1.0

        with autocast('cuda', dtype=torch.bfloat16 if cfg.get("bf16", False) else torch.float16):
            output = model(batch)
            logits = output["logits"]

            iou_scores = batch.get("iou_score", None)

            if targets_b is not None:
                loss_cls = lam * criterion(logits, batch["label"], iou_scores) + \
                           (1 - lam) * criterion(logits, targets_b, iou_scores)
            else:
                loss_cls = criterion(logits, batch["label"], iou_scores)

            loss = loss_cls

            # Regression loss (Huber) for sign boundaries
            if use_regression and "bounds_pred" in output and "sign_bounds" in batch:
                bounds_pred = output["bounds_pred"]  # (B, 2)
                bounds_target = batch["sign_bounds"]  # (B, 2)
                loss_reg = F.huber_loss(bounds_pred, bounds_target, delta=1.0)
                loss = loss + huber_weight * loss_reg

            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 5.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
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

        with autocast('cuda', dtype=torch.bfloat16 if cfg.get("bf16", False) else torch.float16):
            output = model(batch)
            logits = output["logits"]
            loss = criterion(logits, batch["label"])

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == batch["label"]).sum().item()

        _, top5_preds = logits.topk(5, dim=1)
        correct_top5 += (top5_preds == batch["label"].unsqueeze(1)).any(dim=1).sum().item()
        total += batch["label"].size(0)

    acc = 100.0 * correct / total
    top5_acc = 100.0 * correct_top5 / total
    return total_loss / len(loader), acc, top5_acc


def build_optimizer(model, cfg):
    mode = cfg.get("mode", "both")
    base_lr = float(cfg.get("lr", 0.0016))

    if mode == "both":
        param_groups = [
            {"params": list(model.get_rgb_params()),
             "lr": base_lr * cfg.get("rgb_lr_mult", 0.1),
             "weight_decay": cfg.get("weight_decay", 0.05)},
            {"params": list(model.get_kp_params()),
             "lr": base_lr,
             "weight_decay": cfg.get("weight_decay", 0.05)},
            {"params": list(model.get_fusion_params()),
             "lr": base_lr,
             "weight_decay": cfg.get("weight_decay", 0.05)},
        ]
        param_groups = [g for g in param_groups if len(list(g["params"])) > 0]
    else:
        param_groups = [
            {"params": model.parameters(),
             "lr": base_lr,
             "weight_decay": cfg.get("weight_decay", 0.05)},
        ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    return optimizer


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Linear warmup 20 epochs + Cosine decay (Krivov recipe)."""
    warmup_epochs = cfg.get("warmup_epochs", 20)
    total_epochs = cfg.get("epochs", 100)
    cosine_epochs = total_epochs - warmup_epochs
    min_lr_ratio = cfg.get("min_lr_ratio", 0.001)

    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = cosine_epochs * steps_per_epoch

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=min_lr_ratio, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=0
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    return scheduler


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
        num_frames_rgb=cfg.get("num_frames_rgb", 32),
        drop_path_rate=cfg.get("drop_path_rate", 0.2),
        kp_d_model=cfg.get("kp_d_model", 256),
        kp_num_heads=cfg.get("kp_num_heads", 8),
        kp_num_layers=cfg.get("kp_num_layers", 6),
        kp_dropout=cfg.get("kp_dropout", 0.2),
        kp_drop_path=cfg.get("kp_drop_path", 0.15),
        fusion_type=cfg.get("fusion_type", "gated"),
        fused_dim=cfg.get("fused_dim", 512),
        mode=cfg.get("mode", "both"),
        classifier_dropout=cfg.get("classifier_dropout", 0.0),
        label_smooth_eps=cfg.get("label_smoothing", 0.1),
        use_regression_head=cfg.get("use_regression_head", True),
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss
    criterion = IoUBalancedCELoss(
        num_classes=num_classes,
        smoothing=cfg.get("label_smoothing", 0.1),
        no_event_idx=num_classes - 1,
    )

    # Optimizer & scheduler (Krivov: linear warmup 20ep + cosine)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    scaler = GradScaler('cuda')

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

    # Output
    output_dir = Path(cfg.get("output_dir", "checkpoints")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Training
    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 7)
    min_delta = cfg.get("min_delta", 0.003)
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

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch + 1)
        writer.add_scalar("accuracy/val_top5", val_top5, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # Save best (with min_delta like Krivov)
        is_best = val_acc > best_acc + min_delta * 100
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
            # Only count "no improvement" AFTER warmup completes.
            # During warmup LR is tiny so model can't improve meaningfully.
            warmup_epochs = cfg.get("warmup_epochs", 20)
            if epoch + 1 > warmup_epochs:
                no_improve += 1

        if (epoch + 1) % cfg.get("save_every", 3) == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
            }, output_dir / f"epoch_{epoch+1}.pt")

        if no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
