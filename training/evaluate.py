"""
Evaluation script with Test-Time Augmentation (TTA) and ensemble support.

Usage:
    # Single model eval
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/fusion.yaml

    # TTA (mirror + multi-crop temporal)
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/fusion.yaml --tta

    # Ensemble multiple models
    python evaluate.py --ensemble ckpt1.pt ckpt2.pt ckpt3.pt --config configs/fusion.yaml
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from data.dataset import build_dataloaders
from models.fusion import DualStreamModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ensemble", nargs="+", default=None, help="Multiple checkpoint paths")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--output", type=str, default=None, help="Save predictions to .npy")
    return parser.parse_args()


def load_model(checkpoint_path, cfg, num_classes):
    model = DualStreamModel(
        num_classes=num_classes,
        rgb_backbone=cfg.get("rgb_backbone", "mvit_v2_s"),
        rgb_pretrained=False,  # don't need pretrained weights for eval
        kp_d_model=cfg.get("kp_d_model", 256),
        kp_num_heads=cfg.get("kp_num_heads", 8),
        kp_num_layers=cfg.get("kp_num_layers", 6),
        kp_dropout=cfg.get("kp_dropout", 0.2),
        kp_drop_path=0.0,  # no drop path at eval
        fusion_type=cfg.get("fusion_type", "gated"),
        fused_dim=cfg.get("fused_dim", 512),
        mode=cfg.get("mode", "both"),
        classifier_dropout=0.0,
    ).cuda()

    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def tta_forward(model, batch, cfg):
    """
    Test-Time Augmentation: average predictions over augmented versions.

    Augmentations:
    1. Original
    2. Mirrored keypoints (swap hands, flip x)
    3. Temporal shift (offset by a few frames)
    """
    logits_list = []

    # 1. Original
    with autocast(dtype=torch.bfloat16 if cfg.get("bf16", False) else torch.float16):
        logits_list.append(model(batch))

    # 2. Mirror keypoints (RTMW SLR layout: left hand 30:51, right hand 51:72)
    if "keypoints" in batch:
        mirrored = batch.copy()
        kp = batch["keypoints"].clone()
        kp_mirror = kp.clone()
        kp_mirror[:, :, 30:51, :] = kp[:, :, 51:72, :]
        kp_mirror[:, :, 51:72, :] = kp[:, :, 30:51, :]
        # Flip x coordinate
        mask = (kp_mirror != 0).float()
        kp_mirror[..., 0] = (1.0 - kp_mirror[..., 0]) * mask[..., 0]
        mirrored["keypoints"] = kp_mirror

        with autocast(dtype=torch.bfloat16 if cfg.get("bf16", False) else torch.float16):
            logits_list.append(model(mirrored))

    # 3. Horizontal flip for RGB
    if "rgb" in batch:
        flipped = batch.copy()
        flipped["rgb"] = torch.flip(batch["rgb"], dims=[-1])  # flip W dimension
        with autocast(dtype=torch.bfloat16 if cfg.get("bf16", False) else torch.float16):
            logits_list.append(model(flipped))

    # Average logits
    avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return avg_logits


@torch.no_grad()
def evaluate(models, loader, cfg, use_tta=False):
    """Evaluate one or more models (ensemble)."""
    all_preds = []
    all_labels = []
    all_logits = []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        ensemble_logits = []
        for model in models:
            if use_tta:
                logits = tta_forward(model, batch, cfg)
            else:
                with autocast(dtype=torch.bfloat16 if cfg.get("bf16", False) else torch.float16):
                    logits = model(batch)
            ensemble_logits.append(logits)

        # Average across ensemble
        avg_logits = torch.stack(ensemble_logits, dim=0).mean(dim=0)

        preds = avg_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].cpu().numpy())
        all_logits.append(avg_logits.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0).numpy()

    # Metrics
    top1 = accuracy_score(all_labels, all_preds) * 100

    # Top-5
    top5_preds = np.argsort(all_logits, axis=1)[:, -5:]
    top5_correct = np.array([l in p for l, p in zip(all_labels, top5_preds)])
    top5 = top5_correct.mean() * 100

    f1 = f1_score(all_labels, all_preds, average="macro") * 100
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Top-1 Accuracy: {top1:.2f}%")
    print(f"  Top-5 Accuracy: {top5:.2f}%")
    print(f"  F1-Score (macro): {f1:.2f}%")
    print(f"  Precision (macro): {precision:.2f}%")
    print(f"  Recall (macro): {recall:.2f}%")
    print(f"{'='*50}")

    if use_tta:
        print(f"  (with TTA enabled)")
    if len(models) > 1:
        print(f"  (ensemble of {len(models)} models)")

    return {
        "top1": top1, "top5": top5,
        "f1": f1, "precision": precision, "recall": recall,
        "predictions": all_preds, "logits": all_logits,
    }


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    _, val_loader, label_map = build_dataloaders(cfg)
    num_classes = len(label_map)

    # Load model(s)
    if args.ensemble:
        print(f"Loading ensemble of {len(args.ensemble)} models...")
        models = [load_model(p, cfg, num_classes) for p in args.ensemble]
    elif args.checkpoint:
        models = [load_model(args.checkpoint, cfg, num_classes)]
    else:
        raise ValueError("Provide --checkpoint or --ensemble")

    results = evaluate(models, val_loader, cfg, use_tta=args.tta)

    if args.output:
        np.save(args.output, {
            "predictions": results["predictions"],
            "logits": results["logits"],
        })
        print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
