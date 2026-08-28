"""
Fine-tunes the hybrid model on the synthetic degradation dataset.
Two-stage schedule: freeze the pretrained backbone for a warm-up (train only
the new fusion/heads), then unfreeze and fine-tune end-to-end at a lower LR —
this converges faster and more stably than fine-tuning everything from step 1.

v4 approach, replacing the class-weighted-loss experiments (v2/v3):
  - ConvNeXt-Tiny backbone: 82.1% ImageNet top-1 vs ResNet34's 73.3% at a
    comparable parameter count, and the better choice over a transformer
    backbone (e.g. Swin) specifically because this is a low-data fine-tuning
    regime (~38k images) where transformers don't have enough data to earn
    their usual accuracy edge.
  - Oversampling ("none" is ~3x under-represented) via WeightedRandomSampler
    instead of loss reweighting. v2's full inverse-frequency *loss* weighting
    fixed "none" recall but measurably hurt accuracy on the other 6 classes
    by distorting the loss landscape; oversampling rebalances what the model
    sees during training without changing what the loss function optimizes
    for, which should preserve accuracy on the other classes better.
  - MixUp augmentation: blends pairs of training images (and their score/
    label targets) each batch. This is a general accuracy booster, and a
    good conceptual fit for this dataset's specific weak spot -- the fuzzy
    boundary between "clean" and "mildly degraded" is exactly the kind of
    soft decision boundary MixUp is known to help with.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import CLASS_TO_IDX, QualityDataset
from model import DISTORTION_CLASSES, build_model, save_checkpoint


def build_oversampling_sampler(train_ds) -> WeightedRandomSampler:
    counts = train_ds.df["distortion_type"].value_counts()
    class_weight = {c: 1.0 / counts.get(c, 1) for c in DISTORTION_CLASSES}
    sample_weights = train_ds.df["distortion_type"].map(class_weight).values
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def mixup_batch(images, feats, scores, labels, n_classes, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[perm]
    mixed_feats = lam * feats + (1 - lam) * feats[perm]
    mixed_scores = lam * scores + (1 - lam) * scores[perm]
    labels_a, labels_b = labels, labels[perm]
    return mixed_images, mixed_feats, mixed_scores, labels_a, labels_b, lam


def run_epoch(model, loader, device, optimizer=None, scheduler=None, use_mixup=False,
              score_weight=1.0, class_weight=1.0):
    training = optimizer is not None
    model.train(training)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    total_loss, total_correct, n = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        feats = batch["features"].to(device)
        scores = batch["quality_score"].to(device)
        labels = batch["distortion_label"].to(device)

        with torch.set_grad_enabled(training):
            if training and use_mixup:
                images, feats, scores, labels_a, labels_b, lam = mixup_batch(
                    images, feats, scores, labels, len(DISTORTION_CLASSES)
                )
                pred_score, logits = model(images, feats)
                cls_loss = lam * ce(logits, labels_a) + (1 - lam) * ce(logits, labels_b)
                loss = score_weight * mse(pred_score, scores) + class_weight * cls_loss
                correct_labels = labels_a  # for the accuracy metric, score against the dominant label
            else:
                pred_score, logits = model(images, feats)
                loss = score_weight * mse(pred_score, scores) + class_weight * ce(logits, labels)
                correct_labels = labels

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == correct_labels).sum().item()
        n += bs

    return total_loss / n, total_correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../data")
    ap.add_argument("--backbone", type=str, default="convnext_tiny",
                     choices=["resnet18", "resnet34", "convnext_tiny"])
    ap.add_argument("--epochs_frozen", type=int, default=4)
    ap.add_argument("--epochs_finetune", type=int, default=14)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_finetune", type=float, default=1.5e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--mixup", action="store_true", default=True)
    ap.add_argument("--no_mixup", dest="mixup", action="store_false")
    ap.add_argument("--out", type=str, default="../models/hybrid_model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, backbone: {args.backbone}, mixup: {args.mixup}")

    data_root = Path(args.data_root).resolve()
    train_ds = QualityDataset(data_root, "train", augment=True)
    val_ds = QualityDataset(data_root, "val", augment=False)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    sampler = build_oversampling_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(pretrained=True, backbone_name=args.backbone).to(device)

    history = []
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # Stage 1: freeze backbone, train fusion heads only (no mixup here --
    # let the new heads see clean signal first while they're still random)
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr_head
    )
    for epoch in range(args.epochs_frozen):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, use_mixup=False)
        val_loss, val_acc = run_epoch(model, val_loader, device, None)
        dt = time.time() - t0
        print(f"[frozen {epoch+1}/{args.epochs_frozen}] train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} ({dt:.1f}s)")
        history.append({"stage": "frozen", "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, args.backbone, out_path)

    # Stage 2: unfreeze, fine-tune end-to-end with cosine LR annealing + mixup
    for p in model.backbone.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
    total_steps = args.epochs_finetune * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    for epoch in range(args.epochs_finetune):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, scheduler, use_mixup=args.mixup)
        val_loss, val_acc = run_epoch(model, val_loader, device, None)
        dt = time.time() - t0
        print(f"[finetune {epoch+1}/{args.epochs_finetune}] train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} lr={scheduler.get_last_lr()[0]:.2e} ({dt:.1f}s)")
        history.append({"stage": "finetune", "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, args.backbone, out_path)

    with open(out_path.parent / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val_loss={best_val_loss:.3f}. Model saved to {out_path}")


if __name__ == "__main__":
    main()
