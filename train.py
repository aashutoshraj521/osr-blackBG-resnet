#!/usr/bin/env python3
"""
Training Pipeline for Open Set Recognition & Object Matching
============================================================
Trains metric-learning vision models on extracted black-background objects.
Optimizes hyperspherical prototype alignment for robust known-class matching
and accurate open-set unknown rejection.

Outputs saved to checkpoint directory:
- best_model.pth: Model state dict
- prototypes.pt: High-fidelity empirical class prototypes
- class_to_idx.json / idx_to_class.json: Class label mappings
- config.json: Model hyperparameters and calibrated open-set thresholds
- training_history.json: Per-epoch loss and accuracy metrics

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import BlackBackgroundTransforms, OSRImageDataset, create_osr_dataloaders
from models import OSRObjectMatcherModel, build_model


def calibrate_thresholds(
    model: OSRObjectMatcherModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Calibrate Open-Set decision thresholds based on known validation samples.
    Computes score percentiles so that known acceptance rate can be precisely controlled:
    - tau_95: Accepts 95% of known objects (5th percentile)
    - tau_99: Accepts 99% of known objects (1st percentile)
    - tau_mean: Mean known score
    - tau_recommended: Conservative threshold balancing acceptance and rejection
    """
    model.eval()
    all_scores: List[float] = []

    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Only use known validation samples
            mask = labels >= 0
            if not mask.any():
                continue

            valid_imgs = images[mask]
            match_res = model.match_object(valid_imgs, threshold=0.0)
            scores = match_res["confidence_score"].cpu().numpy().tolist()
            all_scores.extend(scores)

    if not all_scores:
        return {"tau_95": 0.5, "tau_99": 0.4, "tau_recommended": 0.5}

    arr = np.array(all_scores)
    tau_99 = float(np.percentile(arr, 1.0))
    tau_95 = float(np.percentile(arr, 5.0))
    tau_90 = float(np.percentile(arr, 10.0))
    tau_mean = float(np.mean(arr))
    tau_std = float(np.std(arr))

    # Recommended threshold: clamp(tau_95, 0.40, 0.85)
    rec = max(0.35, min(0.85, tau_95))

    return {
        "tau_99": round(tau_99, 4),
        "tau_95": round(tau_95, 4),
        "tau_90": round(tau_90, 4),
        "tau_mean": round(tau_mean, 4),
        "tau_std": round(tau_std, 4),
        "tau_recommended": round(rec, 4),
    }


def train_one_epoch(
    model: OSRObjectMatcherModel,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch over known classes."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels, _, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # In training, labels are guaranteed to be >= 0 (known classes)
        optimizer.zero_grad()
        logits, cosine_sim, _ = model(images, targets=labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def evaluate_known_val(
    model: OSRObjectMatcherModel,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate closed-set loss and accuracy on known validation samples."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            mask = labels >= 0
            if not mask.any():
                continue

            valid_imgs = images[mask]
            valid_labels = labels[mask]

            logits, _, _ = model(valid_imgs)
            loss = criterion(logits, valid_labels)

            total_loss += loss.item() * valid_imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == valid_labels).sum().item()
            total_samples += valid_imgs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def train_model(
    data_dir: str,
    output_dir: str = "./checkpoints",
    backbone: str = "resnet18",
    feature_dim: int = 256,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    backbone_lr: Optional[float] = None,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    margin: float = 0.15,
    label_smoothing: float = 0.05,
    dropout: float = 0.2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    known_classes: Optional[List[str]] = None,
    num_known_classes: Optional[int] = None,
    known_ratio: float = 0.8,
    target_size: Tuple[int, int] = (224, 224),
    device_name: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Full training routine for the Open Set Recognition & Object Matching system.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Resolve Device
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n[Train] Starting training run on device: {device}")
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. Setup DataLoaders
    train_loader, val_loader, test_loader, class_to_idx, known_cls, unknown_cls = create_osr_dataloaders(
        data_dir=data_dir,
        known_classes=known_classes,
        num_known=num_known_classes,
        known_ratio=known_ratio,
        batch_size=batch_size,
        target_size=target_size,
        seed=seed,
    )

    num_classes = len(known_cls)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(f"[Train] Known classes ({num_classes}): {known_cls}")
    print(f"[Train] Unknown (Open-Set) classes ({len(unknown_cls)}): {unknown_cls}")
    print(f"[Train] Samples: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # 3. Build Model
    model = build_model(
        num_classes=num_classes,
        backbone=backbone,
        feature_dim=feature_dim,
        temperature=temperature,
        pretrained=pretrained,
        margin=margin,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Differential learning rate or frozen backbone
    if freeze_backbone and hasattr(model.backbone, "conv1"):
        for name, param in model.backbone.named_parameters():
            if not name.startswith("projection"):
                param.requires_grad = False
        trainable_params = list(model.backbone.projection.parameters()) + list(model.classifier.parameters())
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        print(f"[Train] Robust Setup: Pretrained {backbone.upper()} (FROZEN BACKBONE), Head LR={lr:g}, Margin={margin}, LabelSmoothing={label_smoothing}, Dropout={dropout}")
    elif pretrained and hasattr(model.backbone, "conv1"):
        bb_lr = backbone_lr if backbone_lr is not None else (lr * 0.1)
        backbone_params = []
        for name, param in model.backbone.named_parameters():
            if not name.startswith("projection"):
                backbone_params.append(param)
        head_params = list(model.backbone.projection.parameters()) + list(model.classifier.parameters())
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": bb_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
        ])
        print(f"[Train] Robust Setup: Pretrained {backbone.upper()}, Backbone LR={bb_lr:g}, Head LR={lr:g}, Margin={margin}, LabelSmoothing={label_smoothing}, Dropout={dropout}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[Train] Standard Setup: Backbone={backbone}, LR={lr:g}, Margin={margin}")

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n[Train] Commencing training across {epochs} epochs...")
    line = "─" * 70
    print(line)
    print(f" {'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^12} | {'Val Acc':^11} ")
    print(line)

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_known_val(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc * 100.0, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc * 100.0, 2))

        print(f" {epoch:^7d} | {tr_loss:^12.4f} | {tr_acc*100:^10.2f}% | {val_loss:^12.4f} | {val_acc*100:^10.2f}%")

        # Save Best Model Checkpoint based on validation loss and accuracy
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = max(best_val_acc, val_acc)
            best_epoch = epoch
            is_best = True
        elif val_acc > best_val_acc and val_loss <= best_val_loss * 1.15:
            best_val_acc = val_acc
            best_epoch = epoch
            is_best = True

        if is_best:
            torch.save(model.state_dict(), out_path / "best_model.pth")

    total_training_time = time.time() - t_start
    print(line)
    print(f"[Train] Training complete in {total_training_time:.1f}s. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%)")

    # 4. Load best model weights
    model.load_state_dict(torch.load(out_path / "best_model.pth", map_location=device))

    # 5. Compute Empirical Centroid Prototypes on training set using canonical eval transforms
    print("[Train] Computing empirical class centroid prototypes with canonical transforms...")
    eval_transform = BlackBackgroundTransforms(target_size=target_size, is_training=False)
    canonical_train_dataset = OSRImageDataset(
        train_loader.dataset.samples,
        class_to_idx=class_to_idx,
        transform=eval_transform,
    )
    canonical_train_loader = torch.utils.data.DataLoader(
        canonical_train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    prototypes = model.update_empirical_prototypes(canonical_train_loader, device=device)
    torch.save(prototypes.cpu(), out_path / "prototypes.pt")

    # 6. Calibrate Open-Set Decision Thresholds
    print("[Train] Calibrating Open-Set decision thresholds on validation set...")
    thresholds = calibrate_thresholds(model, val_loader, device=device)
    print(f"[Train] Calibrated Thresholds: {thresholds}")

    # 7. Save Mappings & Config
    with open(out_path / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    with open(out_path / "idx_to_class.json", "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)

    config_info = {
        "backbone": backbone,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "temperature": temperature,
        "margin": margin,
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
        "target_size": list(target_size),
        "known_classes": known_cls,
        "unknown_classes": unknown_cls,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc * 100.0, 2),
        "total_training_time_s": round(total_training_time, 2),
        "calibrated_thresholds": thresholds,
    }
    with open(out_path / "config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    with open(out_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] All artifacts successfully saved to: '{out_path}'")
    return config_info


def parse_args():
    parser = argparse.ArgumentParser(description="Train OSR & Object Matching Model on Black-Background Objects.")
    parser.add_argument("-d", "--data-dir", required=True, type=str, help="Directory containing preprocessed black-background dataset.")
    parser.add_argument("-o", "--output-dir", default="./checkpoints", type=str, help="Directory to save model checkpoints and prototypes.")
    parser.add_argument("--backbone", default="simple_cnn", choices=["simple_cnn", "resnet18", "resnet34", "resnet50", "mobilenet_v3"], help="Backbone architecture.")
    parser.add_argument("--feature-dim", type=int, default=256, help="Dimension of embedding space.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Learning rate for pretrained backbone (defaults to lr * 0.1).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature scaling factor.")
    parser.add_argument("--margin", type=float, default=0.15, help="CosFace additive angular margin on cosine similarity.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability in projection head.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing factor for CrossEntropyLoss.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights for backbone.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Do not load pretrained backbone weights.")
    parser.add_argument("--freeze-backbone", action="store_true", default=False, help="Freeze convolutional backbone layers to preserve ImageNet features.")
    parser.add_argument("--num-known", type=int, default=None, help="Number of classes to designate as known.")
    parser.add_argument("--known-classes", type=str, nargs="+", default=None, help="Explicit list of known class names.")
    parser.add_argument("--known-ratio", type=float, default=0.8, help="Fraction of classes designated as known if num-known is omitted.")
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224], metavar=("WIDTH", "HEIGHT"), help="Input image size.")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="Compute device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        margin=args.margin,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        known_classes=args.known_classes,
        num_known_classes=args.num_known,
        known_ratio=args.known_ratio,
        target_size=tuple(args.target_size),
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
