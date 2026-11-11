#!/usr/bin/env python3
"""
SVHN Benchmark Runner: Vaze et al. (CVPR 2022) vs. Thesis Black-Background OSR
=============================================================================
1. Prepares SVHN dataset into Raw and Black-Background Twin Datasets.
2. Official Split 0 from Vaze et al. / Neal et al.:
   - Known Classes (6): digit_2, digit_3, digit_5, digit_6, digit_7, digit_8
   - Unknown Classes (4): digit_0, digit_1, digit_4, digit_9
3. Executes 2x2 Factorial Study:
   - Paper MLS on Raw SVHN
   - Paper MLS on Black SVHN
   - Thesis Prototypes on Raw SVHN
   - Thesis Prototypes on Black SVHN
4. Computes Closed-Set Accuracy, AUROC, OSCR, FPR@95%TPR, and (K+1) Macro F1.

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision

from dataset import create_osr_dataloaders
from evaluate import OSREvaluator
from train import train_model


# Official Neal et al. / Vaze et al. Split 0 for SVHN
# Known: [5, 3, 7, 2, 8, 6], Unknown: [0, 1, 4, 9]
SVHN_KNOWN = ["digit_2", "digit_3", "digit_5", "digit_6", "digit_7", "digit_8"]
SVHN_UNKNOWN = ["digit_0", "digit_1", "digit_4", "digit_9"]


def extract_svhn_data(
    raw_dir: Path,
    black_dir: Path,
    samples_per_class: int = 15,
    target_size: Tuple[int, int] = (224, 224),
):
    """Extract SVHN samples and create raw + black-background twin directories."""
    print("\n[SVHN Prep] Loading SVHN dataset...")
    svhn_ds = torchvision.datasets.SVHN(root="data/svhn_raw", split="train", download=True)

    raw_dir.mkdir(parents=True, exist_ok=True)
    black_dir.mkdir(parents=True, exist_ok=True)

    class_counts = {f"digit_{i}": 0 for i in range(10)}
    total_needed = 10 * samples_per_class

    print(f"[SVHN Prep] Extracting {samples_per_class} samples per digit class ({total_needed} total)...")

    for idx in range(len(svhn_ds)):
        if all(count >= samples_per_class for count in class_counts.values()):
            break

        img, label = svhn_ds[idx]
        cls_name = f"digit_{label}"

        if class_counts[cls_name] >= samples_per_class:
            continue

        sample_idx = class_counts[cls_name]
        class_counts[cls_name] += 1

        # 1. Save Raw Image (upscaled with bicubic to target_size)
        cls_raw_dir = raw_dir / cls_name
        cls_raw_dir.mkdir(parents=True, exist_ok=True)
        raw_img_path = cls_raw_dir / f"{cls_name}_{sample_idx:03d}.png"
        img_resized = img.resize(target_size, Image.Resampling.BICUBIC)
        img_resized.save(raw_img_path)

        # 2. Extract Digit Stroke onto Pure Black Background (RGB 0,0,0)
        gray = np.array(img_resized.convert("L"))
        h, w = gray.shape
        center_region = gray[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        center_mean = np.mean(center_region)
        border_mean = (
            np.mean(gray[:int(h * 0.15), :]) +
            np.mean(gray[-int(h * 0.15):, :]) +
            np.mean(gray[:, :int(w * 0.15)]) +
            np.mean(gray[:, -int(w * 0.15):])
        ) / 4.0

        # Contrast mask for digit vs background street plate/wall
        if center_mean < border_mean:
            # Dark digit on lighter plate
            mask = gray < (border_mean - 18)
        else:
            # Light digit on darker background
            mask = gray > (border_mean + 18)

        # Apply soft boundary feathering and black composite
        rgb_arr = np.array(img_resized, dtype=np.float32)
        alpha = mask.astype(np.float32)[:, :, None]
        black_rgb = (rgb_arr * alpha).astype(np.uint8)
        black_img = Image.fromarray(black_rgb)

        cls_black_dir = black_dir / cls_name
        cls_black_dir.mkdir(parents=True, exist_ok=True)
        black_img_path = cls_black_dir / f"{cls_name}_{sample_idx:03d}.png"
        black_img.save(black_img_path)

    print(f"[SVHN Prep] Successfully created raw images in '{raw_dir}' and black-background images in '{black_dir}'!")


def run_svhn_study(
    epochs: int = 12,
    batch_size: int = 8,
    samples_per_class: int = 15,
    device: str = "cpu",
    output_root: str = "checkpoints/svhn_comparison",
):
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    raw_dir = Path("data/svhn/raw")
    black_dir = Path("data/svhn/black")

    # Step 1: Prepare data if not already extracted
    if not (raw_dir.exists() and any(raw_dir.iterdir())):
        extract_svhn_data(raw_dir, black_dir, samples_per_class=samples_per_class)
    else:
        print(f"[SVHN] Using existing dataset directories: {raw_dir} and {black_dir}")

    # Step 2: Define 2x2 Factorial Matrix
    experiments = [
        {
            "name": "Paper Baseline (ResNet-18 + MLS) on Raw SVHN",
            "key": "paper_svhn_raw_mls",
            "data_dir": str(raw_dir),
            "classifier_type": "linear",
            "scoring_method": "mls",
            "margin": 0.0,
            "label_smoothing": 0.0,
            "bg_condition": "Raw Background",
            "model_desc": "Paper MLS (Vaze et al., 2022)",
        },
        {
            "name": "Paper Baseline (ResNet-18 + MLS) on Black SVHN",
            "key": "paper_svhn_black_mls",
            "data_dir": str(black_dir),
            "classifier_type": "linear",
            "scoring_method": "mls",
            "margin": 0.0,
            "label_smoothing": 0.0,
            "bg_condition": "Black Background",
            "model_desc": "Paper MLS (Vaze et al., 2022)",
        },
        {
            "name": "Thesis Prototype Model (ResNet-18) on Raw SVHN",
            "key": "thesis_svhn_raw_prototype",
            "data_dir": str(raw_dir),
            "classifier_type": "cosine",
            "scoring_method": "cosine",
            "margin": 0.15,
            "label_smoothing": 0.05,
            "bg_condition": "Raw Background",
            "model_desc": "Thesis Prototype Matcher",
        },
        {
            "name": "Thesis Prototype Model (ResNet-18) on Black SVHN",
            "key": "thesis_svhn_black_prototype",
            "data_dir": str(black_dir),
            "classifier_type": "cosine",
            "scoring_method": "cosine",
            "margin": 0.15,
            "label_smoothing": 0.05,
            "bg_condition": "Black Background",
            "model_desc": "Thesis Prototype Matcher",
        },
    ]

    all_results = []

    print("\n" + "=" * 80)
    print(" 🚀 COMMENCING SVHN BENCHMARK: VAZE ET AL. (CVPR 2022) vs THESIS")
    print("=" * 80)

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/4] >>> Training & Evaluating: {exp['name']}")
        exp_dir = out_root / exp["key"]

        # Train model
        train_model(
            data_dir=exp["data_dir"],
            output_dir=str(exp_dir),
            backbone="resnet18",
            classifier_type=exp["classifier_type"],
            scoring_method=exp["scoring_method"],
            feature_dim=256,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
            weight_decay=1e-4,
            temperature=0.1,
            margin=exp["margin"],
            label_smoothing=exp["label_smoothing"],
            dropout=0.2,
            pretrained=True,
            freeze_backbone=False,
            known_classes=SVHN_KNOWN,
            target_size=(224, 224),
            device_name=device,
            seed=42,
        )

        # Evaluate model on test split
        evaluator = OSREvaluator(checkpoint_dir=str(exp_dir), device_name=device)
        _, _, test_loader, _, _, _ = create_osr_dataloaders(
            data_dir=exp["data_dir"],
            known_classes=SVHN_KNOWN,
            batch_size=batch_size,
            target_size=(224, 224),
            seed=42,
        )

        report = evaluator.evaluate(test_loader=test_loader, scoring_method=exp["scoring_method"])

        summary_entry = {
            "experiment": exp["name"],
            "key": exp["key"],
            "dataset": "SVHN",
            "model_desc": exp["model_desc"],
            "bg_condition": exp["bg_condition"],
            "classifier_type": exp["classifier_type"],
            "scoring_method": exp["scoring_method"],
            "closed_set_acc": report["closed_set"]["top1_accuracy"],
            "auroc": report["open_set"]["auroc"],
            "a_oscr": report["open_set"]["a_oscr"],
            "fpr_at_95_tpr": report["open_set"]["fpr_at_95_tpr"],
            "macro_f1": report["open_set"]["openset_k_plus_1_macro_f1"],
            "mean_known_score": report["open_set"]["mean_known_score"],
            "mean_unknown_score": report["open_set"]["mean_unknown_score"],
            "threshold_used": report["threshold_used"],
        }
        all_results.append(summary_entry)

    # Save summary report
    report_path = out_root / "comparative_study_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print Comparative Table
    print("\n" + "=" * 98)
    print(f" {'SVHN COMPARATIVE BENCHMARK: VAZE ET AL. (CVPR 2022) vs THESIS':^96}")
    print("=" * 98)
    header = f"| {'Model Paradigm':<26} | {'Background':<16} | {'Acc (%)':<8} | {'AUROC (%)':<10} | {'OSCR (%)':<9} | {'FPR@95%':<8} | {'F1 (%)':<8} |"
    sep = "|-" + "-" * 26 + "-+-" + "-" * 16 + "-+-" + "-" * 8 + "-+-" + "-" * 10 + "-+-" + "-" * 9 + "-+-" + "-" * 8 + "-+-" + "-" * 8 + "-|"
    print(sep)
    print(header)
    print(sep)

    for r in all_results:
        row = f"| {r['model_desc']:<26} | {r['bg_condition']:<16} | {r['closed_set_acc']:^8.1f} | {r['auroc']:^10.2f} | {r['a_oscr']:^9.2f} | {r['fpr_at_95_tpr']:^8.1f} | {r['macro_f1']:^8.2f} |"
        print(row)
    print(sep)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_svhn_study(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples,
        device=args.device,
    )
