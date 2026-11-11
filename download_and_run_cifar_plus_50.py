#!/usr/bin/env python3
"""
CIFAR+50 Benchmark Runner: Vaze et al. (CVPR 2022) vs. Thesis Black-Background OSR
==================================================================================
Extracts and builds the official CIFAR+50 benchmark from Neal et al. / Vaze et al.
- Known Classes (4): airplane, automobile, bird, cat (from CIFAR-10)
- Unknown Classes (50): 50 non-overlapping classes from CIFAR-100 (Split 0)
- High-throughput neural background removal using RMBG-1.4 (384x384 fast inference)
- Runs 2x2 Factorial Study:
  1. Paper ResNet-18 (Linear + MLS) on Raw Background
  2. Paper ResNet-18 (Linear + MLS) on Black Background
  3. Thesis ResNet-18 (Cosine Prototypes) on Raw Background
  4. Thesis ResNet-18 (Cosine Prototypes) on Black Background

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image
import torch

from bg_removal import BackgroundRemover, PipelineConfig
from dataset import create_osr_dataloaders
from evaluate import OSREvaluator
from train import train_model

KNOWN_4 = ["airplane", "automobile", "bird", "cat"]


def get_cifar_plus_50_classes() -> List[str]:
    """Returns the official 50 unknown class names for Split 0 from Vaze et al."""
    sys.path.append("/home/aashu/Thesis/osr_closed_set_all_you_need")
    from data.open_set_splits.osr_splits import osr_splits

    with tarfile.open("data/cifar100.tgz") as tar:
        names = tar.getnames()
        all_classes = sorted(list(set([p.split("/")[-2] for p in names if p.endswith(".png")])))

    split_50_idxs = osr_splits["cifar-10-100-50"][0]
    unknown_50 = [all_classes[i] for i in split_50_idxs]
    return unknown_50


def build_cifar_plus_50_dataset(
    raw_dir: Path,
    black_dir: Path,
    unknown_classes: List[str],
    samples_per_class: int = 5,
    target_size: Tuple[int, int] = (224, 224),
):
    """Extract and preprocess CIFAR+50 raw and black background datasets."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    black_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy known classes from cifar_plus_10 if available
    cifar10_raw = Path("data/cifar_plus_10/raw")
    cifar10_black = Path("data/cifar_plus_10/black")

    print("[CIFAR+50 Prep] Populating known classes...")
    for kc in KNOWN_4:
        dst_raw = raw_dir / kc
        dst_black = black_dir / kc
        if not dst_raw.exists() and (cifar10_raw / kc).exists():
            shutil.copytree(cifar10_raw / kc, dst_raw)
        if not dst_black.exists() and (cifar10_black / kc).exists():
            shutil.copytree(cifar10_black / kc, dst_black)

    # 2. Extract remaining unknown classes from cifar100.tgz
    print(f"[CIFAR+50 Prep] Extracting and segmenting {len(unknown_classes)} unknown classes...")
    cfg = PipelineConfig(device="cpu", input_size=(384, 384))
    remover = BackgroundRemover(cfg)

    tar_path = Path("data/cifar100.tgz")
    class_counts: Dict[str, int] = {c: 0 for c in unknown_classes}
    target_set: Set[str] = set(unknown_classes)

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if all(class_counts[c] >= samples_per_class for c in unknown_classes):
                break

            if not member.name.endswith(".png"):
                continue

            cls_name = member.name.split("/")[-2]
            if cls_name not in target_set:
                continue

            if class_counts[cls_name] >= samples_per_class:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            sample_idx = class_counts[cls_name]
            class_counts[cls_name] += 1

            raw_bytes = f.read()
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            img_resized = img.resize(target_size, Image.Resampling.BICUBIC)

            # Save raw
            c_raw_dir = raw_dir / cls_name
            c_raw_dir.mkdir(parents=True, exist_ok=True)
            img_resized.save(c_raw_dir / f"{cls_name}_{sample_idx:03d}.png")

            # Segment and save black background
            c_black_dir = black_dir / cls_name
            c_black_dir.mkdir(parents=True, exist_ok=True)
            black_img, alpha, _ = remover.remove_background(img_resized, background_color=(0, 0, 0))

            # Bounding box tight crop & square center
            bbox = alpha.getbbox()
            if bbox:
                cw = bbox[2] - bbox[0]
                ch = bbox[3] - bbox[1]
                side = max(cw, ch)
                sq = Image.new("RGB", (side, side), (0, 0, 0))
                sq.paste(black_img.crop(bbox), ((side - cw) // 2, (side - ch) // 2))
                final_black = sq.resize(target_size, Image.Resampling.BILINEAR)
            else:
                final_black = black_img

            final_black.save(c_black_dir / f"{cls_name}_{sample_idx:03d}.png")

    print("[CIFAR+50 Prep] Dataset construction complete!")


def run_cifar_plus_50_benchmark(
    epochs: int = 12,
    batch_size: int = 8,
    output_root: str = "checkpoints/cifar_plus_50_comparison",
    device: str = "cpu",
):
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    raw_dir = Path("data/cifar_plus_50/raw")
    black_dir = Path("data/cifar_plus_50/black")

    unknown_classes = get_cifar_plus_50_classes()

    if not (raw_dir.exists() and len(list(raw_dir.iterdir())) >= 54):
        build_cifar_plus_50_dataset(raw_dir, black_dir, unknown_classes, samples_per_class=5)
    else:
        print("[CIFAR+50] Using existing dataset directories.")

    experiments = [
        {
            "name": "Paper Baseline (ResNet-18 + MLS) on Raw CIFAR+50",
            "key": "paper_cifar50_raw_mls",
            "data_dir": str(raw_dir),
            "classifier_type": "linear",
            "scoring_method": "mls",
            "margin": 0.0,
            "label_smoothing": 0.05,
            "bg_condition": "Raw Background",
            "model_desc": "Paper MLS (Vaze et al., 2022)",
        },
        {
            "name": "Paper Baseline (ResNet-18 + MLS) on Black CIFAR+50",
            "key": "paper_cifar50_black_mls",
            "data_dir": str(black_dir),
            "classifier_type": "linear",
            "scoring_method": "mls",
            "margin": 0.0,
            "label_smoothing": 0.05,
            "bg_condition": "Black Background",
            "model_desc": "Paper MLS (Vaze et al., 2022)",
        },
        {
            "name": "Thesis Prototype Model (ResNet-18) on Raw CIFAR+50",
            "key": "thesis_cifar50_raw_prototype",
            "data_dir": str(raw_dir),
            "classifier_type": "cosine",
            "scoring_method": "cosine",
            "margin": 0.15,
            "label_smoothing": 0.05,
            "bg_condition": "Raw Background",
            "model_desc": "Thesis Prototype Matcher",
        },
        {
            "name": "Thesis Prototype Model (ResNet-18) on Black CIFAR+50",
            "key": "thesis_cifar50_black_prototype",
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
    print(" 🚀 COMMENCING CIFAR+50 BENCHMARK: VAZE ET AL. (CVPR 2022) vs THESIS")
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
            known_classes=KNOWN_4,
            target_size=(224, 224),
            device_name=device,
            seed=42,
        )

        # Evaluate model
        evaluator = OSREvaluator(checkpoint_dir=str(exp_dir), device_name=device)
        _, _, test_loader, _, _, _ = create_osr_dataloaders(
            data_dir=exp["data_dir"],
            known_classes=KNOWN_4,
            batch_size=batch_size,
            target_size=(224, 224),
            seed=42,
        )

        report = evaluator.evaluate(test_loader=test_loader, scoring_method=exp["scoring_method"])

        summary_entry = {
            "experiment": exp["name"],
            "key": exp["key"],
            "dataset": "CIFAR+50",
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
    print(f" {'CIFAR+50 COMPARATIVE BENCHMARK: VAZE ET AL. (CVPR 2022) vs THESIS':^96}")
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_cifar_plus_50_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
