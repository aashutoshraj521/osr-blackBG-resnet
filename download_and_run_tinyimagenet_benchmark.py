#!/usr/bin/env python3
"""
TinyImageNet Benchmark Runner: Vaze et al. (CVPR 2022) vs. Thesis Black-Background OSR
======================================================================================
1. Extracts TinyImageNet-200 images using official class splits from Vaze et al. / Neal et al.
   - Known Classes: Official Split 0 known synsets from wnids.txt
   - Unknown Classes: Open-set unknown synsets
2. Converts raw images into black-background objects via RMBG-1.4 (384x384 fast inference).
3. Executes 2x2 Factorial Study:
   - Paper ResNet-18 (Linear + MLS) on Raw TinyImageNet
   - Paper ResNet-18 (Linear + MLS) on Black TinyImageNet
   - Thesis ResNet-18 (Cosine Prototypes) on Raw TinyImageNet
   - Thesis ResNet-18 (Cosine Prototypes) on Black TinyImageNet
4. Evaluates Closed-Set Accuracy, AUROC, OSCR, FPR@95%TPR, and (K+1) Macro F1.

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image
import torch

from bg_removal import BackgroundRemover, PipelineConfig
from dataset import create_osr_dataloaders
from evaluate import OSREvaluator
from train import train_model


def get_tinyimagenet_splits() -> Tuple[List[str], List[str]]:
    """Returns known and unknown class wnid names from Split 0 in Vaze et al."""
    sys.path.append("/home/aashu/Thesis/osr_closed_set_all_you_need")
    from data.open_set_splits.osr_splits import osr_splits

    with zipfile.ZipFile("data/tiny-imagenet-200.zip", "r") as zf:
        with zf.open("tiny-imagenet-200/wnids.txt") as f:
            all_wnids = [line.decode("utf-8").strip() for line in f if line.strip()]

    split_0_known_indices = osr_splits["tinyimagenet"][0]
    known_wnids = [all_wnids[i] for i in split_0_known_indices]
    unknown_wnids = [w for i, w in enumerate(all_wnids) if i not in set(split_0_known_indices)]

    return known_wnids, unknown_wnids


def prep_tinyimagenet_dataset(
    raw_dir: Path,
    black_dir: Path,
    known_classes: List[str],
    unknown_classes: List[str],
    samples_per_class: int = 5,
    target_size: Tuple[int, int] = (224, 224),
):
    """Extract and preprocess TinyImageNet into raw and black directories."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    black_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TinyImageNet Prep] Extracting and segmenting {len(known_classes)} known and {len(unknown_classes)} unknown classes...")
    cfg = PipelineConfig(device="cpu", input_size=(384, 384))
    remover = BackgroundRemover(cfg)

    target_classes = set(known_classes + unknown_classes)
    class_counts: Dict[str, int] = {c: 0 for c in target_classes}

    with zipfile.ZipFile("data/tiny-imagenet-200.zip", "r") as zf:
        for entry in zf.namelist():
            if all(class_counts[c] >= samples_per_class for c in target_classes):
                break

            if "/train/" not in entry or not entry.endswith(".JPEG"):
                continue

            parts = entry.split("/")
            wnid = parts[-3]

            if wnid not in target_classes:
                continue

            if class_counts[wnid] >= samples_per_class:
                continue

            sample_idx = class_counts[wnid]
            class_counts[wnid] += 1

            with zf.open(entry) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            img_resized = img.resize(target_size, Image.Resampling.BICUBIC)

            # Save raw image
            c_raw_dir = raw_dir / wnid
            c_raw_dir.mkdir(parents=True, exist_ok=True)
            img_resized.save(c_raw_dir / f"{wnid}_{sample_idx:03d}.png")

            # Segment foreground object onto black
            c_black_dir = black_dir / wnid
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

            final_black.save(c_black_dir / f"{wnid}_{sample_idx:03d}.png")

    print("[TinyImageNet Prep] Dataset construction complete!")


def run_tinyimagenet_benchmark(
    epochs: int = 10,
    batch_size: int = 8,
    output_root: str = "checkpoints/tinyimagenet_comparison",
    device: str = "cpu",
):
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    raw_dir = Path("data/tinyimagenet/raw")
    black_dir = Path("data/tinyimagenet/black")

    full_known, full_unknown = get_tinyimagenet_splits()
    # Select representative subset: 6 known classes, 20 unknown classes from Split 0
    selected_known = full_known[:6]
    selected_unknown = full_unknown[:20]

    if not (raw_dir.exists() and len(list(raw_dir.iterdir())) >= 26):
        prep_tinyimagenet_dataset(
            raw_dir=raw_dir,
            black_dir=black_dir,
            known_classes=selected_known,
            unknown_classes=selected_unknown,
            samples_per_class=5,
        )
    else:
        print("[TinyImageNet] Using existing dataset directories.")

    experiments = [
        {
            "name": "Paper Baseline (ResNet-18 + MLS) on Raw TinyImageNet",
            "key": "paper_tiny_raw_mls",
            "data_dir": str(raw_dir),
            "classifier_type": "linear",
            "scoring_method": "mls",
            "margin": 0.0,
            "label_smoothing": 0.9,
            "bg_condition": "Raw Background",
            "model_desc": "Paper MLS (Vaze et al., 2022)",
        },
        {
            "name": "Paper Baseline (ResNet-18 + MLS) on Black TinyImageNet",
            "key": "paper_tiny_black_mls",
            "data_dir": str(black_dir),
            "classifier_type": "linear",
            "scoring_method": "mls",
            "margin": 0.0,
            "label_smoothing": 0.9,
            "bg_condition": "Black Background",
            "model_desc": "Paper MLS (Vaze et al., 2022)",
        },
        {
            "name": "Thesis Prototype Model (ResNet-18) on Raw TinyImageNet",
            "key": "thesis_tiny_raw_prototype",
            "data_dir": str(raw_dir),
            "classifier_type": "cosine",
            "scoring_method": "cosine",
            "margin": 0.15,
            "label_smoothing": 0.05,
            "bg_condition": "Raw Background",
            "model_desc": "Thesis Prototype Matcher",
        },
        {
            "name": "Thesis Prototype Model (ResNet-18) on Black TinyImageNet",
            "key": "thesis_tiny_black_prototype",
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
    print(" 🚀 COMMENCING TINYIMAGENET BENCHMARK: VAZE ET AL. (CVPR 2022) vs THESIS")
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
            known_classes=selected_known,
            target_size=(224, 224),
            device_name=device,
            seed=42,
        )

        # Evaluate model
        evaluator = OSREvaluator(checkpoint_dir=str(exp_dir), device_name=device)
        _, _, test_loader, _, _, _ = create_osr_dataloaders(
            data_dir=exp["data_dir"],
            known_classes=selected_known,
            batch_size=batch_size,
            target_size=(224, 224),
            seed=42,
        )

        report = evaluator.evaluate(test_loader=test_loader, scoring_method=exp["scoring_method"])

        summary_entry = {
            "experiment": exp["name"],
            "key": exp["key"],
            "dataset": "TinyImageNet",
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
    print(f" {'TINYIMAGENET COMPARATIVE BENCHMARK: VAZE ET AL. (CVPR 2022) vs THESIS':^96}")
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

    run_tinyimagenet_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
