#!/usr/bin/env python3
"""
Vaze et al. (CVPR 2022) vs. Thesis Black-Background OSR Comparison Suite
========================================================================
Executes a rigorous 2x2 factorial study:
- Factor 1 (Architecture / Loss):
  * Paper Baseline: ResNet-18 + Linear Head + Cross-Entropy + Maximum Logit Score (MLS)
  * Thesis Method : ResNet-18 + Hyperspherical Cosine Head + Empirical Centroid Prototypes
- Factor 2 (Background Condition):
  * Raw natural background images (CIFAR+10 Raw)
  * Black background normalized objects (CIFAR+10 Black)

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from dataset import create_osr_dataloaders
from evaluate import OSREvaluator
from train import train_model


def run_benchmark(
    raw_data_dir: str = "data/cifar_plus_10/raw",
    black_data_dir: str = "data/cifar_plus_10/black",
    output_root: str = "checkpoints/comparison_study",
    backbone: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    known_classes: List[str] = ["airplane", "automobile", "bird", "cat"],
    device_name: str = "cpu",
    seed: int = 42,
) -> Dict:
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "Paper Baseline (ResNet-18 + MLS) on Raw Background",
            "key": "paper_raw_mls",
            "data_dir": raw_data_dir,
            "classifier_type": "linear",
            "scoring_method": "mls",
            "margin": 0.0,
            "label_smoothing": 0.05,
            "bg_condition": "Raw Background",
            "model_desc": "Paper MLS (Vaze et al., 2022)",
        },
        {
            "name": "Paper Baseline (ResNet-18 + MLS) on Black Background",
            "key": "paper_black_mls",
            "data_dir": black_data_dir,
            "classifier_type": "linear",
            "scoring_method": "mls",
            "margin": 0.0,
            "label_smoothing": 0.05,
            "bg_condition": "Black Background",
            "model_desc": "Paper MLS (Vaze et al., 2022)",
        },
        {
            "name": "Thesis Prototype Model (ResNet-18) on Raw Background",
            "key": "thesis_raw_prototype",
            "data_dir": raw_data_dir,
            "classifier_type": "cosine",
            "scoring_method": "cosine",
            "margin": 0.15,
            "label_smoothing": 0.05,
            "bg_condition": "Raw Background",
            "model_desc": "Thesis Prototype Matcher",
        },
        {
            "name": "Thesis Prototype Model (ResNet-18) on Black Background",
            "key": "thesis_black_prototype",
            "data_dir": black_data_dir,
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
    print(" 🚀 STARTING 2x2 OSR COMPARATIVE BENCHMARK: VAZE ET AL. (CVPR 2022) vs THESIS")
    print("=" * 80)

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/4] >>> Training & Evaluating: {exp['name']}")
        exp_dir = out_root / exp["key"]

        # Train model
        train_model(
            data_dir=exp["data_dir"],
            output_dir=str(exp_dir),
            backbone=backbone,
            classifier_type=exp["classifier_type"],
            scoring_method=exp["scoring_method"],
            feature_dim=256,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=1e-4,
            temperature=0.1,
            margin=exp["margin"],
            label_smoothing=exp["label_smoothing"],
            dropout=0.2,
            pretrained=True,
            freeze_backbone=False,
            known_classes=known_classes,
            target_size=(224, 224),
            device_name=device_name,
            seed=seed,
        )

        # Evaluate model
        evaluator = OSREvaluator(checkpoint_dir=str(exp_dir), device_name=device_name)
        _, _, test_loader, _, _, _ = create_osr_dataloaders(
            data_dir=exp["data_dir"],
            known_classes=known_classes,
            batch_size=batch_size,
            target_size=(224, 224),
            seed=seed,
        )

        report = evaluator.evaluate(test_loader=test_loader, scoring_method=exp["scoring_method"])

        # Record summary
        summary_entry = {
            "experiment": exp["name"],
            "key": exp["key"],
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

    # Save complete study report
    study_report_path = out_root / "comparative_study_report.json"
    with open(study_report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print Comparative Table
    print("\n" + "=" * 98)
    print(f" {'FINAL COMPARATIVE BENCHMARK RESULTS (Vaze et al. CVPR 2022 vs Thesis)':^96}")
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
    parser = argparse.ArgumentParser(description="Run Vaze et al. (CVPR 2022) vs Thesis Black Background Benchmark.")
    parser.add_argument("--raw-dir", default="data/cifar_plus_10/raw", help="Path to raw CIFAR+10 images.")
    parser.add_argument("--black-dir", default="data/cifar_plus_10/black", help="Path to black background CIFAR+10 images.")
    parser.add_argument("--output-dir", default="checkpoints/comparison_study", help="Path to output checkpoints and report.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Compute device.")
    args = parser.parse_args()

    run_benchmark(
        raw_data_dir=args.raw_dir,
        black_data_dir=args.black_dir,
        output_root=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_name=args.device,
    )
