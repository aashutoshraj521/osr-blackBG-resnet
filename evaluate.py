#!/usr/bin/env python3
"""
Open Set Recognition & Object Matching Evaluation
=================================================
Academic evaluation benchmark for MTech Thesis OSR research.
Computes comprehensive Closed-Set and Open-Set performance metrics.

Metrics Computed:
1. Closed-Set Metrics:
   - Top-1 Accuracy
   - Macro Precision, Recall, and F1-Score
   - Per-Class Performance Breakdown
   - ASCII Confusion Matrix
2. Open-Set Recognition (OSR) Metrics:
   - AUROC (Area under ROC curve for Known vs. Unknown discrimination)
   - A-OSCR (Area under Open Set Classification Rate curve)
   - FPR@95%TPR (False Positive Rate on Unknowns at 95% True Positive Rate)
   - Open-Set (K+1) Macro F1-Score at calibrated threshold

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from dataset import create_osr_dataloaders
from models import OSRObjectMatcherModel, build_model


# ==============================================================================
# Pure NumPy Metric Implementations (Zero External Dependencies)
# ==============================================================================

def compute_binary_auroc(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic curve (AUROC)
    using the exact Mann-Whitney U statistic / Wilcoxon rank-sum test.

    Args:
        scores: 1D array of continuous confidence scores (higher = more likely positive/known).
        is_positive: 1D boolean array (True for Known, False for Unknown).

    Returns:
        AUROC in range [0.0, 1.0].
    """
    n_pos = np.sum(is_positive)
    n_neg = len(is_positive) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Degenerate case (no positives or no negatives)

    # Sort scores in ascending order
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Handle ties by assigning average rank
    sorted_scores = scores[order]
    unique_vals, inverse_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    tie_indices = np.where(counts > 1)[0]
    for tie_idx in tie_indices:
        mask = inverse_indices == tie_idx
        ranks[order[mask]] = np.mean(ranks[order[mask]])

    # Sum of ranks for positive class
    rank_sum_pos = np.sum(ranks[is_positive])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)
    return float(np.clip(auroc, 0.0, 1.0))


def compute_oscr(
    known_preds: np.ndarray,
    known_targets: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    num_thresholds: int = 200,
) -> Tuple[float, float]:
    """
    Compute Open Set Classification Rate (OSCR) curve and Area under OSCR (A-OSCR).
    - Correct Classification Rate (CCR): Fraction of knowns correctly classified AND score >= theta.
    - False Positive Rate (FPR): Fraction of unknowns with score >= theta.

    Returns:
        (A-OSCR, fpr_at_95_tpr)
    """
    n_known = len(known_targets)
    n_unknown = len(unknown_scores)

    if n_known == 0 or n_unknown == 0:
        return 0.0, 1.0

    all_scores = np.concatenate([known_scores, unknown_scores])
    thresholds = np.linspace(all_scores.min() - 1e-4, all_scores.max() + 1e-4, num_thresholds)
    thresholds = np.sort(thresholds)[::-1]  # High threshold to low threshold

    ccr_list = []
    fpr_list = []

    is_correct = known_preds == known_targets

    for th in thresholds:
        # Correctly classified knowns with score >= th
        ccr = np.sum(is_correct & (known_scores >= th)) / n_known
        # Unknowns incorrectly accepted with score >= th
        fpr = np.sum(unknown_scores >= th) / n_unknown
        ccr_list.append(ccr)
        fpr_list.append(fpr)

    ccr_arr = np.array(ccr_list)
    fpr_arr = np.array(fpr_list)

    # Sort by FPR for trapezoidal integration
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    ccr_sorted = ccr_arr[sort_idx]

    # Integrate A-OSCR using trapezoidal rule
    a_oscr = float(np.trapezoid(ccr_sorted, fpr_sorted))

    # Calculate FPR at 95% TPR on knowns
    # TPR on knowns: fraction of knowns with score >= th
    tpr_arr = np.array([np.sum(known_scores >= th) / n_known for th in thresholds])
    idx_95 = np.where(tpr_arr >= 0.95)[0]
    if len(idx_95) > 0:
        th_95 = thresholds[idx_95[0]]
        fpr_at_95_tpr = float(np.sum(unknown_scores >= th_95) / n_unknown)
    else:
        fpr_at_95_tpr = 1.0

    return a_oscr, fpr_at_95_tpr


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix for known classes [0, K-1]."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


# ==============================================================================
# Model Evaluator
# ==============================================================================

class OSREvaluator:
    """
    Evaluates an OSRObjectMatcherModel on an OSR test dataset.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        device_name: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()

        # Load configs and mappings
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        with open(self.checkpoint_dir / "class_to_idx.json", "r") as f:
            self.class_to_idx = json.load(f)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        # Resolve device
        if device_name:
            self.device = torch.device(device_name)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Build & Load model
        self.model = build_model(
            num_classes=self.num_classes,
            backbone=self.config["backbone"],
            feature_dim=self.config["feature_dim"],
            temperature=self.config["temperature"],
        ).to(self.device)

        model_weights = self.checkpoint_dir / "best_model.pth"
        self.model.load_state_dict(torch.load(model_weights, map_location=self.device))

        # Load empirical prototypes if available
        proto_path = self.checkpoint_dir / "prototypes.pt"
        if proto_path.exists():
            prototypes = torch.load(proto_path, map_location=self.device)
            self.model.empirical_prototypes.copy_(prototypes)
            self.model.has_empirical_prototypes.copy_(torch.tensor(True))

        self.model.eval()

    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Evaluate full test set (knowns + unknowns).
        """
        if threshold is None:
            # Use recommended calibrated threshold from config
            threshold = self.config.get("calibrated_thresholds", {}).get("tau_recommended", 0.5)

        known_targets: List[int] = []
        known_preds: List[int] = []
        known_scores: List[float] = []

        unknown_scores: List[float] = []
        all_scores: List[float] = []
        all_is_known: List[bool] = []

        # For (K+1) open set F1
        all_true_labels: List[int] = []
        all_pred_labels: List[int] = []

        with torch.no_grad():
            for images, labels, _, _ in test_loader:
                images = images.to(self.device)
                match_res = self.model.match_object(images, threshold=threshold)

                scores = match_res["confidence_score"].cpu().numpy()
                preds = match_res["predicted_class_idx"].cpu().numpy()
                labels_np = labels.numpy()

                for sc, pred, target in zip(scores, preds, labels_np):
                    is_pos = (target >= 0)
                    all_scores.append(float(sc))
                    all_is_known.append(is_pos)

                    # Map target and pred for (K+1) class matrix (Unknown = num_classes)
                    mapped_target = target if is_pos else self.num_classes
                    mapped_pred = pred if pred >= 0 else self.num_classes
                    all_true_labels.append(mapped_target)
                    all_pred_labels.append(mapped_pred)

                    if is_pos:
                        # Extract unthresholded best class index for closed-set evaluation
                        top_class = int(np.argmax(match_res["all_scores"][len(known_targets) % len(scores)].cpu().numpy()))
                        known_targets.append(int(target))
                        known_preds.append(top_class)
                        known_scores.append(float(sc))
                    else:
                        unknown_scores.append(float(sc))

        scores_arr = np.array(all_scores)
        is_known_arr = np.array(all_is_known)
        known_targets_arr = np.array(known_targets)
        known_preds_arr = np.array(known_preds)
        known_scores_arr = np.array(known_scores)
        unknown_scores_arr = np.array(unknown_scores)

        # 1. Closed-set metrics on Known samples
        if len(known_targets_arr) > 0:
            closed_set_acc = float(np.mean(known_preds_arr == known_targets_arr) * 100.0)
            cm = compute_confusion_matrix(known_targets, known_preds, self.num_classes)
            # Per-class metrics
            per_class = {}
            precisions, recalls, f1s = [], [], []
            for k in range(self.num_classes):
                cname = self.idx_to_class[k]
                tp = cm[k, k]
                fp = np.sum(cm[:, k]) - tp
                fn = np.sum(cm[k, :]) - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                per_class[cname] = {
                    "precision": round(prec * 100.0, 2),
                    "recall": round(rec * 100.0, 2),
                    "f1": round(f1 * 100.0, 2),
                    "support": int(np.sum(cm[k, :])),
                }
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)

            closed_macro_f1 = float(np.mean(f1s) * 100.0)
            closed_macro_prec = float(np.mean(precisions) * 100.0)
            closed_macro_rec = float(np.mean(recalls) * 100.0)
        else:
            closed_set_acc = 0.0
            closed_macro_f1 = 0.0
            closed_macro_prec = 0.0
            closed_macro_rec = 0.0
            per_class = {}
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

        # 2. Open-set metrics
        auroc = compute_binary_auroc(scores_arr, is_known_arr) * 100.0
        a_oscr, fpr_at_95 = compute_oscr(
            known_preds_arr, known_targets_arr, known_scores_arr, unknown_scores_arr
        )
        a_oscr *= 100.0
        fpr_at_95 *= 100.0

        # 3. Open-set (K+1) Macro F1 at decision threshold
        # Labels 0..K-1 are known classes, class K is Unknown
        total_eval_classes = self.num_classes + 1
        cm_k1 = np.zeros((total_eval_classes, total_eval_classes), dtype=int)
        for t, p in zip(all_true_labels, all_pred_labels):
            cm_k1[t, p] += 1

        k1_f1s = []
        for k in range(total_eval_classes):
            tp = cm_k1[k, k]
            fp = np.sum(cm_k1[:, k]) - tp
            fn = np.sum(cm_k1[k, :]) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            k1_f1s.append(f1)
        openset_macro_f1 = float(np.mean(k1_f1s) * 100.0)

        report = {
            "num_known_samples": len(known_targets),
            "num_unknown_samples": len(unknown_scores),
            "total_test_samples": len(all_scores),
            "threshold_used": round(threshold, 4),
            "closed_set": {
                "top1_accuracy": round(closed_set_acc, 2),
                "macro_f1": round(closed_macro_f1, 2),
                "macro_precision": round(closed_macro_prec, 2),
                "macro_recall": round(closed_macro_rec, 2),
                "per_class": per_class,
                "confusion_matrix": cm.tolist(),
            },
            "open_set": {
                "auroc": round(auroc, 2),
                "a_oscr": round(a_oscr, 2),
                "fpr_at_95_tpr": round(fpr_at_95, 2),
                "openset_k_plus_1_macro_f1": round(openset_macro_f1, 2),
                "mean_known_score": round(float(np.mean(known_scores_arr)), 4) if len(known_scores_arr) > 0 else 0.0,
                "mean_unknown_score": round(float(np.mean(unknown_scores_arr)), 4) if len(unknown_scores_arr) > 0 else 0.0,
            },
        }

        self.print_report(report)
        return report

    def print_report(self, r: Dict) -> None:
        """Print academic evaluation report."""
        line = "═" * 70
        subline = "─" * 70

        print(f"\n{line}")
        print(f"  ACADEMIC EVALUATION REPORT: OPEN SET RECOGNITION (OSR)")
        print(f"{line}")
        print(f"  Model Backbone    : {self.config.get('backbone')}")
        print(f"  Feature Dimension : {self.config.get('feature_dim')}")
        print(f"  Known Classes (K) : {self.num_classes} ({', '.join(list(self.class_to_idx.keys()))})")
        print(f"  Test Distribution : {r['num_known_samples']} Known Samples | {r['num_unknown_samples']} Unknown (Open-Set) Samples")
        print(f"  Decision Threshold: {r['threshold_used']}")
        print(f"{subline}")
        print(f"  1. CLOSED-SET RECOGNITION (Known Test Classes):")
        print(f"     • Top-1 Accuracy   : {r['closed_set']['top1_accuracy']:6.2f}%")
        print(f"     • Macro F1-Score   : {r['closed_set']['macro_f1']:6.2f}%")
        print(f"     • Macro Precision  : {r['closed_set']['macro_precision']:6.2f}%")
        print(f"     • Macro Recall     : {r['closed_set']['macro_recall']:6.2f}%")
        print(f"{subline}")
        print(f"  2. OPEN-SET RECOGNITION & OUT-OF-DISTRIBUTION DISCRIMINATION:")
        print(f"     • AUROC (Known vs Unknown) : {r['open_set']['auroc']:6.2f}%  (Gold standard threshold-free metric)")
        print(f"     • A-OSCR (Area Under OSCR) : {r['open_set']['a_oscr']:6.2f}%  (Correct classification vs False Positive)")
        print(f"     • FPR @ 95% TPR            : {r['open_set']['fpr_at_95_tpr']:6.2f}%  (False alarms at 95% acceptance)")
        print(f"     • Open-Set (K+1) Macro F1  : {r['open_set']['openset_k_plus_1_macro_f1']:6.2f}%")
        print(f"     • Mean Known Score         : {r['open_set']['mean_known_score']:6.4f}")
        print(f"     • Mean Unknown Score       : {r['open_set']['mean_unknown_score']:6.4f}")
        print(f"{subline}")

        print("  3. PER-CLASS PERFORMANCE BREAKDOWN (Known):")
        print(f"     {'Class Name':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print(f"     {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        for cname, metrics in r["closed_set"]["per_class"].items():
            print(f"     {cname:<20} | {metrics['precision']:^9.1f}% | {metrics['recall']:^9.1f}% | {metrics['f1']:^9.1f}% | {metrics['support']:^8d}")
        print(f"{line}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OSR & Object Matching Model on Test Set.")
    parser.add_argument("-d", "--data-dir", required=True, type=str, help="Directory containing preprocessed black-background dataset.")
    parser.add_argument("-c", "--checkpoint-dir", required=True, type=str, help="Directory containing model checkpoints (best_model.pth, config.json).")
    parser.add_argument("-t", "--threshold", type=float, default=None, help="OSR decision threshold (default: use calibrated value).")
    parser.add_argument("-o", "--output-report", type=str, default=None, help="Path to save evaluation_report.json.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="Inference device.")
    return parser.parse_args()


def main():
    args = parse_args()

    evaluator = OSREvaluator(checkpoint_dir=args.checkpoint_dir, device_name=args.device)

    # Reconstruct test loader using saved known/unknown splits
    known_classes = evaluator.config["known_classes"]
    target_size = tuple(evaluator.config.get("target_size", [224, 224]))

    _, _, test_loader, _, _, _ = create_osr_dataloaders(
        data_dir=args.data_dir,
        known_classes=known_classes,
        batch_size=args.batch_size,
        target_size=target_size,
    )

    report = evaluator.evaluate(test_loader=test_loader, threshold=args.threshold)

    out_json = args.output_report or str(Path(args.checkpoint_dir) / "evaluation_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✔ Evaluation report written to: {out_json}")


if __name__ == "__main__":
    main()
