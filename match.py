#!/usr/bin/env python3
"""
CLI Tool for End-to-End Object Matching
=======================================
Command-line interface to test query images:
1. Extracts the foreground object and turns the background black (0, 0, 0).
2. Matches the object embedding against trained known prototypes.
3. Decides Known class vs Open-Set rejection (Unknown).
4. Generates visual multi-panel diagnostic artifacts.

Usage:
    # Single image test
    python3 match.py -i query.jpg -c ./checkpoints/ --save-vis match_vis.png

    # Batch test a folder of query images
    python3 match.py -i ./test_queries/ -c ./checkpoints/ -o ./results/

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from matcher import ObjectMatcher

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Object Matching: Extract object -> Black background -> Match."
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str,
        help="Input query image file path OR directory of query images."
    )
    parser.add_argument(
        "-c", "--checkpoint-dir", required=True, type=str,
        help="Directory containing trained model checkpoints (best_model.pth, config.json)."
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=None,
        help="OSR acceptance threshold (default: uses calibrated tau_recommended)."
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=5,
        help="Number of top candidates to report (default: 5)."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./match_results",
        help="Output directory when processing a folder of query images."
    )
    parser.add_argument(
        "--save-vis", type=str, default=None,
        help="File path to save the 4-panel visual match diagnostic."
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Inference compute device."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results in JSON format."
    )
    return parser.parse_args()


def process_single(matcher: ObjectMatcher, img_path: Path, args) -> dict:
    result = matcher.match(img_path, top_k=args.top_k, threshold=args.threshold)

    vis_path = args.save_vis
    if not vis_path and not args.json:
        vis_path = str(img_path.parent / f"{img_path.stem}_match_report.png")

    if vis_path:
        matcher.create_visualization(result, save_path=vis_path)

    out_dict = {
        "file": str(img_path),
        "decision": "MATCH" if result.is_known else "UNKNOWN",
        "predicted_class": result.predicted_class,
        "confidence_pct": round(result.confidence_pct, 2),
        "threshold_pct": round(result.threshold * 100.0, 2),
        "latency_ms": round(result.latency_ms, 1),
        "top_candidates": [
            {
                "class_name": c.class_name,
                "confidence_pct": round(c.confidence_pct, 2),
                "cosine_similarity": round(c.similarity, 4),
            }
            for c in result.top_candidates
        ],
    }

    if args.json:
        print(json.dumps(out_dict, indent=2))
    else:
        line = "─" * 60
        print(f"\n{line}")
        print(f"  END-TO-END OBJECT MATCH RESULT: {img_path.name}")
        print(f"{line}")
        status = "✔ MATCH ACCEPTED" if result.is_known else "✘ REJECTED (OPEN-SET UNKNOWN)"
        print(f"  Decision        : {status}")
        print(f"  Predicted Class : {result.predicted_class}")
        print(f"  Confidence Score: {result.confidence_pct:5.1f}% (Threshold: {result.threshold*100:.1f}%)")
        print(f"  Total Latency   : {result.latency_ms:5.1f} ms")
        print(f"{line}")
        print("  TOP MATCH CANDIDATES:")
        for idx, c in enumerate(result.top_candidates, start=1):
            marker = "★" if (idx == 1 and result.is_known) else " "
            print(f"   {marker} {idx}. {c.class_name:<18} : {c.confidence_pct:5.1f}% (Cosine: {c.similarity:+.3f})")
        print(f"{line}\n")

    return out_dict


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    matcher = ObjectMatcher(
        checkpoint_dir=args.checkpoint_dir,
        threshold=args.threshold,
        device_name=args.device,
    )

    if input_path.is_file():
        process_single(matcher, input_path, args)
    else:
        # Directory mode
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
        files.sort()

        print(f"[Match CLI] Processing {len(files)} query images from '{input_path}'...")
        all_results = []
        for p in files:
            res_vis = str(out_dir / f"{p.stem}_match_report.png")
            args.save_vis = res_vis
            res = process_single(matcher, p, args)
            all_results.append(res)

        summary_file = out_dir / "batch_match_results.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✔ All batch match results saved to: {summary_file}")


if __name__ == "__main__":
    main()
