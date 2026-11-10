#!/usr/bin/env python3
"""
CIFAR+10 Dataset Downloader & Black Background Processor
========================================================
Downloads and prepares the standard Open Set Recognition (OSR) CIFAR+10 benchmark
(as established by Neal et al., ECCV 2018 and Dhamija et al., NeurIPS 2019).

Dataset Partitioning:
- Known Classes (from CIFAR-10):
    Protocol A (Neal et al. standard): 4 classes ('airplane', 'automobile', 'bird', 'cat')
    Protocol B (Full CIFAR-10): 10 classes
- Unknown Classes (from CIFAR-100, 10 non-overlapping semantic categories):
    ('apple', 'bed', 'bee', 'bottle', 'bowl', 'bridge', 'chair', 'clock', 'cup', 'table')

Processing Steps:
1. Stream/download CIFAR-10 and CIFAR-100 image sets from fast Amazon S3 mirror.
2. Filter and extract target known and unknown classes into structured raw directory.
3. Apply RMBG-1.4 alpha matting to segment foreground objects and composite onto solid black (0, 0, 0).
4. Save processed black-background images and masks into structured directories.
5. Generate visual comparison artifact and metadata JSON.

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw

from bg_removal import BackgroundRemover, PipelineConfig
from dataset_preprocessor import crop_and_pad_square


# Fast S3 CDN Mirrors for CIFAR-10 and CIFAR-100
CIFAR10_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
CIFAR100_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"

# Standard OSR Class Sets
NEAL_KNOWN_4 = ["airplane", "automobile", "bird", "cat"]
FULL_KNOWN_10 = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
OSR_UNKNOWN_10 = [
    "apple", "bed", "bee", "bottle", "bowl",
    "bridge", "chair", "clock", "cup", "table"
]


def stream_extract_classes(
    tar_url: str,
    target_classes: Set[str],
    output_dir: Path,
    samples_per_class: int,
    is_cifar100: bool = False,
) -> Dict[str, int]:
    """
    Stream tarball directly from URL and extract images for target classes
    up to samples_per_class per category.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts: Dict[str, int] = {c: 0 for c in target_classes}

    print(f"\n[Downloader] Streaming from {tar_url} ...")
    print(f"[Downloader] Target classes ({len(target_classes)}): {sorted(list(target_classes))}")
    print(f"[Downloader] Target samples per class: {samples_per_class if samples_per_class > 0 else 'All'}")

    req = urllib.request.Request(
        tar_url,
        headers={"User-Agent": "Mozilla/5.0 (OSR-Project-Thesis/1.0)"}
    )

    extracted_total = 0
    t0 = time.time()

    with urllib.request.urlopen(req, timeout=30) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                # Check if all classes have reached target count
                if samples_per_class > 0 and all(c >= samples_per_class for c in class_counts.values()):
                    print("[Downloader] Reached target count for all requested classes!")
                    break

                if not member.name.endswith(".png"):
                    continue

                parts = member.name.split("/")
                # CIFAR-10 structure: cifar10/train/<class>/<id>.png or cifar10/test/<class>/<id>.png
                # CIFAR-100 structure: cifar100/train/<superclass>/<class>/<id>.png
                cls_name = parts[-2] if is_cifar100 else parts[-2]

                if cls_name in target_classes:
                    if samples_per_class > 0 and class_counts[cls_name] >= samples_per_class:
                        continue

                    # Extract file
                    f = tar.extractfile(member)
                    if f is not None:
                        cls_dir = output_dir / cls_name
                        cls_dir.mkdir(parents=True, exist_ok=True)

                        filename = f"{cls_name}_{class_counts[cls_name]:04d}.png"
                        out_path = cls_dir / filename
                        with open(out_path, "wb") as out_fp:
                            out_fp.write(f.read())

                        class_counts[cls_name] += 1
                        extracted_total += 1

                        if extracted_total % 25 == 0 or extracted_total == 1:
                            print(f"  • Extracted {extracted_total} images... (Current: '{cls_name}' = {class_counts[cls_name]})")

    elapsed = time.time() - t0
    print(f"[Downloader] Extracted {extracted_total} images in {elapsed:.1f}s.")
    return class_counts


def create_cifar_comparison_sheet(
    raw_images: List[Tuple[str, Image.Image]],
    black_images: List[Tuple[str, Image.Image]],
    save_path: Path,
) -> Image.Image:
    """
    Create a side-by-side visual contact sheet showing original CIFAR images
    and their extracted black-background counterparts for every class.
    """
    num_samples = min(len(raw_images), len(black_images), 10)
    thumb_w, thumb_h = 120, 120
    header_h = 45
    pad = 10

    total_w = (thumb_w * 2 + pad * 3)
    total_h = header_h + num_samples * (thumb_h + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), (24, 25, 29))
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([(0, 0), (total_w, header_h)], fill=(38, 41, 51))
    draw.text((pad, 14), "CIFAR+10: Raw Input vs Black Background Cutout", fill=(255, 255, 255))

    y = header_h + pad
    for i in range(num_samples):
        cls_name, raw_img = raw_images[i]
        _, black_img = black_images[i]

        # Upscale thumb for clarity
        r_thumb = raw_img.resize((thumb_w, thumb_h), Image.Resampling.NEAREST)
        b_thumb = black_img.resize((thumb_w, thumb_h), Image.Resampling.NEAREST)

        # Draw raw
        canvas.paste(r_thumb, (pad, y))
        draw.rectangle([(pad, y), (pad + thumb_w, y + 20)], fill=(0, 0, 0, 180))
        draw.text((pad + 4, y + 3), f"Raw: {cls_name[:8]}", fill=(255, 255, 255))

        # Draw black
        bx = pad * 2 + thumb_w
        canvas.paste(b_thumb, (bx, y))
        draw.rectangle([(bx, y), (bx + thumb_w, y + 20)], fill=(0, 0, 0, 180))
        draw.text((bx + 4, y + 3), "Black BG", fill=(50, 205, 50))

        y += thumb_h + pad

    canvas.save(save_path)
    print(f"✔ Saved visual comparison sheet to: {save_path}")
    return canvas


def download_and_process_cifar_plus_10(
    output_dir: str = "./data/cifar_plus_10",
    protocol: str = "neal",
    samples_per_class: int = 20,
    target_size: Tuple[int, int] = (224, 224),
    crop_object: bool = True,
    device: Optional[str] = None,
    save_masks: bool = True,
) -> Dict:
    """
    Main orchestration routine:
    1. Downloads CIFAR+10 raw images.
    2. Runs neural background removal with black compositing (0, 0, 0).
    3. Saves structured datasets and visual artifacts.
    """
    root_dir = Path(output_dir).resolve()
    raw_dir = root_dir / "raw"
    black_dir = root_dir / "black"
    masks_dir = black_dir / "masks" if save_masks else None

    raw_dir.mkdir(parents=True, exist_ok=True)
    black_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine Known & Unknown Class Sets
    if protocol.lower() == "neal":
        known_classes = NEAL_KNOWN_4
    elif protocol.lower() == "full":
        known_classes = FULL_KNOWN_10
    else:
        raise ValueError(f"Unknown protocol: '{protocol}'. Choose 'neal' (4 known) or 'full' (10 known).")

    unknown_classes = OSR_UNKNOWN_10

    print("=" * 65)
    print(" CIFAR+10 OPEN SET RECOGNITION DATASET PIPELINE")
    print("=" * 65)
    print(f" Output Root Directory   : {root_dir}")
    print(f" Benchmark Protocol      : {protocol.upper()}")
    print(f" Known Classes ({len(known_classes):2d})     : {known_classes}")
    print(f" Unknown Classes ({len(unknown_classes):2d})   : {unknown_classes}")
    print(f" Samples Per Class       : {samples_per_class}")
    print(f" Target Processed Size   : {target_size}")
    print(f" Crop & Square Object    : {crop_object}")
    print("=" * 65)

    # 2. Download / Stream Raw Images
    print("\n>>> Stage 1: Downloading CIFAR-10 (Known Classes)...")
    cifar10_counts = stream_extract_classes(
        tar_url=CIFAR10_URL,
        target_classes=set(known_classes),
        output_dir=raw_dir,
        samples_per_class=samples_per_class,
        is_cifar100=False,
    )

    print("\n>>> Stage 2: Downloading CIFAR-100 (Unknown / Open-Set Classes)...")
    cifar100_counts = stream_extract_classes(
        tar_url=CIFAR100_URL,
        target_classes=set(unknown_classes),
        output_dir=raw_dir,
        samples_per_class=samples_per_class,
        is_cifar100=True,
    )

    total_raw = sum(cifar10_counts.values()) + sum(cifar100_counts.values())
    print(f"\n✔ Successfully assembled {total_raw} raw CIFAR+10 images in '{raw_dir}'.")

    # 3. Background Removal & Black Background Compositing
    print("\n>>> Stage 3: Removing Backgrounds with RMBG-1.4 onto Black (0, 0, 0)...")
    config = PipelineConfig(device=device)
    remover = BackgroundRemover(config)

    all_raw_files: List[Path] = []
    for cls in known_classes + unknown_classes:
        cls_p = raw_dir / cls
        if cls_p.exists():
            all_raw_files.extend(sorted(cls_p.glob("*.png")))

    sample_raw_vis: List[Tuple[str, Image.Image]] = []
    sample_black_vis: List[Tuple[str, Image.Image]] = []

    t_start_bg = time.time()
    processed_count = 0
    total_latency_ms = 0.0

    print(f"[Preprocessor] Processing {len(all_raw_files)} images...")
    line = "─" * 65
    print(line)

    for idx, raw_path in enumerate(all_raw_files, start=1):
        cls_name = raw_path.parent.name
        out_cls_dir = black_dir / cls_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)
        out_file_path = out_cls_dir / raw_path.name

        out_mask_path = None
        if save_masks and masks_dir:
            mask_cls_dir = masks_dir / cls_name
            mask_cls_dir.mkdir(parents=True, exist_ok=True)
            out_mask_path = mask_cls_dir / raw_path.name

        try:
            with Image.open(raw_path) as raw_img:
                raw_rgb = raw_img.convert("RGB")

            # Remove background with pure solid black (0, 0, 0)
            black_bg_img, alpha_mask, lat = remover.remove_background(
                raw_rgb,
                background_color=(0, 0, 0),
                return_latency=True,
            )

            # Optional bounding-box square crop & centered padding
            if crop_object:
                final_img, final_mask = crop_and_pad_square(
                    black_bg_img,
                    alpha_mask,
                    target_size=target_size,
                    padding=0.1,
                    bg_color=(0, 0, 0),
                )
            else:
                final_img = black_bg_img.resize(target_size, Image.Resampling.BILINEAR)
                final_mask = alpha_mask.resize(target_size, Image.Resampling.NEAREST)

            # Save PNG outputs
            final_img.save(out_file_path, format="PNG")
            if out_mask_path:
                final_mask.save(out_mask_path, format="PNG")

            processed_count += 1
            if lat:
                total_latency_ms += lat.total_ms

            # Collect representative samples for visual contact sheet
            if len(sample_raw_vis) < len(known_classes) + len(unknown_classes):
                if not any(s[0] == cls_name for s in sample_raw_vis):
                    sample_raw_vis.append((cls_name, raw_rgb))
                    sample_black_vis.append((cls_name, final_img))

            if idx % 10 == 0 or idx == len(all_raw_files):
                dt_ms = lat.total_ms if lat else 0.0
                print(f"[{idx:3d}/{len(all_raw_files):3d}] {cls_name:<16} -> {out_file_path.name} ({dt_ms:.1f}ms)")

        except Exception as e:
            print(f"[{idx:3d}/{len(all_raw_files):3d}] FAILED on {raw_path}: {e}", file=sys.stderr)

    total_elapsed = time.time() - t_start_bg
    avg_latency = total_latency_ms / processed_count if processed_count > 0 else 0.0
    print(line)
    print(f"✔ Completed background removal in {total_elapsed:.1f}s. Avg Latency: {avg_latency:.1f}ms/image.")

    # 4. Generate Visual Comparison Sheet
    vis_path = root_dir / "cifar_plus_10_comparison.png"
    create_cifar_comparison_sheet(sample_raw_vis, sample_black_vis, vis_path)

    # 5. Save Dataset Metadata JSON
    info = {
        "dataset_name": "CIFAR+10",
        "benchmark_protocol": protocol,
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "num_known_classes": len(known_classes),
        "num_unknown_classes": len(unknown_classes),
        "total_classes": len(known_classes) + len(unknown_classes),
        "samples_per_class": samples_per_class,
        "total_raw_images": total_raw,
        "total_black_bg_images": processed_count,
        "target_size": list(target_size),
        "crop_object": crop_object,
        "bg_color": [0, 0, 0],
        "avg_latency_ms": round(avg_latency, 2),
        "total_processing_time_s": round(total_elapsed, 2),
        "raw_dir": str(raw_dir),
        "black_dir": str(black_dir),
        "comparison_sheet": str(vis_path),
    }

    meta_file = root_dir / "cifar_plus_10_info.json"
    with open(meta_file, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✔ Dataset metadata saved to: {meta_file}")

    print("\n" + "=" * 65)
    print(" CIFAR+10 DATASET EXTRACTION SUMMARY")
    print("=" * 65)
    print(f" Total Raw Images Downloaded : {total_raw}")
    print(f" Total Black BG Images Saved : {processed_count}")
    print(f" Destination Directory       : {black_dir}")
    print(f" Visual Sheet                : {vis_path}")
    print("=" * 65 + "\n")

    return info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CIFAR+10 OSR dataset, remove backgrounds, and composite on solid black (0, 0, 0)."
    )
    parser.add_argument(
        "-o", "--output-dir", default="./data/cifar_plus_10", type=str,
        help="Root directory to store CIFAR+10 raw and black-background images."
    )
    parser.add_argument(
        "-p", "--protocol", default="neal", choices=["neal", "full"],
        help="OSR benchmark protocol: 'neal' (4 known + 10 unknown) or 'full' (10 known + 10 unknown)."
    )
    parser.add_argument(
        "-n", "--samples-per-class", default=15, type=int,
        help="Number of images per class to extract (default: 15 for fast setup; use larger for full training)."
    )
    parser.add_argument(
        "--target-size", type=int, nargs=2, default=[224, 224], metavar=("WIDTH", "HEIGHT"),
        help="Output image size (default: 224 224)."
    )
    parser.add_argument(
        "--no-crop", dest="crop_object", action="store_false",
        help="Do not crop around object bounding box."
    )
    parser.set_defaults(crop_object=True)
    parser.add_argument(
        "--no-masks", dest="save_masks", action="store_false",
        help="Do not save alpha masks."
    )
    parser.set_defaults(save_masks=True)
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Compute device for RMBG-1.4 inference."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download_and_process_cifar_plus_10(
        output_dir=args.output_dir,
        protocol=args.protocol,
        samples_per_class=args.samples_per_class,
        target_size=tuple(args.target_size),
        crop_object=args.crop_object,
        device=args.device,
        save_masks=args.save_masks,
    )


if __name__ == "__main__":
    main()
