#!/usr/bin/env python3
"""
Dataset Preprocessor: Foreground Object Extraction with Black Background
========================================================================
Batch-extracts foreground objects from raw dataset images and converts
the background into pure black (RGB (0, 0, 0)) using RMBG-1.4.

Maintains directory structure (e.g., train/class_x/img_y.jpg) and supports:
- Solid black background compositing
- Foreground bounding-box square cropping & centering (optional)
- Standardized image resizing (e.g., 224x224, 256x256)
- Progress tracking with tqdm
- Resuming / skipping already processed images
- Alpha mask export

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Import existing modular background remover
from bg_removal import BackgroundRemover, PipelineConfig


# Supported image extensions
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def crop_and_pad_square(
    image: Image.Image,
    alpha_mask: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    padding: float = 0.1,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Image.Image]:
    """
    Crop tightly around the object's alpha mask with padding, and place centered
    on a square background of `bg_color`.

    Args:
        image: RGB PIL image with black background.
        alpha_mask: 8-bit grayscale alpha mask.
        target_size: Optional (width, height) to resize the final square image.
        padding: Fractional padding around the bounding box (e.g. 0.1 = 10%).
        bg_color: Solid background color tuple (default: black (0, 0, 0)).

    Returns:
        (square_rgb_image, square_alpha_mask)
    """
    bbox = alpha_mask.getbbox()
    if bbox is None:
        # No object detected; return empty square
        size = target_size or (224, 224)
        blank = Image.new("RGB", size, bg_color)
        blank_mask = Image.new("L", size, 0)
        return blank, blank_mask

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(image.width, x2 + pad_w)
    y2 = min(image.height, y2 + pad_h)

    cropped_img = image.crop((x1, y1, x2, y2))
    cropped_mask = alpha_mask.crop((x1, y1, x2, y2))

    cw, ch = cropped_img.size
    max_side = max(cw, ch)

    # Pad to square
    square_img = Image.new("RGB", (max_side, max_side), bg_color)
    square_mask = Image.new("L", (max_side, max_side), 0)

    offset_x = (max_side - cw) // 2
    offset_y = (max_side - ch) // 2

    square_img.paste(cropped_img, (offset_x, offset_y))
    square_mask.paste(cropped_mask, (offset_x, offset_y))

    if target_size:
        square_img = square_img.resize(target_size, Image.Resampling.BILINEAR)
        square_mask = square_mask.resize(target_size, Image.Resampling.NEAREST)

    return square_img, square_mask


@dataclass
class PreprocessingStats:
    """Statistics for dataset extraction run."""
    total_found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    total_time_s: float = 0.0

    @property
    def avg_time_per_image_ms(self) -> float:
        if self.processed == 0:
            return 0.0
        return (self.total_time_s / self.processed) * 1000.0


class DatasetPreprocessor:
    """
    Processes a directory or dataset of raw images, extracting foreground objects
    and placing them onto pure black backgrounds.
    """

    def __init__(
        self,
        remover: Optional[BackgroundRemover] = None,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        crop_object: bool = True,
        padding: float = 0.1,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        save_masks: bool = False,
    ):
        self.remover = remover or BackgroundRemover()
        self.target_size = target_size
        self.crop_object = crop_object
        self.padding = padding
        self.bg_color = bg_color
        self.save_masks = save_masks

    def process_single_image(
        self,
        input_image: Union[str, Path, Image.Image],
        output_image_path: Optional[Union[str, Path]] = None,
        output_mask_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Extract foreground object, composite onto black background, optionally
        crop around object bounding box, and save to disk.
        """
        # Step 1: Background removal with black compositing
        black_bg_img, alpha_mask, _ = self.remover.remove_background(
            input_image,
            background_color=self.bg_color,
            return_latency=False,
        )

        # Step 2: Optional Bounding-Box Square Cropping
        if self.crop_object:
            final_img, final_mask = crop_and_pad_square(
                black_bg_img,
                alpha_mask,
                target_size=self.target_size,
                padding=self.padding,
                bg_color=self.bg_color,
            )
        else:
            final_img = black_bg_img
            final_mask = alpha_mask
            if self.target_size:
                final_img = final_img.resize(self.target_size, Image.Resampling.BILINEAR)
                final_mask = final_mask.resize(self.target_size, Image.Resampling.NEAREST)

        # Step 3: Save outputs if paths provided
        if output_image_path:
            out_p = Path(output_image_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            # Save as PNG to avoid compression artifacts on pure black background
            final_img.save(out_p, format="PNG")

        if output_mask_path:
            mask_p = Path(output_mask_path)
            mask_p.parent.mkdir(parents=True, exist_ok=True)
            final_mask.save(mask_p, format="PNG")

        return final_img, final_mask

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        overwrite: bool = False,
        max_images: Optional[int] = None,
    ) -> PreprocessingStats:
        """
        Recursively process all images in input_dir, recreating the directory
        hierarchy under output_dir with black-background extracted objects.
        """
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        # Gather all image files
        all_files = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
        ]
        all_files.sort()

        if max_images and max_images > 0:
            all_files = all_files[:max_images]

        stats = PreprocessingStats(total_found=len(all_files))
        print(f"\n[DatasetPreprocessor] Found {len(all_files)} images in '{input_dir}'")
        print(f"[DatasetPreprocessor] Destination directory: '{output_dir}'")
        print(f"[DatasetPreprocessor] Settings: crop_object={self.crop_object}, target_size={self.target_size}, bg_color={self.bg_color}\n")

        start_time = time.time()

        for idx, img_path in enumerate(all_files, start=1):
            rel_path = img_path.relative_to(input_dir)
            # Force output to .png
            out_rel_path = rel_path.with_suffix(".png")
            out_file_path = output_dir / out_rel_path
            out_mask_path = output_dir / "masks" / out_rel_path if self.save_masks else None

            if out_file_path.exists() and not overwrite:
                stats.skipped += 1
                if idx % 20 == 0 or idx == len(all_files):
                    print(f"[{idx}/{len(all_files)}] (Skipped existing) {rel_path}")
                continue

            try:
                t0 = time.time()
                self.process_single_image(
                    img_path,
                    output_image_path=out_file_path,
                    output_mask_path=out_mask_path,
                )
                dt = time.time() - t0
                stats.processed += 1
                stats.total_time_s += dt

                print(f"[{idx}/{len(all_files)}] Processed: {rel_path} -> {out_rel_path} ({dt*1000:.1f}ms)")
            except Exception as e:
                stats.failed += 1
                print(f"[{idx}/{len(all_files)}] FAILED: {rel_path} (Error: {e})", file=sys.stderr)

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(" PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f" Total files scanned    : {stats.total_found}")
        print(f" Successfully processed : {stats.processed}")
        print(f" Skipped (already exist): {stats.skipped}")
        print(f" Failed                 : {stats.failed}")
        print(f" Total time elapsed     : {total_elapsed:.2f} s")
        if stats.processed > 0:
            print(f" Average latency        : {stats.avg_time_per_image_ms:.2f} ms/image ({1000.0/stats.avg_time_per_image_ms:.2f} FPS)")
        print("=" * 60 + "\n")

        return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract foreground objects from dataset images and convert background to black (0, 0, 0)."
    )
    parser.add_argument("-i", "--input-dir", required=True, type=str, help="Input directory containing raw images.")
    parser.add_argument("-o", "--output-dir", required=True, type=str, help="Output directory to save black-background images.")
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224], metavar=("WIDTH", "HEIGHT"), help="Resize output images (default: 224 224).")
    parser.add_argument("--no-crop", dest="crop_object", action="store_false", help="Do not crop around object bounding box.")
    parser.set_defaults(crop_object=True)
    parser.add_argument("--padding", type=float, default=0.1, help="Fractional padding around object bounding box (default: 0.1).")
    parser.add_argument("--bg-color", type=int, nargs=3, default=[0, 0, 0], metavar=("R", "G", "B"), help="RGB background color (default: 0 0 0 for black).")
    parser.add_argument("--save-masks", action="store_true", help="Also save binary alpha masks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process.")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="Inference device.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = PipelineConfig(device=args.device)
    remover = BackgroundRemover(config)

    preprocessor = DatasetPreprocessor(
        remover=remover,
        target_size=tuple(args.target_size) if args.target_size else None,
        crop_object=args.crop_object,
        padding=args.padding,
        bg_color=tuple(args.bg_color),
        save_masks=args.save_masks,
    )

    preprocessor.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
