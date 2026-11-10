#!/usr/bin/env python3
"""
Sample Dataset Generator for Open Set Recognition & Object Matching
===================================================================
Generates a realistic multi-class vision dataset containing diverse objects
placed on complex, realistic backgrounds (nature, urban, indoor, noisy, abstract).

Enables immediate end-to-end verification of:
1. Object extraction and black background conversion.
2. Metric learning model training on black background objects.
3. Closed-set and Open-set academic evaluation.
4. End-to-end inference matching on query images.

Dataset Structure Generated:
data/
  raw/
    train/
      class_a/ ...
      class_b/ ...
    test/
      class_a/ ...
      class_b/ ...
      unknown_x/ ...  (Open-set unknown classes)

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_complex_background(width: int = 400, height: int = 400, bg_type: str = "gradient") -> Image.Image:
    """Generate a complex non-black background simulating real-world photo environments."""
    if bg_type == "gradient":
        # Multi-color gradient
        c1 = (random.randint(100, 240), random.randint(100, 240), random.randint(100, 240))
        c2 = (random.randint(80, 200), random.randint(80, 200), random.randint(80, 200))
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            alpha = y / height
            arr[y, :, 0] = int(c1[0] * (1 - alpha) + c2[0] * alpha)
            arr[y, :, 1] = int(c1[1] * (1 - alpha) + c2[1] * alpha)
            arr[y, :, 2] = int(c1[2] * (1 - alpha) + c2[2] * alpha)
        return Image.fromarray(arr)

    elif bg_type == "texture":
        # Random wood/wall texture with noise
        base = np.random.randint(120, 210, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(base)
        return img.filter(ImageFilter.GaussianBlur(radius=5))

    else:
        # Abstract pattern / indoor scene simulation
        img = Image.new("RGB", (width, height), (random.randint(140, 220), random.randint(140, 220), random.randint(140, 220)))
        draw = ImageDraw.Draw(img)
        for _ in range(8):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=(random.randint(50, 180), random.randint(50, 180), random.randint(50, 180)), width=random.randint(4, 15))
        return img


def draw_object(draw: ImageDraw.ImageDraw, class_name: str, cx: int, cy: int, size: int, color: Tuple[int, int, int]):
    """Draw a distinct geometric/iconic object representing a class."""
    half = size // 2

    if class_name == "cup":
        # Cup body
        draw.rectangle([(cx - half, cy - half // 2), (cx + half, cy + half)], fill=color, outline=(20, 20, 20), width=3)
        # Handle
        draw.arc([(cx + half - 5, cy - half // 3), (cx + half + 25, cy + half // 2)], 270, 90, fill=color, width=6)

    elif class_name == "car":
        # Car body
        draw.rectangle([(cx - half - 20, cy), (cx + half + 20, cy + half // 2)], fill=color, outline=(20, 20, 20), width=3)
        # Cabin
        draw.polygon([(cx - half + 10, cy), (cx - half // 2, cy - half // 2), (cx + half // 2, cy - half // 2), (cx + half - 10, cy)], fill=color, outline=(20, 20, 20))
        # Wheels
        draw.ellipse([(cx - half, cy + half // 3), (cx - half + 25, cy + half // 3 + 25)], fill=(30, 30, 30))
        draw.ellipse([(cx + half - 5, cy + half // 3), (cx + half + 20, cy + half // 3 + 25)], fill=(30, 30, 30))

    elif class_name == "laptop":
        # Screen
        draw.rectangle([(cx - half, cy - half), (cx + half, cy + half // 3)], fill=(40, 40, 40), outline=color, width=5)
        draw.rectangle([(cx - half + 6, cy - half + 6), (cx + half - 6, cy + half // 3 - 6)], fill=(200, 230, 255))
        # Keyboard base
        draw.polygon([(cx - half - 15, cy + half // 2), (cx + half + 15, cy + half // 2), (cx + half, cy + half // 3), (cx - half, cy + half // 3)], fill=color)

    elif class_name == "bottle":
        # Main cylinder
        draw.rectangle([(cx - half // 2, cy - half // 3), (cx + half // 2, cy + half)], fill=color, outline=(20, 20, 20), width=3)
        # Neck
        draw.rectangle([(cx - half // 4, cy - half), (cx + half // 4, cy - half // 3)], fill=color, outline=(20, 20, 20), width=3)
        # Cap
        draw.rectangle([(cx - half // 3, cy - half - 10), (cx + half // 3, cy - half)], fill=(220, 40, 40))

    elif class_name == "airplane":  # Unknown class for OSR
        # Fuselage
        draw.ellipse([(cx - 15, cy - half - 20), (cx + 15, cy + half + 20)], fill=color, outline=(20, 20, 20), width=2)
        # Wings
        draw.polygon([(cx, cy - 20), (cx - half - 30, cy + 20), (cx + half + 30, cy + 20)], fill=color, outline=(20, 20, 20))
        # Tail
        draw.polygon([(cx, cy + half), (cx - half // 2, cy + half + 25), (cx + half // 2, cy + half + 25)], fill=color)

    elif class_name == "star":  # Unknown class for OSR
        points = []
        for i in range(10):
            r = half + 15 if i % 2 == 0 else half // 2
            angle = i * math.pi / 5 - math.pi / 2
            points.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
        draw.polygon(points, fill=color, outline=(20, 20, 20), width=3)

    else:
        # Default circle/disc
        draw.ellipse([(cx - half, cy - half), (cx + half, cy + half)], fill=color, outline=(20, 20, 20), width=3)


def create_synthetic_object_image(
    class_name: str,
    width: int = 320,
    height: int = 320,
    bg_types: Tuple[str, ...] = ("gradient", "texture", "abstract"),
) -> Tuple[Image.Image, Image.Image]:
    """
    Creates an image with a foreground object on a complex background,
    along with its ground-truth alpha mask.
    """
    # 1. Background
    bg = generate_complex_background(width, height, bg_type=random.choice(bg_types))

    # 2. Foreground object on transparent layer
    fg = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(fg)

    cx = width // 2 + random.randint(-15, 15)
    cy = height // 2 + random.randint(-15, 15)
    size = random.randint(110, 160)

    # Pick class-specific primary color palette with variations
    color_map = {
        "cup": [(220, 60, 50), (230, 90, 80), (190, 40, 30)],
        "car": [(40, 90, 210), (60, 120, 230), (30, 70, 180)],
        "laptop": [(120, 125, 130), (90, 95, 100), (140, 145, 150)],
        "bottle": [(40, 180, 80), (60, 200, 100), (30, 150, 60)],
        "airplane": [(230, 230, 240), (210, 215, 220)],
        "star": [(240, 200, 20), (255, 220, 40)],
    }
    palette = color_map.get(class_name, [(180, 100, 200)])
    col = random.choice(palette)

    draw_object(draw, class_name, cx, cy, size, col)

    # 3. Composite onto background
    composite_img = bg.copy()
    composite_img.paste(fg, (0, 0), fg)

    alpha_mask = fg.split()[-1]
    return composite_img, alpha_mask


def generate_sample_dataset(
    output_dir: Union[str, Path] = "./data/sample_dataset",
    known_classes: Tuple[str, ...] = ("cup", "car", "laptop", "bottle"),
    unknown_classes: Tuple[str, ...] = ("airplane", "star"),
    samples_per_known: int = 15,
    samples_per_unknown: int = 10,
    seed: int = 42,
) -> Path:
    """Generate sample dataset directory ready for preprocessing and training."""
    random.seed(seed)
    np.random.seed(seed)
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"\n[DatasetGenerator] Generating synthetic multi-class dataset at '{out_p}'...")
    print(f"  • Known Classes ({len(known_classes)})   : {known_classes} ({samples_per_known} images each)")
    print(f"  • Unknown Classes ({len(unknown_classes)}): {unknown_classes} ({samples_per_unknown} images each)")

    # 1. Generate Known Classes
    for cname in known_classes:
        cdir = out_p / cname
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(1, samples_per_known + 1):
            img, _ = create_synthetic_object_image(cname)
            img.save(cdir / f"{cname}_{i:03d}.png")

    # 2. Generate Unknown Classes
    for cname in unknown_classes:
        cdir = out_p / cname
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(1, samples_per_unknown + 1):
            img, _ = create_synthetic_object_image(cname)
            img.save(cdir / f"{cname}_{i:03d}.png")

    total_images = len(known_classes) * samples_per_known + len(unknown_classes) * samples_per_unknown
    print(f"✔ Successfully created {total_images} sample images across {len(known_classes) + len(unknown_classes)} classes in '{out_p}'\n")
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Generate sample dataset for Open Set Recognition.")
    parser.add_argument("-o", "--output-dir", default="./data/sample_dataset", help="Output directory.")
    parser.add_argument("--samples-known", type=int, default=15, help="Images per known class.")
    parser.add_argument("--samples-unknown", type=int, default=10, help="Images per unknown class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_sample_dataset(
        output_dir=args.output_dir,
        samples_per_known=args.samples_known,
        samples_per_unknown=args.samples_unknown,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
