#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoaders for Open Set Recognition (OSR)
=============================================================
Manages image datasets with black-background extracted objects.
Handles known-class vs unknown-class partitioning for Open Set Recognition.

Key Components:
1. BlackBackgroundTransforms: Specialized augmentations that preserve pure black borders.
2. OSRImageDataset: PyTorch Dataset for known/unknown class partitions.
3. OnlineBlackBackgroundDataset: On-the-fly foreground extraction + black bg compositing.
4. DataLoader utilities: Batching, shuffling, and multi-worker loading.

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance, ImageOps

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ==============================================================================
# Specialized Black-Background Transforms & Augmentations
# ==============================================================================

class BlackBackgroundTransforms:
    """
    Data augmentations tailored for objects on pure black backgrounds.
    Ensures padding and out-of-boundary regions remain strictly black (0, 0, 0).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        is_training: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_rotation_deg: float = 15.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.3,
    ):
        self.target_size = target_size
        self.is_training = is_training
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.max_rotation_deg = max_rotation_deg
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply augmentations and convert PIL RGB Image to normalized PyTorch tensor."""
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.is_training:
            # 1. Random Horizontal Flip
            if random.random() < self.hflip_prob:
                img = ImageOps.mirror(img)

            # 2. Random Rotation with pure black fill (0, 0, 0)
            if self.max_rotation_deg > 0:
                angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

            # 3. Random Affine Scaling with black padding
            if self.scale_range != (1.0, 1.0):
                scale = random.uniform(self.scale_range[0], self.scale_range[1])
                w, h = img.size
                nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
                scaled = img.resize((nw, nh), Image.Resampling.BILINEAR)

                # Paste onto black canvas
                canvas = Image.new("RGB", (w, h), (0, 0, 0))
                px = (w - nw) // 2
                py = (h - nh) // 2
                # If larger than canvas, crop; if smaller, center paste
                if scale > 1.0:
                    crop_x = (nw - w) // 2
                    crop_y = (nh - h) // 2
                    canvas = scaled.crop((crop_x, crop_y, crop_x + w, crop_y + h))
                else:
                    canvas.paste(scaled, (px, py))
                img = canvas

            # 4. Color Jitter on foreground pixels
            if random.random() < self.color_jitter_prob:
                factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(factor)
                factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)

        # Standard Resize
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.Resampling.BILINEAR)

        # Convert to Tensor [3, H, W] in [0, 1]
        np_arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_arr.transpose(2, 0, 1))

        # Normalize with Mean and Std
        tensor = (tensor - self.mean) / self.std
        return tensor


# ==============================================================================
# Open Set Recognition PyTorch Dataset
# ==============================================================================

class OSRImageDataset(Dataset):
    """
    Dataset representing an Open Set Recognition scenario.

    Classes designated as `known_classes` have non-negative labels in range [0, K-1].
    Classes designated as `unknown_classes` (or not in known_classes) have label -1.

    Directory structure supported:
    - Root with class subfolders: `root/<class_name>/<image_file>`
    """

    def __init__(
        self,
        samples: List[Tuple[Path, str]],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        is_unknown: Optional[Callable[[str], bool]] = None,
    ):
        """
        Args:
            samples: List of (image_path, class_name) tuples.
            class_to_idx: Mapping from known class name to integer label [0, K-1].
            transform: Optional image transformation callable.
            is_unknown: Optional boolean predicate to test if a class name is unknown.
        """
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        self.transform = transform or BlackBackgroundTransforms(is_training=False)
        self.is_unknown = is_unknown or (lambda c: c not in self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, str]:
        """
        Returns:
            (tensor, label, class_name, file_path_str)
            label is in [0, K-1] for known classes, and -1 for unknown classes.
        """
        path, class_name = self.samples[index]

        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            # Return empty black tensor on corrupt image
            print(f"Warning: failed reading image {path}: {e}")
            img_rgb = Image.new("RGB", (224, 224), (0, 0, 0))

        tensor = self.transform(img_rgb)

        if self.is_unknown(class_name):
            label = -1
        else:
            label = self.class_to_idx[class_name]

        return tensor, label, class_name, str(path)


# ==============================================================================
# Online Black Background Dataset (Extracts on-the-fly)
# ==============================================================================

class OnlineBlackBackgroundDataset(Dataset):
    """
    Applies BackgroundRemover on-the-fly to raw images and composites onto black background.
    Includes in-memory caching to avoid repeated forward passes on the same image across epochs.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, str]],
        class_to_idx: Dict[str, int],
        remover,  # BackgroundRemover
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        crop_object: bool = True,
        cache_in_memory: bool = True,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        self.remover = remover
        self.transform = transform or BlackBackgroundTransforms(is_training=False)
        self.crop_object = crop_object
        self.cache_in_memory = cache_in_memory
        self.cache: Dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, str]:
        path, class_name = self.samples[index]
        path_str = str(path)

        if self.cache_in_memory and path_str in self.cache:
            black_bg_img = self.cache[path_str]
        else:
            # Extract object and composite on black
            black_bg_img, alpha, _ = self.remover.remove_background(
                path, background_color=(0, 0, 0), return_latency=False
            )
            if self.crop_object:
                bbox = alpha.getbbox()
                if bbox:
                    cw = bbox[2] - bbox[0]
                    ch = bbox[3] - bbox[1]
                    side = max(cw, ch)
                    sq = Image.new("RGB", (side, side), (0, 0, 0))
                    sq.paste(black_bg_img.crop(bbox), ((side - cw) // 2, (side - ch) // 2))
                    black_bg_img = sq

            if self.cache_in_memory:
                self.cache[path_str] = black_bg_img

        tensor = self.transform(black_bg_img)
        label = self.class_to_idx.get(class_name, -1)
        return tensor, label, class_name, path_str


# ==============================================================================
# Helper Functions: Splitting and DataLoaders
# ==============================================================================

def scan_dataset_directory(data_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """
    Scan directory for class folders: data_dir/<class_name>/<images>.
    Returns dictionary: {class_name: [list of Path]}
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    class_images: Dict[str, List[Path]] = {}
    for sub in sorted(data_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith((".", "__")) and sub.name != "masks":
            imgs = [
                p for p in sub.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
            ]
            imgs.sort()
            if imgs:
                class_images[sub.name] = imgs

    return class_images


def split_known_unknown_classes(
    all_classes: List[str],
    known_classes: Optional[Sequence[str]] = None,
    num_known: Optional[int] = None,
    known_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Partition all available classes into Known classes and Unknown (Open-Set) classes.

    Args:
        all_classes: Full list of class names.
        known_classes: Explicit list of known class names (if provided).
        num_known: Specific number of classes to treat as known.
        known_ratio: Fraction of classes to designate as known (if num_known is None).
        seed: Random seed for reproducible class partition.

    Returns:
        (known_classes_list, unknown_classes_list)
    """
    classes = sorted(all_classes)
    if known_classes is not None:
        known_set = set(known_classes)
        known = [c for c in classes if c in known_set]
        unknown = [c for c in classes if c not in known_set]
        return known, unknown

    rng = random.Random(seed)
    shuffled = list(classes)
    rng.shuffle(shuffled)

    if num_known is not None:
        k = min(len(shuffled), max(1, num_known))
    else:
        k = max(1, int(len(shuffled) * known_ratio))

    known = sorted(shuffled[:k])
    unknown = sorted(shuffled[k:])
    return known, unknown


def partition_train_val_test_samples(
    class_images: Dict[str, List[Path]],
    known_classes: List[str],
    unknown_classes: List[str],
    val_ratio: float = 0.15,
    test_known_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """
    Partition images into:
    - Train: only Known classes (~65% of known images)
    - Val: only Known classes (~15% of known images)
    - Test: contains BOTH Known classes (~20% of known images) AND ALL Unknown class images!

    Returns:
        (train_samples, val_samples, test_samples)
    """
    rng = random.Random(seed)

    train_samples: List[Tuple[Path, str]] = []
    val_samples: List[Tuple[Path, str]] = []
    test_samples: List[Tuple[Path, str]] = []

    # Partition Known classes
    for cls in known_classes:
        imgs = list(class_images.get(cls, []))
        rng.shuffle(imgs)
        n = len(imgs)

        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        n_test = max(1, int(n * test_known_ratio)) if n >= 4 else (1 if n >= 2 else 0)
        n_train = n - n_val - n_test

        if n_train <= 0:
            n_train = max(1, n - 1)
            n_val = 1 if n > 1 else 0
            n_test = 0

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        train_samples.extend([(p, cls) for p in train_imgs])
        val_samples.extend([(p, cls) for p in val_imgs])
        test_samples.extend([(p, cls) for p in test_imgs])

    # All unknown classes go exclusively into the test set!
    for cls in unknown_classes:
        for p in class_images.get(cls, []):
            test_samples.append((p, cls))

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def create_osr_dataloaders(
    data_dir: Union[str, Path],
    known_classes: Optional[Sequence[str]] = None,
    num_known: Optional[int] = None,
    known_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 2,
    target_size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], List[str], List[str]]:
    """
    Create ready-to-use PyTorch DataLoaders for Open Set Recognition.

    Returns:
        (train_loader, val_loader, test_loader, class_to_idx, known_classes, unknown_classes)
    """
    class_images = scan_dataset_directory(data_dir)
    all_classes = sorted(class_images.keys())

    if not all_classes:
        raise ValueError(f"No class folders with images found in: {data_dir}")

    known, unknown = split_known_unknown_classes(
        all_classes,
        known_classes=known_classes,
        num_known=num_known,
        known_ratio=known_ratio,
        seed=seed,
    )

    class_to_idx = {cls: idx for idx, cls in enumerate(known)}

    train_samples, val_samples, test_samples = partition_train_val_test_samples(
        class_images, known, unknown, seed=seed
    )

    train_transform = BlackBackgroundTransforms(target_size=target_size, is_training=True)
    eval_transform = BlackBackgroundTransforms(target_size=target_size, is_training=False)

    train_dataset = OSRImageDataset(train_samples, class_to_idx, transform=train_transform)
    val_dataset = OSRImageDataset(val_samples, class_to_idx, transform=eval_transform)
    test_dataset = OSRImageDataset(test_samples, class_to_idx, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, class_to_idx, known, unknown
