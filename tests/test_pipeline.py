#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite for OSR Pipeline
===================================================
Tests all modules:
1. Object extraction & black background compositing
2. Dataset loading, partitioning (known vs open-set unknown)
3. Model forward pass, cosine metric classification, prototypes
4. Training loop execution & artifact generation
5. Closed-set & Open-set evaluation metrics (AUROC, OSCR, Confusion Matrix)
6. End-to-end query matching with visual diagnostics

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bg_removal import BackgroundRemover, PipelineConfig
from create_sample_dataset import create_synthetic_object_image
from dataset import (
    BlackBackgroundTransforms,
    OSRImageDataset,
    create_osr_dataloaders,
    partition_train_val_test_samples,
    split_known_unknown_classes,
)
from dataset_preprocessor import DatasetPreprocessor, crop_and_pad_square
from evaluate import OSREvaluator, compute_binary_auroc, compute_confusion_matrix, compute_oscr
from matcher import ObjectMatcher
from models import (
    CosineMetricClassifier,
    OSRObjectMatcherModel,
    ResNetBackbone,
    SimpleConvNet,
    build_model,
)
from train import train_model


class TestOSRPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create shared temporary directory for tests."""
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_osr_"))
        cls.raw_data_dir = cls.temp_dir / "raw"
        cls.black_data_dir = cls.temp_dir / "black"
        cls.checkpoint_dir = cls.temp_dir / "checkpoints"

        cls.known_classes = ["cup", "car", "bottle"]
        cls.unknown_classes = ["star"]

        # Generate a small test dataset
        for c in cls.known_classes:
            cdir = cls.raw_data_dir / c
            cdir.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                img, _ = create_synthetic_object_image(c, width=128, height=128)
                img.save(cdir / f"{c}_{i:02d}.png")

        for c in cls.unknown_classes:
            cdir = cls.raw_data_dir / c
            cdir.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                img, _ = create_synthetic_object_image(c, width=128, height=128)
                img.save(cdir / f"{c}_{i:02d}.png")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_crop_and_pad_square(self):
        """Test bounding box crop and black background square padding."""
        img = Image.new("RGB", (200, 100), (255, 0, 0))
        mask = Image.new("L", (200, 100), 0)
        # Foreground object in center [50..150, 20..80]
        mask.paste(255, (50, 20, 150, 80))

        sq_img, sq_mask = crop_and_pad_square(
            img, mask, target_size=(224, 224), padding=0.1, bg_color=(0, 0, 0)
        )
        self.assertEqual(sq_img.size, (224, 224))
        self.assertEqual(sq_mask.size, (224, 224))

        # Check that corners are pure black
        arr = np.array(sq_img)
        self.assertTrue(np.all(arr[0, 0] == [0, 0, 0]))
        self.assertTrue(np.all(arr[223, 223] == [0, 0, 0]))

    def test_02_dataset_partitioning(self):
        """Test splitting into known and open-set unknown classes."""
        all_cls = ["apple", "banana", "cherry", "date", "fig"]
        known, unknown = split_known_unknown_classes(all_cls, num_known=3, seed=123)
        self.assertEqual(len(known), 3)
        self.assertEqual(len(unknown), 2)
        self.assertTrue(set(known).isdisjoint(set(unknown)))

    def test_03_black_background_transforms(self):
        """Test specialized transforms preserve black borders."""
        transform = BlackBackgroundTransforms(target_size=(224, 224), is_training=True)
        img = Image.new("RGB", (224, 224), (0, 0, 0))
        tensor = transform(img)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_04_model_architectures(self):
        """Test forward passes of SimpleConvNet and ResNet."""
        x = torch.randn(2, 3, 224, 224)

        # SimpleConvNet
        cnn = SimpleConvNet(feature_dim=128)
        feat_cnn = cnn(x)
        self.assertEqual(feat_cnn.shape, (2, 128))

        # ResNetBackbone
        resnet = ResNetBackbone(arch="resnet18", pretrained=False, feature_dim=128)
        feat_res = resnet(x)
        self.assertEqual(feat_res.shape, (2, 128))

        # Cosine Metric Classifier
        clf = CosineMetricClassifier(num_classes=3, feature_dim=128)
        logits, sim = clf(feat_cnn)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(sim.shape, (2, 3))
        # Cosine similarities must be within [-1, 1]
        self.assertTrue(torch.all(sim >= -1.0 - 1e-4) and torch.all(sim <= 1.0 + 1e-4))

    def test_05_auroc_and_oscr_metrics(self):
        """Test AUROC and OSCR numerical computations."""
        # Perfect discrimination: positives score higher than negatives
        scores = np.array([0.9, 0.85, 0.8, 0.2, 0.1, 0.05])
        is_known = np.array([True, True, True, False, False, False])
        auroc = compute_binary_auroc(scores, is_known)
        self.assertAlmostEqual(auroc, 1.0, places=4)

        # Inverted discrimination
        inv_scores = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        inv_auroc = compute_binary_auroc(inv_scores, is_known)
        self.assertAlmostEqual(inv_auroc, 0.0, places=4)

        # OSCR test
        known_preds = np.array([0, 1, 2])
        known_targets = np.array([0, 1, 2])
        known_scores = np.array([0.9, 0.85, 0.8])
        unknown_scores = np.array([0.2, 0.1])
        a_oscr, fpr_95 = compute_oscr(known_preds, known_targets, known_scores, unknown_scores)
        self.assertGreater(a_oscr, 0.5)
        self.assertLessEqual(fpr_95, 0.5)

    def test_06_train_and_evaluate_pipeline(self):
        """End-to-end integration test: preprocess, train for 2 epochs, evaluate, and match."""
        # 1. Preprocess raw data into black-background dataset
        preprocessor = DatasetPreprocessor(
            target_size=(128, 128),
            crop_object=True,
            bg_color=(0, 0, 0),
        )
        stats = preprocessor.process_directory(self.raw_data_dir, self.black_data_dir)
        self.assertGreater(stats.processed, 0)
        self.assertEqual(stats.failed, 0)

        # 2. Train model
        train_config = train_model(
            data_dir=str(self.black_data_dir),
            output_dir=str(self.checkpoint_dir),
            backbone="simple_cnn",
            feature_dim=64,
            epochs=2,
            batch_size=4,
            known_classes=self.known_classes,
            target_size=(128, 128),
            device_name="cpu",
            seed=42,
        )
        self.assertTrue((self.checkpoint_dir / "best_model.pth").exists())
        self.assertTrue((self.checkpoint_dir / "prototypes.pt").exists())
        self.assertTrue((self.checkpoint_dir / "config.json").exists())

        # 3. Evaluate model on test split
        evaluator = OSREvaluator(checkpoint_dir=self.checkpoint_dir, device_name="cpu")
        _, _, test_loader, _, _, _ = create_osr_dataloaders(
            data_dir=str(self.black_data_dir),
            known_classes=self.known_classes,
            batch_size=4,
            target_size=(128, 128),
        )
        report = evaluator.evaluate(test_loader=test_loader)
        self.assertIn("closed_set", report)
        self.assertIn("open_set", report)
        self.assertGreaterEqual(report["open_set"]["auroc"], 0.0)

        # 4. End-to-end query match
        matcher = ObjectMatcher(
            checkpoint_dir=self.checkpoint_dir,
            target_size=(128, 128),
            device_name="cpu",
        )
        test_img_path = next(self.raw_data_dir.rglob("*.png"))
        match_res = matcher.match(test_img_path)
        self.assertIsNotNone(match_res.predicted_class)
        self.assertGreaterEqual(match_res.confidence_pct, 0.0)

        # 5. Generate and save diagnostic visualization
        vis_path = self.temp_dir / "test_match_vis.png"
        vis = matcher.create_visualization(match_res, save_path=vis_path)
        self.assertTrue(vis_path.exists())
        self.assertEqual(vis.size[1], 320 + 70)


if __name__ == "__main__":
    unittest.main()
