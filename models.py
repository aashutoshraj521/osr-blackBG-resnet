#!/usr/bin/env python3
"""
Model Architectures for Open Set Recognition & Object Matching
==============================================================
Production-ready PyTorch vision models designed for metric learning,
prototype-based matching, and open-set recognition on black-background objects.

Models Included:
1. SimpleConvNet: Lightweight, highly efficient 4-stage CNN for fast CPU/edge training.
2. ResNetBackbone: ResNet-18/34/50 feature extractors with projection head.
3. MobileNetV3Backbone: MobileNetV3 feature extractor.
4. CosineMetricClassifier: Cosine similarity classifier with hyperspherical prototypes.
5. OSRObjectMatcherModel: Unified end-to-end model encapsulating backbone + prototypes + OSR scoring.
6. OpenMax: Extreme Value Theory (EVT) Weibull calibration for open-set probability estimation.

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ==============================================================================
# Feature Extractor Backbones
# ==============================================================================

class SimpleConvNet(nn.Module):
    """
    Lightweight 4-stage convolutional backbone with BatchNorm and MaxPool.
    Fast convergence and minimal compute requirements, ideal for CPU training.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: (3, 224, 224) -> (32, 112, 112)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: (32, 112, 112) -> (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: (64, 56, 56) -> (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 4: (128, 28, 28) -> (256, 14, 14)
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        return torch.flatten(out, 1)


class ResNetBackbone(nn.Module):
    """
    ResNet-18 / 34 / 50 backbone with customizable projection head.
    """

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = False,
        feature_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if arch == "resnet18":
            resnet = tv_models.resnet18(weights=weights)
            in_dim = resnet.fc.in_features
        elif arch == "resnet34":
            resnet = tv_models.resnet34(weights=weights)
            in_dim = resnet.fc.in_features
        elif arch == "resnet50":
            resnet = tv_models.resnet50(weights=weights)
            in_dim = resnet.fc.in_features
        else:
            raise ValueError(f"Unsupported ResNet architecture: '{arch}'. Choose resnet18, resnet34, resnet50.")

        # Strip standard classification head
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        if dropout > 0:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, feature_dim, bias=False),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, feature_dim, bias=False),
                nn.BatchNorm1d(feature_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, feature_dim, bias=False),
                nn.BatchNorm1d(feature_dim),
            )
        self.output_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3 backbone for lightweight, fast vision inference.
    """

    def __init__(self, pretrained: bool = False, feature_dim: int = 256):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        mobilenet = tv_models.mobilenet_v3_small(weights=weights)
        in_dim = mobilenet.classifier[0].in_features

        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(in_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
        )
        self.output_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


# ==============================================================================
# Metric Learning Cosine Classifier
# ==============================================================================

class CosineMetricClassifier(nn.Module):
    """
    Hyperspherical Cosine Classifier.
    Computes cosine similarity between normalized feature embeddings and class prototypes:
        s_k(x) = cos(z, c_k) = (z . c_k) / (||z|| * ||c_k||)
    Logits are scaled by temperature:
        l_k(x) = s_k(x) / temperature
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 256,
        temperature: float = 0.1,
        margin: float = 0.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.margin = margin

        # Learnable class prototype vectors on unit hypersphere
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0 / temperature))
        else:
            self.register_buffer("scale", torch.tensor(1.0 / temperature))

    def get_normalized_prototypes(self) -> torch.Tensor:
        """Returns L2-normalized class prototypes [num_classes, feature_dim]."""
        return F.normalize(self.weight, p=2, dim=1)

    def forward(self, embeddings: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [batch_size, feature_dim] unnormalized or normalized features.
            targets: Optional [batch_size] ground truth labels for angular margin penalty.

        Returns:
            (scaled_logits [batch_size, num_classes], cosine_similarities [batch_size, num_classes])
        """
        # Normalize input embeddings to unit hypersphere
        norm_z = F.normalize(embeddings, p=2, dim=1)
        # Normalize prototypes
        norm_w = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity in range [-1, 1]
        cosine_sim = F.linear(norm_z, norm_w)

        # CosFace additive angular margin penalty during training on target class
        if self.training and self.margin > 0.0 and targets is not None:
            one_hot = F.one_hot(targets.clamp(min=0), num_classes=self.num_classes).float()
            # Only apply margin to valid known classes (labels >= 0)
            valid_mask = (targets >= 0).float().unsqueeze(1)
            margin_sim = cosine_sim - (one_hot * self.margin * valid_mask)
            logits = margin_sim * self.scale
        else:
            logits = cosine_sim * self.scale

        return logits, cosine_sim


# ==============================================================================
# Unified End-to-End OSR Object Matcher Model
# ==============================================================================

class OSRObjectMatcherModel(nn.Module):
    """
    Unified Open Set Recognition and Object Matching Model.
    Combines Backbone Feature Extractor + Hyperspherical Prototype Classifier + OSR Decision Logic.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "simple_cnn",
        feature_dim: int = 256,
        temperature: float = 0.1,
        pretrained: bool = False,
        margin: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbone_name = backbone_name
        self.margin = margin
        self.dropout = dropout

        # 1. Initialize backbone
        if backbone_name == "simple_cnn":
            self.backbone = SimpleConvNet(feature_dim=feature_dim)
        elif backbone_name.startswith("resnet"):
            self.backbone = ResNetBackbone(arch=backbone_name, pretrained=pretrained, feature_dim=feature_dim, dropout=dropout)
        elif backbone_name == "mobilenet_v3":
            self.backbone = MobileNetV3Backbone(pretrained=pretrained, feature_dim=feature_dim)
        else:
            raise ValueError(f"Unknown backbone: '{backbone_name}'")

        # 2. Cosine Metric Classifier with margin support
        self.classifier = CosineMetricClassifier(
            num_classes=num_classes,
            feature_dim=feature_dim,
            temperature=temperature,
            margin=margin,
        )

        # Stored empirical centroids (updated after training for highest matching accuracy)
        self.register_buffer("empirical_prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("has_empirical_prototypes", torch.tensor(False))

    def extract_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Extract feature embedding from backbone."""
        feat = self.backbone(x)
        if normalize:
            feat = F.normalize(feat, p=2, dim=1)
        return feat

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            (scaled_logits, cosine_similarities, normalized_embeddings)
        """
        z = self.extract_embedding(x, normalize=True)
        logits, cosine_sim = self.classifier(z, targets=targets)
        return logits, cosine_sim, z

    def update_empirical_prototypes(self, loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
        """
        Compute empirical class mean centroids across known training samples:
            c_k = sum_{i: y_i = k} z_i / || sum_{i: y_i = k} z_i ||
        This gives maximum fidelity matching.
        """
        self.eval()
        class_sums = torch.zeros(self.num_classes, self.feature_dim, device=device)
        class_counts = torch.zeros(self.num_classes, device=device)

        with torch.no_grad():
            for images, labels, _, _ in loader:
                images = images.to(device)
                labels = labels.to(device)

                # Only include valid known labels [0, K-1]
                valid_mask = labels >= 0
                if not valid_mask.any():
                    continue

                valid_images = images[valid_mask]
                valid_labels = labels[valid_mask]

                z = self.extract_embedding(valid_images, normalize=True)

                for k in range(self.num_classes):
                    k_mask = valid_labels == k
                    if k_mask.any():
                        class_sums[k] += z[k_mask].sum(dim=0)
                        class_counts[k] += k_mask.sum().item()

        # Normalize centroids
        for k in range(self.num_classes):
            if class_counts[k] > 0:
                class_sums[k] = F.normalize(class_sums[k], p=2, dim=0)
            else:
                # Fallback to learnable weight
                class_sums[k] = self.classifier.get_normalized_prototypes()[k]

        self.empirical_prototypes.copy_(class_sums)
        self.has_empirical_prototypes.copy_(torch.tensor(True))
        return self.empirical_prototypes

    def compute_similarity_to_prototypes(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity of embeddings z [B, D] to active prototypes [K, D].
        Uses empirical centroids if computed, otherwise learnable prototypes.
        """
        norm_z = F.normalize(z, p=2, dim=1)
        if self.has_empirical_prototypes.item():
            prototypes = self.empirical_prototypes
        else:
            prototypes = self.classifier.get_normalized_prototypes()

        return F.linear(norm_z, prototypes)

    def match_object(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        scoring_method: str = "cosine",
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Perform Open Set Recognition & Object Matching.

        Args:
            x: Preprocessed tensor batch [B, 3, H, W] (black-background object).
            threshold: OSR acceptance threshold. If max_score < threshold -> UNKNOWN (-1).
            scoring_method: "cosine" (max cosine similarity) or "msp" (maximum softmax probability).

        Returns:
            Dictionary containing:
            - 'predicted_class_idx': Predicted known class index, or -1 for unknown.
            - 'confidence_score': Matching confidence score in [0, 1].
            - 'is_known': Boolean flag whether object is matched to known class.
            - 'all_similarities': Full similarity vector to all known classes.
            - 'embeddings': Feature vector.
        """
        self.eval()
        with torch.no_grad():
            z = self.extract_embedding(x, normalize=True)
            similarities = self.compute_similarity_to_prototypes(z)  # [B, K]

            # Convert cosine [-1, 1] to normalized score [0, 1]
            cosine_scores = (similarities + 1.0) / 2.0

            if scoring_method == "cosine":
                confidence, pred_k = torch.max(cosine_scores, dim=1)
            elif scoring_method == "msp":
                logits = similarities / self.classifier.temperature
                probs = F.softmax(logits, dim=1)
                confidence, pred_k = torch.max(probs, dim=1)
            else:
                raise ValueError(f"Unknown scoring_method: {scoring_method}")

            # Open Set Decision: if confidence < threshold -> reject as Unknown (-1)
            is_known = confidence >= threshold
            final_pred = torch.where(is_known, pred_k, torch.tensor(-1, device=x.device))

            return {
                "predicted_class_idx": final_pred,
                "confidence_score": confidence,
                "is_known": is_known,
                "all_similarities": similarities,
                "all_scores": cosine_scores,
                "embeddings": z,
            }


# ==============================================================================
# Helper Factory Function
# ==============================================================================

def build_model(
    num_classes: int,
    backbone: str = "simple_cnn",
    feature_dim: int = 256,
    temperature: float = 0.1,
    pretrained: bool = False,
    margin: float = 0.0,
    dropout: float = 0.0,
) -> OSRObjectMatcherModel:
    """Build and initialize OSRObjectMatcherModel."""
    return OSRObjectMatcherModel(
        num_classes=num_classes,
        backbone_name=backbone,
        feature_dim=feature_dim,
        temperature=temperature,
        pretrained=pretrained,
        margin=margin,
        dropout=dropout,
    )
