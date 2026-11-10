#!/usr/bin/env python3
"""
End-to-End Object Matcher Pipeline
==================================
Fulfills the core workflow:
1. Extract object from raw image and convert background to pure black (0, 0, 0).
2. Feed black-background object into trained metric matcher model.
3. Compute cosine similarities against class prototypes.
4. Output match prediction (Known Class with confidence, or Rejection as Unknown).
5. Generate side-by-side visual diagnostic artifact.

Author: Antigravity Assistant (for MTech Thesis OSR Project)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from bg_removal import BackgroundRemover, PipelineConfig
from dataset import BlackBackgroundTransforms
from dataset_preprocessor import crop_and_pad_square
from models import OSRObjectMatcherModel, build_model


@dataclass
class MatchCandidate:
    """A candidate class match with its similarity score."""
    class_name: str
    class_idx: int
    similarity: float       # Raw cosine similarity [-1, 1]
    confidence_pct: float   # Normalized score [0, 100]%


@dataclass
class MatchResult:
    """Full diagnostic result of end-to-end object matching."""
    input_path: Optional[str]
    is_known: bool
    predicted_class: str
    confidence_score: float
    confidence_pct: float
    threshold: float
    top_candidates: List[MatchCandidate]
    raw_image: Image.Image
    black_bg_image: Image.Image
    alpha_mask: Image.Image
    latency_ms: float

    def summary_text(self) -> str:
        """Textual summary of the match decision."""
        status = "MATCH ACCEPTED" if self.is_known else "REJECTED (OPEN-SET UNKNOWN)"
        lines = [
            f"Decision: {status}",
            f"Predicted Class: '{self.predicted_class}' (Confidence: {self.confidence_pct:.1f}%, Threshold: {self.threshold*100:.1f}%)",
            "Top Candidates:",
        ]
        for c in self.top_candidates:
            lines.append(f"  • {c.class_name:<20}: {c.confidence_pct:5.1f}% (Cosine Sim: {c.similarity:+.3f})")
        return "\n".join(lines)


class ObjectMatcher:
    """
    End-to-end inference engine:
    Raw Image -> Object Extraction + Black BG -> Feature Extraction -> Prototype Matching.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        remover: Optional[BackgroundRemover] = None,
        threshold: Optional[float] = None,
        crop_object: bool = True,
        target_size: Tuple[int, int] = (224, 224),
        device_name: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()

        # Load configs and mappings
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in checkpoint directory: {self.checkpoint_dir}")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        with open(self.checkpoint_dir / "class_to_idx.json", "r") as f:
            self.class_to_idx = json.load(f)

        self.idx_to_class = {int(k): v for k, v in self.config.get("idx_to_class", {}).items()}
        if not self.idx_to_class:
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.num_classes = len(self.class_to_idx)
        self.target_size = tuple(self.config.get("target_size", target_size))
        self.crop_object = crop_object

        # Determine threshold
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self.config.get("calibrated_thresholds", {}).get("tau_recommended", 0.5)

        # Resolve device
        if device_name:
            self.device = torch.device(device_name)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load Model
        self.model = build_model(
            num_classes=self.num_classes,
            backbone=self.config["backbone"],
            feature_dim=self.config["feature_dim"],
            temperature=self.config["temperature"],
        ).to(self.device)

        model_weights = self.checkpoint_dir / "best_model.pth"
        self.model.load_state_dict(torch.load(model_weights, map_location=self.device))

        # Load empirical prototypes
        proto_path = self.checkpoint_dir / "prototypes.pt"
        if proto_path.exists():
            prototypes = torch.load(proto_path, map_location=self.device)
            self.model.empirical_prototypes.copy_(prototypes)
            self.model.has_empirical_prototypes.copy_(torch.tensor(True))

        self.model.eval()

        # Initialize Background Remover
        self.remover = remover or BackgroundRemover()
        self.transform = BlackBackgroundTransforms(target_size=self.target_size, is_training=False)

    def match(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """
        Execute end-to-end object matching on an input image.

        1. Extract foreground object and composite onto black background (0, 0, 0).
        2. Crop to object bounding box with black square padding.
        3. Extract normalized embedding and compute prototype similarity.
        4. Decide Known match vs Open-Set rejection.
        """
        import time
        t0 = time.time()
        active_threshold = threshold if threshold is not None else self.threshold

        input_path_str = str(image_input) if isinstance(image_input, (str, Path)) else None

        # Step 1: Remove background and composite onto pure black
        black_bg_raw, alpha_mask, _ = self.remover.remove_background(
            image_input,
            background_color=(0, 0, 0),
            return_latency=False,
        )

        # Step 2: Bounding-box Square Cropping
        if self.crop_object:
            black_bg_square, _ = crop_and_pad_square(
                black_bg_raw,
                alpha_mask,
                target_size=self.target_size,
                padding=0.1,
                bg_color=(0, 0, 0),
            )
        else:
            black_bg_square = black_bg_raw.resize(self.target_size, Image.Resampling.BILINEAR)

        # Step 3: PyTorch Model Inference
        tensor = self.transform(black_bg_square).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.model.extract_embedding(tensor, normalize=True)
            similarities = self.model.compute_similarity_to_prototypes(z).squeeze(0)  # [num_classes]

            # Cosine in [-1, 1] mapped to confidence [0, 1]
            cosine_scores = (similarities + 1.0) / 2.0

            sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()

            candidates: List[MatchCandidate] = []
            for idx in sorted_indices[:top_k]:
                idx = int(idx)
                sim = float(similarities[idx].item())
                score = float(cosine_scores[idx].item())
                cname = self.idx_to_class.get(idx, f"Class_{idx}")
                candidates.append(
                    MatchCandidate(
                        class_name=cname,
                        class_idx=idx,
                        similarity=sim,
                        confidence_pct=score * 100.0,
                    )
                )

            top_candidate = candidates[0]
            top_score = top_candidate.confidence_pct / 100.0

            is_known = bool(top_score >= active_threshold)
            predicted_class = top_candidate.class_name if is_known else "UNKNOWN"

        latency_ms = (time.time() - t0) * 1000.0

        # Load raw image for visualization if given as path
        if isinstance(image_input, (str, Path)):
            raw_img = Image.open(image_input).convert("RGB")
        else:
            raw_img = image_input.convert("RGB")

        return MatchResult(
            input_path=input_path_str,
            is_known=is_known,
            predicted_class=predicted_class,
            confidence_score=top_score,
            confidence_pct=top_score * 100.0,
            threshold=active_threshold,
            top_candidates=candidates,
            raw_image=raw_img,
            black_bg_image=black_bg_square,
            alpha_mask=alpha_mask,
            latency_ms=latency_ms,
        )

    def create_visualization(
        self,
        result: MatchResult,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        Generate a multi-panel visual report:
        [Raw Input] | [Black BG Object] | [Alpha Mask] | [Match Decision & Similarity Bars]
        """
        panel_w, panel_h = 320, 320
        banner_h = 70
        total_w = panel_w * 4
        total_h = panel_h + banner_h

        canvas = Image.new("RGB", (total_w, total_h), (24, 25, 29))
        draw = ImageDraw.Draw(canvas)

        # Banner color: Green if Known Match, Amber/Red if Unknown Rejection
        banner_color = (34, 139, 34) if result.is_known else (178, 34, 34)
        draw.rectangle([(0, 0), (total_w, banner_h)], fill=banner_color)

        # Status text
        status_txt = f"MATCH: '{result.predicted_class}' ({result.confidence_pct:.1f}%)" if result.is_known else f"REJECTED: UNKNOWN OBJECT (Top: {result.top_candidates[0].class_name} @ {result.confidence_pct:.1f}%)"
        sub_txt = f"Decision Threshold: {result.threshold*100:.1f}% | Inference Latency: {result.latency_ms:.1f}ms"
        draw.text((20, 15), status_txt, fill=(255, 255, 255))
        draw.text((20, 42), sub_txt, fill=(230, 230, 230))

        y_offset = banner_h

        # Panel 1: Raw Image
        p1 = result.raw_image.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        canvas.paste(p1, (0, y_offset))
        draw.rectangle([(0, y_offset), (panel_w, y_offset + 25)], fill=(0, 0, 0, 180))
        draw.text((10, y_offset + 5), "1. Raw Query Image", fill=(255, 255, 255))

        # Panel 2: Black BG Object
        p2 = result.black_bg_image.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        canvas.paste(p2, (panel_w, y_offset))
        draw.rectangle([(panel_w, y_offset), (panel_w * 2, y_offset + 25)], fill=(0, 0, 0, 180))
        draw.text((panel_w + 10, y_offset + 5), "2. Extracted Black BG", fill=(255, 255, 255))

        # Panel 3: Alpha Mask
        p3 = result.alpha_mask.convert("RGB").resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        canvas.paste(p3, (panel_w * 2, y_offset))
        draw.rectangle([(panel_w * 2, y_offset), (panel_w * 3, y_offset + 25)], fill=(0, 0, 0, 180))
        draw.text((panel_w * 2 + 10, y_offset + 5), "3. Segmentation Mask", fill=(255, 255, 255))

        # Panel 4: Match Diagnostic Bars
        chart_x = panel_w * 3
        draw.rectangle([(chart_x, y_offset), (total_w, total_h)], fill=(32, 34, 40))
        draw.text((chart_x + 15, y_offset + 10), "4. Prototype Similarity Match", fill=(255, 255, 255))

        bar_y = y_offset + 45
        bar_max_w = panel_w - 90

        for cand in result.top_candidates[:5]:
            bar_len = int((cand.confidence_pct / 100.0) * bar_max_w)
            is_top = (cand == result.top_candidates[0])
            b_color = (65, 140, 240) if not is_top else ((50, 205, 50) if result.is_known else (220, 20, 60))

            # Label
            draw.text((chart_x + 15, bar_y), cand.class_name[:12], fill=(220, 220, 220))
            # Bar background
            draw.rectangle([(chart_x + 95, bar_y + 2), (chart_x + 95 + bar_max_w, bar_y + 14)], fill=(55, 58, 68))
            # Filled bar
            if bar_len > 0:
                draw.rectangle([(chart_x + 95, bar_y + 2), (chart_x + 95 + bar_len, bar_y + 14)], fill=b_color)
            # Value text
            draw.text((chart_x + 100 + bar_max_w, bar_y), f"{cand.confidence_pct:.1f}%", fill=(200, 200, 200))
            bar_y += 35

        # Draw threshold marker line across the bars
        th_pos_x = chart_x + 95 + int(result.threshold * bar_max_w)
        draw.line([(th_pos_x, y_offset + 42), (th_pos_x, bar_y - 10)], fill=(255, 215, 0), width=2)
        draw.text((th_pos_x - 30, bar_y - 5), f"Tau={result.threshold*100:.0f}%", fill=(255, 215, 0))

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out_p)
            print(f"✔ Saved visual match report to: {out_p}")

        return canvas
