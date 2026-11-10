#!/usr/bin/env python3
"""
Modular PyTorch Background Removal Pipeline
===========================================
A production-ready, modular PyTorch implementation for removing backgrounds
using state-of-the-art vision models from Hugging Face (such as BRIA RMBG-1.4,
RMBG-2.0, or DIS/IS-Net).

Author: Antigravity Assistant
License: MIT
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # Will be validated inside BackgroundRemover._init_dependencies()

from PIL import Image, ImageFilter, ImageOps


# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclass
class PipelineConfig:
    """Configuration hyperparameters for the background removal pipeline."""
    # Model parameters
    model_id: str = "briaai/RMBG-1.4"
    device: Optional[str] = None             # "cuda", "mps", "cpu", or None (auto-detect)
    dtype: Optional[str] = None              # "float32", "float16", "bfloat16", or None (auto)
    trust_remote_code: bool = True

    # Preprocessing parameters
    input_size: Tuple[int, int] = (1024, 1024)   # (width, height) expected by model
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Post-processing parameters
    hard_threshold: Optional[float] = None       # Optional binarization threshold (e.g., 0.5)
    soft_threshold_bounds: Optional[Tuple[float, float]] = None  # E.g., (0.05, 0.95) to cut noise
    blur_radius: float = 0.0                     # Gaussian blur radius for anti-aliasing edges
    invert_mask: bool = False                    # Invert foreground/background if needed

    # Benchmarking parameters
    warmup_runs: int = 3
    benchmark_runs: int = 10


@dataclass
class StageLatency:
    """Latency breakdown for individual inference stages in milliseconds."""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    composite_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def fps(self) -> float:
        """Calculate frames per second based on total latency."""
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark statistics aggregated across multiple runs."""
    device: str
    dtype: str
    model_id: str
    input_resolution: Tuple[int, int]
    original_resolution: Tuple[int, int]
    warmup_runs: int
    benchmark_runs: int
    latencies: List[StageLatency] = field(default_factory=list)

    @property
    def mean_latency(self) -> StageLatency:
        if not self.latencies:
            return StageLatency()
        n = len(self.latencies)
        return StageLatency(
            preprocess_ms=sum(l.preprocess_ms for l in self.latencies) / n,
            inference_ms=sum(l.inference_ms for l in self.latencies) / n,
            postprocess_ms=sum(l.postprocess_ms for l in self.latencies) / n,
            composite_ms=sum(l.composite_ms for l in self.latencies) / n,
            total_ms=sum(l.total_ms for l in self.latencies) / n,
        )

    @property
    def min_total_ms(self) -> float:
        return min((l.total_ms for l in self.latencies), default=0.0)

    @property
    def max_total_ms(self) -> float:
        return max((l.total_ms for l in self.latencies), default=0.0)

    @property
    def std_total_ms(self) -> float:
        if len(self.latencies) < 2:
            return 0.0
        totals = [l.total_ms for l in self.latencies]
        mean = sum(totals) / len(totals)
        variance = sum((x - mean) ** 2 for x in totals) / (len(totals) - 1)
        return float(math.sqrt(variance))

    def print_summary(self) -> None:
        """Print a formatted benchmark report table."""
        avg = self.mean_latency
        line = "─" * 60
        print(f"\n{line}")
        print(f"  BACKGROUND REMOVAL INFERENCE BENCHMARK REPORT")
        print(f"{line}")
        print(f"  Model ID          : {self.model_id}")
        print(f"  Compute Device    : {self.device}")
        print(f"  Precision (dtype) : {self.dtype}")
        print(f"  Original Image Res: {self.original_resolution[0]}x{self.original_resolution[1]}")
        print(f"  Model Input Res   : {self.input_resolution[0]}x{self.input_resolution[1]}")
        print(f"  Warmup Runs       : {self.warmup_runs}")
        print(f"  Benchmark Iter    : {self.benchmark_runs}")
        print(f"{line}")
        print(f"  STAGE LATENCIES (Mean over {self.benchmark_runs} runs):")
        print(f"    • Preprocessing   : {avg.preprocess_ms:7.2f} ms ({avg.preprocess_ms/avg.total_ms*100:5.1f}%)")
        print(f"    • Model Inference : {avg.inference_ms:7.2f} ms ({avg.inference_ms/avg.total_ms*100:5.1f}%)")
        print(f"    • Post-processing : {avg.postprocess_ms:7.2f} ms ({avg.postprocess_ms/avg.total_ms*100:5.1f}%)")
        print(f"    • Alpha Composite : {avg.composite_ms:7.2f} ms ({avg.composite_ms/avg.total_ms*100:5.1f}%)")
        print(f"  {line}")
        print(f"  TOTAL END-TO-END:")
        print(f"    • Mean Latency    : {avg.total_ms:7.2f} ms ± {self.std_total_ms:.2f} ms")
        print(f"    • Min / Max       : {self.min_total_ms:7.2f} ms / {self.max_total_ms:.2f} ms")
        print(f"    • Throughput      : {avg.fps:7.2f} FPS")
        print(f"{line}\n")


# ==============================================================================
# Background Remover Core Class
# ==============================================================================

class BackgroundRemover:
    """
    Modular Background Remover utilizing Hugging Face vision models.
    
    Encapsulates:
    1. Model weight loading and device placement.
    2. Input preprocessing and normalization.
    3. Neural forward pass in inference mode.
    4. Mask post-processing, upscaling, thresholding, and smoothing.
    5. RGBA alpha compositing.
    6. Stage-by-stage latency benchmarking.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Lazy imports for torch and transformers to keep module importable
        # and provide informative error messages if dependencies are missing.
        self._init_dependencies()
        
        self.device = self._resolve_device(self.config.device)
        self.torch_dtype = self._resolve_dtype(self.config.dtype, self.device)
        self.model = self._load_model()

    def _init_dependencies(self) -> None:
        """Validate required PyTorch and Transformers libraries."""
        import site
        user_site = site.getusersitepackages()
        if os.path.exists(user_site) and user_site not in sys.path:
            sys.path.insert(0, user_site)

        global np
        try:
            if np is None:
                import numpy as np
            import torch
            import torch.nn.functional as F
            from transformers import AutoModelForImageSegmentation, PreTrainedModel
            
            # Compatibility shim for transformers >= 5.x with custom remote code models (like BriaRMBG)
            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                PreTrainedModel.all_tied_weights_keys = property(
                    lambda s: s.__dict__.get("all_tied_weights_keys", {}),
                    lambda s, v: s.__dict__.__setitem__("all_tied_weights_keys", v)
                )

            self.torch = torch
            self.F = F
            self.AutoModelForImageSegmentation = AutoModelForImageSegmentation
        except ImportError as e:
            raise ImportError(
                f"Missing required dependency: {e.name}. "
                f"Please install dependencies via: pip install torch torchvision transformers accelerate numpy pillow"
            ) from e

    def _resolve_device(self, requested_device: Optional[str]) -> "torch.device":
        """Resolve target computation device (CUDA -> MPS -> CPU)."""
        torch = self.torch
        if requested_device:
            return torch.device(requested_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self, requested_dtype: Optional[str], device: "torch.device") -> "torch.dtype":
        """Resolve precision dtype based on user preference and device capability."""
        torch = self.torch
        if requested_dtype:
            mapping = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            if requested_dtype.lower() in mapping:
                return mapping[requested_dtype.lower()]
            raise ValueError(f"Unsupported dtype: '{requested_dtype}'. Choose from float32, float16, bfloat16.")

        # Default dtype: use float16 on CUDA for performance, float32 on CPU/MPS
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    def _load_model(self):
        """Download model weights from Hugging Face Hub and initialize pipeline."""
        print(f"[BackgroundRemover] Initializing model '{self.config.model_id}' on {self.device} ({self.torch_dtype})...")
        model = self.AutoModelForImageSegmentation.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code
        )
        model.to(device=self.device, dtype=self.torch_dtype)
        model.eval()
        print(f"[BackgroundRemover] Model successfully loaded and ready for inference.")
        return model

    def _synchronize_device(self) -> None:
        """Synchronize CUDA/device queue to ensure precise benchmarking."""
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    # --------------------------------------------------------------------------
    # 1. Preprocessing
    # --------------------------------------------------------------------------
    def preprocess(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple["torch.Tensor", Tuple[int, int], Image.Image]:
        """
        Preprocess input image for neural network forward pass.

        Steps:
        1. Open/convert image to RGB with EXIF orientation correction.
        2. Record original dimensions (width, height).
        3. Resize image to model's expected input resolution (e.g. 1024x1024).
        4. Normalize pixels to tensor in range [0, 1] followed by channel normalization.
        5. Add batch dimension and transfer to computation device.

        Returns:
            tensor: [1, 3, H, W] normalized tensor on device.
            orig_size: (width, height) tuple of original image.
            orig_rgb: PIL.Image in RGB mode.
        """
        # Load and normalize PIL image
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(str(image_input))
        elif isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Correct EXIF rotation if present and convert to RGB
        pil_image = ImageOps.exif_transpose(pil_image)
        orig_rgb = pil_image.convert("RGB")
        orig_size = orig_rgb.size  # (width, height)

        # High-quality bilinear resize to model input size
        resized_pil = orig_rgb.resize(self.config.input_size, Image.Resampling.BILINEAR)

        # Convert to numpy array [H, W, C] in range [0, 1]
        np_img = np.array(resized_pil, dtype=np.float32) / 255.0

        # Channel normalization: (x - mean) / std
        mean = np.array(self.config.normalize_mean, dtype=np.float32)
        std = np.array(self.config.normalize_std, dtype=np.float32)
        np_img = (np_img - mean) / std

        # Convert to PyTorch tensor [C, H, W] and add batch dimension [1, C, H, W]
        tensor = self.torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.torch_dtype)

        return tensor, orig_size, orig_rgb

    # --------------------------------------------------------------------------
    # 2. Forward Pass
    # --------------------------------------------------------------------------
    def forward_pass(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """
        Run forward pass in inference mode and extract normalized alpha mask tensor.

        Handles:
        - Inference mode (zero autograd tracking overhead).
        - Multi-output architectures (e.g. BRIA RMBG returns list/tuple of multi-scale masks).
        - Min-max normalization of raw output activations into [0.0, 1.0].
        """
        with self.torch.inference_mode():
            outputs = self.model(tensor)

        # Extract primary mask from model output across various model architectures
        raw_mask = outputs
        if isinstance(raw_mask, dict):
            raw_mask = raw_mask.get("logits", raw_mask.get("pred_masks", next(iter(raw_mask.values()))))
        elif hasattr(raw_mask, "logits"):
            raw_mask = raw_mask.logits

        # Unpack nested lists/tuples (e.g. BRIA RMBG returns tuple of multi-stage lists: outputs[0][0])
        while isinstance(raw_mask, (list, tuple)):
            raw_mask = raw_mask[0]

        # Ensure mask is [Batch, 1, H, W]
        if raw_mask.dim() == 3:
            raw_mask = raw_mask.unsqueeze(1)
        elif raw_mask.dim() == 2:
            raw_mask = raw_mask.unsqueeze(0).unsqueeze(0)

        # Normalize raw activations to [0.0, 1.0] using min-max scaling
        ma = self.torch.max(raw_mask)
        mi = self.torch.min(raw_mask)
        if ma > mi:
            normalized_mask = (raw_mask - mi) / (ma - mi)
        else:
            normalized_mask = self.torch.zeros_like(raw_mask)

        return normalized_mask

    # --------------------------------------------------------------------------
    # 3. Mask Post-processing
    # --------------------------------------------------------------------------
    def postprocess_mask(
        self,
        raw_mask: "torch.Tensor",
        orig_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Post-process raw mask tensor to generate a clean, precise 8-bit alpha mask.

        Steps:
        1. Bilinear interpolation directly on tensor to original (width, height).
        2. Optional soft thresholding (noise clamp) or hard binarization.
        3. Convert to 8-bit grayscale PIL Image (mode 'L').
        4. Optional Gaussian blur edge smoothing / feathering.
        """
        orig_w, orig_h = orig_size

        # Tensor-accelerated bilinear upscaling back to original resolution
        interpolated = self.F.interpolate(
            raw_mask.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        interpolated = self.torch.clamp(interpolated, 0.0, 1.0)

        # Optional soft thresholding (suppresses background noise and saturates foreground)
        if self.config.soft_threshold_bounds is not None:
            low, high = self.config.soft_threshold_bounds
            interpolated = self.torch.clamp((interpolated - low) / (high - low), 0.0, 1.0)

        # Optional hard thresholding
        if self.config.hard_threshold is not None:
            interpolated = (interpolated >= self.config.hard_threshold).float()

        # Invert mask if requested
        if self.config.invert_mask:
            interpolated = 1.0 - interpolated

        # Convert to 8-bit grayscale numpy array [H, W] and PIL Image with proper rounding
        mask_array = np.round(interpolated[0, 0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_array, mode="L")

        # Apply Gaussian blur smoothing / edge feathering if radius > 0
        if self.config.blur_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=self.config.blur_radius))

        return mask_pil

    # --------------------------------------------------------------------------
    # 4. Alpha Compositing
    # --------------------------------------------------------------------------
    def composite(
        self,
        orig_rgb: Image.Image,
        alpha_mask: Image.Image,
        background_color: Optional[Tuple[int, int, int]] = None
    ) -> Image.Image:
        """
        Composite the segmented foreground with transparency or a solid background.

        Args:
            orig_rgb: Original RGB PIL image.
            alpha_mask: 8-bit grayscale PIL image representing alpha channel.
            background_color: Optional RGB tuple (e.g. (255, 255, 255) for white).
                             If None, returns a 4-channel transparent RGBA image.

        Returns:
            RGBA PIL image (if background_color is None) or RGB PIL image (if background_color is set).
        """
        rgba_image = orig_rgb.convert("RGBA")
        rgba_image.putalpha(alpha_mask)

        if background_color is not None:
            solid_bg = Image.new("RGBA", orig_rgb.size, background_color + (255,))
            composited = Image.alpha_composite(solid_bg, rgba_image)
            return composited.convert("RGB")

        return rgba_image

    # --------------------------------------------------------------------------
    # 5. End-to-End Execution & Benchmarking
    # --------------------------------------------------------------------------
    def remove_background(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        background_color: Optional[Tuple[int, int, int]] = None,
        return_latency: bool = False
    ) -> Tuple[Image.Image, Image.Image, Optional[StageLatency]]:
        """
        Execute end-to-end background removal pipeline on a single image.

        Returns:
            (rgba_or_composited_image, alpha_mask, Optional[StageLatency])
        """
        latency = StageLatency() if return_latency else None

        # --- Stage 1: Preprocessing ---
        self._synchronize_device()
        t0 = time.perf_counter()
        tensor, orig_size, orig_rgb = self.preprocess(image_input)
        self._synchronize_device()
        t1 = time.perf_counter()
        if latency:
            latency.preprocess_ms = (t1 - t0) * 1000.0

        # --- Stage 2: Inference ---
        t2 = time.perf_counter()
        raw_mask = self.forward_pass(tensor)
        self._synchronize_device()
        t3 = time.perf_counter()
        if latency:
            latency.inference_ms = (t3 - t2) * 1000.0

        # --- Stage 3: Mask Post-processing ---
        t4 = time.perf_counter()
        alpha_mask = self.postprocess_mask(raw_mask, orig_size)
        t5 = time.perf_counter()
        if latency:
            latency.postprocess_ms = (t5 - t4) * 1000.0

        # --- Stage 4: Compositing ---
        t6 = time.perf_counter()
        output_image = self.composite(orig_rgb, alpha_mask, background_color)
        t7 = time.perf_counter()
        if latency:
            latency.composite_ms = (t7 - t6) * 1000.0
            latency.total_ms = (t7 - t0) * 1000.0

        return output_image, alpha_mask, latency

    def benchmark(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        warmup_runs: Optional[int] = None,
        benchmark_runs: Optional[int] = None
    ) -> BenchmarkReport:
        """
        Conduct a multi-iteration latency benchmark with GPU synchronization.
        """
        warmup = warmup_runs if warmup_runs is not None else self.config.warmup_runs
        runs = benchmark_runs if benchmark_runs is not None else self.config.benchmark_runs

        print(f"\n[Benchmark] Warming up pipeline ({warmup} runs)...")
        for _ in range(warmup):
            self.remove_background(image_input, return_latency=False)

        print(f"[Benchmark] Executing benchmark across {runs} iterations...")
        latencies: List[StageLatency] = []
        for i in range(runs):
            _, _, lat = self.remove_background(image_input, return_latency=True)
            if lat:
                latencies.append(lat)

        # Query sample properties
        _, orig_size, _ = self.preprocess(image_input)

        report = BenchmarkReport(
            device=str(self.device),
            dtype=str(self.torch_dtype).replace("torch.", ""),
            model_id=self.config.model_id,
            input_resolution=self.config.input_size,
            original_resolution=orig_size,
            warmup_runs=warmup,
            benchmark_runs=runs,
            latencies=latencies,
        )
        report.print_summary()
        return report


# ==============================================================================
# Command Line Interface (CLI)
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular PyTorch Background Removal Pipeline using Pre-trained Vision Models (BRIA RMBG, DIS/IS-Net).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input / Output
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image or directory of images.")
    parser.add_argument("-o", "--output", type=str, default="output.png", help="Path to save output RGBA image (or output folder if input is directory).")
    parser.add_argument("--save-mask", type=str, default=None, help="Optional path to save isolated grayscale alpha mask.")

    # Model & Hardware
    parser.add_argument("--model-id", type=str, default="briaai/RMBG-1.4", help="Hugging Face model repository ID.")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="Compute device (default: auto-detect).")
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"], help="Precision dtype (default: float16 on CUDA, float32 on CPU).")
    parser.add_argument("--input-size", type=int, nargs=2, default=[1024, 1024], metavar=("WIDTH", "HEIGHT"), help="Input resolution for model tensor.")

    # Post-processing
    parser.add_argument("--threshold", type=float, default=None, help="Optional hard binarization threshold (0.0 - 1.0).")
    parser.add_argument("--soft-threshold", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="Soft threshold clamp bounds (e.g. 0.05 0.95) to suppress noise.")
    parser.add_argument("--blur", type=float, default=0.0, help="Gaussian blur radius for mask edge smoothing / feathering.")
    parser.add_argument("--bg-color", type=int, nargs=3, default=None, metavar=("R", "G", "B"), help="Optional solid background color (e.g. 255 255 255 for white). Default produces transparent RGBA.")

    # Benchmarking
    parser.add_argument("--benchmark", action="store_true", help="Run multi-iteration latency benchmark and display detailed statistics.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark iterations.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build Pipeline Configuration
    config = PipelineConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        input_size=tuple(args.input_size),
        hard_threshold=args.threshold,
        soft_threshold_bounds=tuple(args.soft_threshold) if args.soft_threshold else None,
        blur_radius=args.blur,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
    )

    # Initialize Pipeline
    remover = BackgroundRemover(config)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    bg_color = tuple(args.bg_color) if args.bg_color is not None else None

    # Handle Directory or Single File
    if input_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        if not image_files:
            print(f"No supported images found in '{input_path}'.", file=sys.stderr)
            sys.exit(1)

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {len(image_files)} images from '{input_path}' -> '{out_dir}'...")

        for idx, img_file in enumerate(image_files, 1):
            out_file = out_dir / f"{img_file.stem}_nobg.png"
            rgba, mask, lat = remover.remove_background(img_file, background_color=bg_color, return_latency=True)
            rgba.save(out_file)
            print(f"[{idx}/{len(image_files)}] {img_file.name} -> {out_file.name} (Latency: {lat.total_ms:.1f}ms, {lat.fps:.1f} FPS)")

    else:
        # Benchmark mode if requested
        if args.benchmark:
            remover.benchmark(input_path, warmup_runs=args.warmup, benchmark_runs=args.runs)

        # Process and save output image
        print(f"Processing '{input_path}'...")
        rgba, mask, lat = remover.remove_background(input_path, background_color=bg_color, return_latency=True)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(out_path)
        print(f"✔ Saved transparent cutout image to: {out_path}")

        if args.save_mask:
            mask_path = Path(args.save_mask)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(mask_path)
            print(f"✔ Saved alpha mask to: {mask_path}")

        if lat and not args.benchmark:
            print(f"Latency: Total={lat.total_ms:.2f}ms (Pre={lat.preprocess_ms:.1f}ms, Inf={lat.inference_ms:.1f}ms, Post={lat.postprocess_ms:.1f}ms, Comp={lat.composite_ms:.1f}ms) | {lat.fps:.1f} FPS")


if __name__ == "__main__":
    main()
