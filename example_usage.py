"""
Example usage of the modular BackgroundRemover class.
This demonstrates both Python programmatic API and how to use it in pipelines.
"""

from pathlib import Path
from PIL import Image
from bg_removal import BackgroundRemover, PipelineConfig


def example_single_image():
    # 1. Define configuration
    config = PipelineConfig(
        model_id="briaai/RMBG-1.4",    # Or "briaai/RMBG-2.0"
        device=None,                    # Auto-detects cuda -> mps -> cpu
        dtype=None,                     # Auto: float16 on cuda, float32 on cpu
        input_size=(1024, 1024),
        blur_radius=1.0,                # Subtle Gaussian anti-aliasing on edges
        soft_threshold_bounds=(0.02, 0.98),  # Clean noise in background
    )

    # 2. Instantiate pipeline (downloads weights on first run)
    remover = BackgroundRemover(config)

    # 3. Find input image
    sample_images = list(Path(".").glob("*.webp")) + list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
    if not sample_images:
        print("No sample images found in current directory.")
        return

    input_img_path = sample_images[0]
    print(f"\nProcessing '{input_img_path.name}'...")

    # 4. Remove background and composite
    rgba_image, alpha_mask, latency = remover.remove_background(
        input_img_path,
        background_color=None,  # None for transparent RGBA; or (255, 255, 255) for white
        return_latency=True
    )

    # 5. Save outputs
    output_path = Path("output_nobg.png")
    mask_path = Path("output_mask.png")
    rgba_image.save(output_path)
    alpha_mask.save(mask_path)

    print(f"✔ Saved transparent cutout to: {output_path}")
    print(f"✔ Saved alpha mask to: {mask_path}")
    print(f"⚡ Latency: Total={latency.total_ms:.1f}ms (Inference={latency.inference_ms:.1f}ms) | {latency.fps:.1f} FPS")

    # 6. Optional: Run in-depth multi-iteration benchmark
    print("\nRunning in-depth benchmark...")
    remover.benchmark(input_img_path, warmup_runs=2, benchmark_runs=5)


if __name__ == "__main__":
    example_single_image()

