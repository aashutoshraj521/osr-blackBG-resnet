# osr-project
Open Set Recognition Project.
This is my MTech Thesis.

---

## ✂️ Modular PyTorch Background Removal Pipeline (`bg_removal.py`)

A high-performance, modular background removal and alpha matting pipeline built with PyTorch and Hugging Face `transformers` (supporting **BRIA RMBG-1.4**, **RMBG-2.0**, and **DIS/IS-Net**).

### Features
- **Automatic Weight Download & Inference Setup**: Loads pre-trained segmentation models via Hugging Face Hub with device auto-detection (`cuda`, `mps`, `cpu`) and mixed precision (`float16` / `bfloat16` on GPU).
- **Image Preprocessing**: Handles EXIF rotation, bilinear resizing to model native resolution (default: `1024x1024`), dynamic range scaling to `[0, 1]`, and channel-wise normalization.
- **Accurate Forward Pass**: Inference mode execution (`torch.inference_mode()`) with multi-scale activation normalization.
- **Mask Post-Processing**: Bilinear interpolation back to original image dimensions, soft/hard thresholding to eliminate background noise, and Gaussian blur edge feathering to anti-alias edges.
- **Alpha Compositing**: Produces 4-channel transparent `RGBA` images or composites against solid background colors.
- **Inference Benchmarking**: GPU synchronization (`torch.cuda.synchronize`), warmup runs, and granular stage-by-stage latency tracking (Preprocessing, Model Inference, Post-Processing, Alpha Compositing, and Throughput FPS).

### Installation

```bash
# Recommended: Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### CLI Quickstart

#### 1. Process a single image into transparent RGBA:
```bash
python3 bg_removal.py -i input.jpg -o output.png --save-mask mask.png
```

#### 2. Process an image with edge smoothing and noise cutoff:
```bash
python3 bg_removal.py -i input.jpg -o output.png --blur 1.0 --soft-threshold 0.05 0.95
```

#### 3. Composite onto a solid white background:
```bash
python3 bg_removal.py -i input.jpg -o output_white.png --bg-color 255 255 255
```

#### 4. Run an in-depth latency benchmark:
```bash
python3 bg_removal.py -i input.jpg --benchmark --warmup 3 --runs 10
```

#### 5. Batch process an entire directory:
```bash
python3 bg_removal.py -i ./input_images/ -o ./processed_images/
```

### Python API Usage

```python
from bg_removal import BackgroundRemover, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    model_id="briaai/RMBG-1.4",
    blur_radius=1.0,
    soft_threshold_bounds=(0.02, 0.98),
)

remover = BackgroundRemover(config)

# Remove background
rgba_image, alpha_mask, latency = remover.remove_background(
    "input.jpg",
    return_latency=True
)

rgba_image.save("cutout.png")
print(f"Inference latency: {latency.inference_ms:.2f} ms ({latency.fps:.1f} FPS)")
```

