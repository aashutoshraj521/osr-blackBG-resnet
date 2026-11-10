# Open Set Recognition (OSR) Project

**MTech Thesis Research Codebase**

A complete, modular PyTorch framework for **Object Extraction, Black-Background Normalization, Metric Learning, and Open Set Recognition (OSR)**.

---

## 🎯 Thesis Overview & Methodology

In real-world computer vision, image backgrounds introduce confounding context, spurious correlations, and domain shifts. This project implements a two-stage methodology:

1. **Object Extraction & Black-Background Normalization**:
   - The foreground object is segmented using high-resolution neural alpha matting (**BRIA RMBG-1.4**).
   - The background is completely converted to pure solid black `(0, 0, 0)`.
   - The extracted object is tightly cropped around its bounding box and centered on a square canvas.
   - This eliminates background clutter and forces the neural representation to learn intrinsic object geometry, texture, and semantic features.

2. **Metric Learning & Prototype-Based Open Set Recognition**:
   - A deep feature extractor projects the black-background object onto a unit hypersphere $\mathbb{S}^{D-1}$.
   - Class prototypes $c_k$ are learned/computed as empirical centroids for all known classes $k \in \{0, \dots, K-1\}$.
   - **At Test Time**:
     - Query images undergo the exact same pipeline: **foreground extraction $\to$ solid black background $\to$ embedding extraction**.
     - Cosine similarity $s_k(x) = \frac{z \cdot c_k}{\|z\| \|c_k\|}$ is computed against all known prototypes.
     - **Open-Set Decision Rule**: If $\max_k s_k(x) \ge \tau$, the object is matched to known class $k^* = \arg\max_k s_k(x)$. If $\max_k s_k(x) < \tau$, the object is rejected as **Unknown (Open-Set)**.

---

## 🏗️ Pipeline Architecture

```
Raw Input Dataset / Query Image
            │
            ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Neural Background Removal (RMBG)   │
│  - Deep alpha matting mask generation        │
│  - Solid black compositing: RGB (0, 0, 0)    │
│  - Object bounding-box square crop & center  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Metric Feature Extraction          │
│  - SimpleConvNet / ResNet / MobileNetV3      │
│  - L2-normalized embedding z on hypersphere  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Prototype Similarity Matching      │
│  - Cosine similarity: s_k = z · c_k          │
│  - Confidence scoring in [0, 1]              │
└──────────────────────┬───────────────────────┘
                       │
          Is max(s_k) >= tau (Threshold)?
         /                              \
       YES                               NO
       /                                  \
      ▼                                    ▼
[MATCH ACCEPTED: Class k*]       [REJECTED: UNKNOWN OBJECT]
```

---

## 📁 Repository Structure

```
osr-project/
├── bg_removal.py               # Modular PyTorch background removal & alpha matting
├── download_cifar_plus_10.py   # Downloader & black background processor for CIFAR+10 benchmark
├── dataset_preprocessor.py     # Batch foreground extractor & black-background converter
├── dataset.py                  # OSR Dataset, DataLoaders, & black-border augmentations
├── models.py                   # SimpleConvNet, ResNet, Cosine Classifier, OSRObjectMatcherModel
├── train.py                    # Metric learning training, prototype computation, threshold calibration
├── evaluate.py                 # Academic evaluation: Closed-Set Acc, Open-Set AUROC, OSCR, Confusion Matrix
├── matcher.py                  # Core inference engine for object matching
├── match.py                    # CLI tool for single/batch query matching with 4-panel visual reports
├── create_sample_dataset.py    # Synthetic multi-class dataset generator for testing
├── example_usage.py            # Quickstart example for bg_removal
├── requirements.txt            # Project dependencies
├── tests/
│   └── test_pipeline.py        # Automated unit and integration test suite
└── README.md                   # Complete documentation
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

```bash
# Recommended: Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Prepare Dataset: CIFAR+10 OSR Benchmark

Download and preprocess the standard Open Set Recognition **CIFAR+10** benchmark (4 known classes from CIFAR-10, 10 non-overlapping unknown classes from CIFAR-100), converting backgrounds into pure black `(0, 0, 0)`:

```bash
# Download and preprocess CIFAR+10 with RMBG-1.4 onto solid black background:
python3 download_cifar_plus_10.py \
    -o ./data/cifar_plus_10 \
    -p neal \
    -n 20 \
    --target-size 224 224

# Directory structure created:
# ./data/cifar_plus_10/
# ├── raw/             <- Original CIFAR images organized by class
# ├── black/           <- Processed images with pure black background (0, 0, 0)
# └── cifar_plus_10_comparison.png <- Side-by-side visual comparison sheet
```

Alternatively, you can generate a synthetic multi-class dataset for quick local testing:

```bash
# Generate a sample multi-class dataset with known and unknown classes
python3 create_sample_dataset.py -o ./data/raw_dataset --samples-known 20 --samples-unknown 10
```

Directory format:
```
data/raw_dataset/
├── cup/          # Known class
├── car/          # Known class
├── laptop/       # Known class
├── bottle/       # Known class
├── airplane/     # Unknown (Open-Set) class
└── star/         # Unknown (Open-Set) class
```

---

### 3. Extract Objects & Convert Backgrounds to Black

Process the raw dataset to extract foreground objects and composite them onto pure black `(0, 0, 0)`:

```bash
python3 dataset_preprocessor.py \
    -i ./data/raw_dataset \
    -o ./data/black_dataset \
    --target-size 224 224 \
    --crop-object \
    --padding 0.1 \
    --bg-color 0 0 0 \
    --save-masks
```

---

### 4. Train the Model

Train the metric learning model on the black-background objects:

```bash
python3 train.py \
    -d ./data/black_dataset \
    -o ./checkpoints \
    --backbone simple_cnn \
    --feature-dim 256 \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --known-ratio 0.8
```

Outputs saved to `./checkpoints/`:
- `best_model.pth`: Model weights with highest validation accuracy
- `prototypes.pt`: High-fidelity empirical class centroids
- `class_to_idx.json`: Class label index mapping
- `config.json`: Hyperparameters & calibrated open-set decision thresholds ($\tau_{95}, \tau_{99}, \tau_{\text{recommended}}$)
- `training_history.json`: Loss and accuracy curves

---

### 5. Evaluate Academic Performance (Closed-Set + Open-Set)

Run evaluation on test images containing both **Known** and **Unknown (Open-Set)** objects:

```bash
python3 evaluate.py \
    -d ./data/black_dataset \
    -c ./checkpoints \
    -o ./checkpoints/evaluation_report.json
```

**Metrics Output**:
- **Closed-Set (Known Classes)**:
  - Top-1 Classification Accuracy
  - Precision, Recall, and F1-Score per class
  - Confusion Matrix
- **Open-Set (Out-of-Distribution Rejection)**:
  - **AUROC (Area Under ROC Curve)**: Threshold-independent metric discriminating Known from Unknown
  - **A-OSCR (Area Under Open Set Classification Rate curve)**: Correct classification vs false alarms
  - **FPR@95%TPR**: False Positive Rate on unknowns when accepting 95% of knowns
  - **Open-Set (K+1) Macro F1-Score**

---

### 6. Test Query Image: Extract $\to$ Black Background $\to$ Match

Test raw query images with complex backgrounds using the end-to-end pipeline:

```bash
# Test a single raw image and generate 4-panel visual diagnostic
python3 match.py \
    -i query_image.jpg \
    -c ./checkpoints \
    --save-vis match_report.png
```

```bash
# Batch match a whole folder of test query images
python3 match.py \
    -i ./test_queries/ \
    -c ./checkpoints \
    -o ./match_results/
```

**Diagnostic Visualization Panels**:
1. **Raw Query Image**: Original photo with complex environment.
2. **Extracted Black BG Object**: Segmented object centered on pure black canvas.
3. **Segmentation Mask**: High-precision alpha channel.
4. **Prototype Similarity Match**: Bar chart of cosine similarities against known classes with threshold indicator line.

---

## 🧪 Automated Test Suite

Run the full automated test suite covering unit tests and end-to-end integration:

```bash
python3 -m unittest tests/test_pipeline.py
```

---

## ✂️ Standalone Background Removal Tool (`bg_removal.py`)

To run background removal independently on any image:

```bash
# Transparent RGBA cutout
python3 bg_removal.py -i input.jpg -o output.png --save-mask mask.png

# Solid black background
python3 bg_removal.py -i input.jpg -o output_black.png --bg-color 0 0 0

# Benchmark latency and FPS
python3 bg_removal.py -i input.jpg --benchmark --warmup 3 --runs 10
```

---

## 📐 Mathematical Formulation

### 1. Hyperspherical Cosine Embedding
Given an image $x$, the backbone extracts embedding $f_\theta(x) \in \mathbb{R}^D$, normalized to the unit sphere:
$$z = \frac{f_\theta(x)}{\|f_\theta(x)\|_2} \in \mathbb{S}^{D-1}$$

### 2. Class Prototypes & Cosine Classifier
For each known class $k \in \{0, \dots, K-1\}$, the empirical prototype is:
$$c_k = \frac{\sum_{i: y_i = k} z_i}{\left\|\sum_{i: y_i = k} z_i\right\|_2}$$

Cosine similarity between query embedding $z$ and prototype $c_k$:
$$s_k(z) = z \cdot c_k \in [-1, 1]$$

### 3. Open Set Decision Rule
Confidence score $S(z) = \frac{\max_k s_k(z) + 1}{2} \in [0, 1]$:
$$\hat{y}(x) = \begin{cases} \arg\max_k s_k(z), & \text{if } S(z) \ge \tau \\ \text{UNKNOWN}, & \text{if } S(z) < \tau \end{cases}$$

---

## 📜 License
MIT License. Developed for MTech Thesis on Open Set Recognition.
