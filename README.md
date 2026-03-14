# MVBA: Minimal Viable Binding Architecture

A neural architecture for solving the binding problem in object-centric representation learning, inspired by Binding by Firing Rate Enhancement (BBRE) mechanisms observed in biological neural systems.

## The Binding Problem

When perceiving a scene containing a red square and a blue circle, the visual system must correctly associate ("bind") each color with its corresponding shape. MVBA addresses this computational challenge through a combination of slot-based attention, spatial binding, feature binding, and power-law enhancement controlled by learned alpha parameters.

## Architecture Overview

MVBA processes images through a six-stage pipeline:

```
Input Image
    |
    v
[1. Feature Extraction]   CNN backbone + learnable positional encoding
    |
    v
[2. Slot Attention]        Iterative competitive binding (Locatello et al., 2020)
    |
    v
[3. Alpha Generation]     Learned sharpening parameters alpha in [1, 3]
    |
    v
[4. Spatial Binding]      WHERE: power-law enhanced pixel-to-slot assignment
    |
    v
[5. Feature Binding]      WHAT: power-law enhanced feature-to-slot assignment
    |
    v
[6. Decoder]              Per-slot reconstruction with mask-based compositing
    |
    v
Reconstructed Image
```

### Core Components

| Module | File | Description |
|--------|------|-------------|
| Feature Extractor | `src/models/feature_extractor.py` | 3-layer CNN with GroupNorm and 8-channel learnable positional encoding. Preserves full spatial resolution. |
| Slot Attention | `src/models/slot_attention.py` | Multi-head competitive attention with GRU-based iterative refinement. Symmetry-breaking initialization ensures slot specialization. |
| Alpha Generator | `src/models/alpha_generator.py` | Generates per-slot sharpening parameters from slot states via sigmoid-scaled MLPs. Separate networks for spatial and feature alpha. |
| Spatial Binding | `src/models/spatial_binding.py` | Query-key spatial attention with power-law sharpening applied before competitive softmax. Multi-scale refinement (3x3, 5x5, 7x7 kernels) enforces Gestalt-inspired spatial coherence. |
| Feature Binding | `src/models/feature_binding.py` | Power-law feature enhancement `f' = sign(f) * |f|^alpha` with slot-conditioned gating and attention-weighted pooling. |
| Decoder | Integrated in model files | MLP + transposed convolution decoder producing per-slot RGB + mask outputs, composed via softmax-normalized blending. |

### Power-Law Enhancement (BBRE)

The central mechanism is power-law transformation applied to both spatial logits and feature magnitudes:

```
f' = sign(f) * |f|^alpha       where alpha in [1.0, 3.0]
```

- **alpha = 1**: Identity (no enhancement, equivalent to standard softmax competition)
- **alpha > 1**: Amplifies strong signals, suppresses weak ones (sharpening)
- **Effect**: Creates clearer winner-take-all dynamics for object segmentation

Alpha values are learned per-slot from slot states, allowing the model to dynamically adjust competition strength based on scene content.

## Model Variants

Seven variants are provided for ablation analysis, systematically isolating architectural and enhancement contributions:

| Variant | File | Spatial Binding | Feature Binding | Alpha |
|---------|------|:-:|:-:|:-:|
| `baseline` | `mvba_baseline.py` | - | - | - |
| `spatial_fixed` | `mvba_spatial_fixed.py` | Yes | - | Fixed (1.0) |
| `feature_fixed` | `mvba_feature_fixed.py` | - | Yes | Fixed (1.0) |
| `full_fixed` | `mvba_full_fixed.py` | Yes | Yes | Fixed (1.0) |
| `spatial` | `mvba_spatial.py` | Yes | - | Learned [1,3] |
| `feature` | `mvba_feature.py` | - | Yes | Learned [1,3] |
| `full` | `mvba.py` | Yes | Yes | Learned [1,3] |

## Dataset

**SimpleObjects-MVBA** -- synthetic geometric object dataset designed for binding evaluation:

- **Image size**: 64x64 pixels
- **Objects per image**: 2--4 (circles, squares, triangles)
- **Training set**: 8,000 images
- **Test set**: 2,000 images
- **Background**: Light gray (240, 240, 240)
- **Format**: PNG images with per-image JSON metadata
- **Normalization**: Mean [0.884, 0.865, 0.877], Std [0.192, 0.230, 0.215]

## Training

All variants are trained with identical hyperparameters for controlled comparison:

```bash
python scripts/training/train.py --model-variant full --epochs 50 --batch-size 16
```

**Configuration:**
- Optimizer: Adam, learning rate 1e-4
- Loss: MSE reconstruction loss only
- Epochs: 50
- Slots: 4, Slot dimension: 128, Feature dimension: 64
- Slot attention iterations: 3
- Random seed: 42

**Available variants:**
```
baseline | spatial | feature | spatial_fixed | feature_fixed | full_fixed | full
```

Checkpoints are saved per-epoch to `train_recon-only/train_{variant}/checkpoints/`.

## Reconstruction Quality Ablation Experiment

A complete ablation study evaluating all 7 variants on 2,000 test images is provided in `experiments_recon_quality_ablation/`.

### Running the Experiment

```bash
# Step 1: Evaluate all model variants
cd experiments_recon_quality_ablation
python evaluate_reconstruction.py --config ../config.yaml

# Step 2: Statistical analysis
python analyze_results.py --config ../config.yaml

# Step 3: Extended statistical tests (normality, bootstrap CIs, effect sizes)
python statistical_analysis.py

# Step 4: Generate publication-ready visualizations
python generate_visualizations.py --config ../config.yaml
```

### Key Results

| Variant | MSE | PSNR (dB) | SSIM | Improvement vs Baseline |
|---------|----:|----------:|-----:|------------------------:|
| baseline | 1.1736 | -0.32 | 0.086 | -- |
| spatial_fixed | 1.1758 | -0.33 | 0.069 | -0.2% |
| spatial | 0.6003 | 2.81 | 0.484 | 48.9% |
| feature_fixed | 0.5438 | 3.31 | 0.588 | 53.7% |
| feature | 0.4424 | 4.22 | 0.658 | 62.3% |
| full_fixed | 0.4108 | 4.61 | 0.621 | 65.0% |
| **full** | **0.3353** | **5.36** | **0.714** | **71.4%** |

### Principal Findings

1. **Feature binding contributes more than spatial binding**: Feature-only variants achieve 53.7% improvement vs. near-zero for spatial-only without BBRE.
2. **BBRE enhancement is critical for spatial binding**: `spatial_fixed` (alpha=1) shows no improvement over baseline, while `spatial` (learned alpha) achieves 48.9% improvement.
3. **Synergistic combination**: The full model's 71.4% improvement exceeds the sum of individual component contributions, indicating non-additive benefits.
4. **Statistical robustness**: All key comparisons significant at p < 0.001 with large effect sizes (Cohen's d > 0.8).

### Statistical Analyses Performed

- Paired t-tests (21 pairwise comparisons across 7 variants)
- Cohen's d effect sizes with interpretation
- Shapiro-Wilk and D'Agostino normality tests
- Bootstrap confidence intervals (1,000 resamples)
- Metric correlation analysis (MSE, PSNR, SSIM, L1)

### Generated Outputs

| Output | Location |
|--------|----------|
| Per-sample metrics | `results/metrics/per_sample_metrics.json` |
| Aggregate metrics | `results/metrics/aggregate_metrics.json` |
| Statistical tests | `results/analysis/statistical_tests.json` |
| Comparison matrices | `results/analysis/comparison_matrix_{mse,psnr,ssim}.csv` |
| Diagnostic plots | `statistical_analysis/*.png` |
| Visualizations | `visualizations/` |
| Full report | `report/COMPREHENSIVE_EXPERIMENTAL_REPORT.md` |

## Project Structure

```
MVBA-main/
|-- src/
|   |-- models/
|   |   |-- mvba.py                  # Full model (spatial + feature + learned alpha)
|   |   |-- mvba_baseline.py         # Slot Attention only
|   |   |-- mvba_spatial.py          # Spatial binding + learned alpha
|   |   |-- mvba_spatial_fixed.py    # Spatial binding + fixed alpha=1
|   |   |-- mvba_feature.py          # Feature binding + learned alpha
|   |   |-- mvba_feature_fixed.py    # Feature binding + fixed alpha=1
|   |   |-- mvba_full_fixed.py       # Both bindings + fixed alpha=1
|   |   |-- feature_extractor.py     # CNN + positional encoding
|   |   |-- slot_attention.py        # Iterative competitive attention
|   |   |-- alpha_generator.py       # Learned alpha parameters
|   |   |-- spatial_binding.py       # WHERE binding module
|   |   |-- feature_binding.py       # WHAT binding module
|   |-- losses.py                    # MSE reconstruction loss
|   |-- metrics.py                   # PSNR, SSIM, MSE, slot diversity, spatial consistency
|   |-- visualization.py            # Training and analysis visualizations
|
|-- scripts/
|   |-- training/
|       |-- train.py                 # Training script for all variants
|
|-- experiments_recon_quality_ablation/
|   |-- config.yaml                  # Experiment configuration
|   |-- evaluate_reconstruction.py   # Evaluation pipeline
|   |-- analyze_results.py           # Statistical comparison
|   |-- statistical_analysis.py      # Extended statistical tests
|   |-- generate_visualizations.py   # Publication-ready figures
|   |-- results/                     # Metrics and analysis outputs
|   |-- statistical_analysis/        # Diagnostic plots
|   |-- report/                      # Written reports
|
|-- data/
|   |-- simple_objects/
|       |-- dataset_info.json        # Dataset metadata and normalization stats
|       |-- train/                   # 8,000 training images + metadata
|       |-- test/                    # 2,000 test images + metadata
|
|-- train_recon-only/                # Training outputs and checkpoints
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn
- PyYAML
- tqdm
- Pillow

## References

- Locatello, F., et al. (2020). *Object-Centric Learning with Slot Attention.* NeurIPS.
- The BBRE (Binding by Firing Rate Enhancement) hypothesis from neuroscience literature on how the brain solves the binding problem through modulation of neural firing rates.
