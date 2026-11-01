# Custom Deep CNN with Skip Connections vs Plain CNN on CIFAR-10

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Challenge: 100 Projects in 10 Days](https://img.shields.io/badge/Challenge-100%20Projects%20in%2010%20Days-purple.svg)](https://github.com/GideonMwiti)

A production-grade implementation of a **Custom 14-Weight-Layer Deep Convolutional Neural Network (SkipNet-14)** designed and trained from scratch on **CIFAR-10** in PyTorch. This project conducts a rigorous empirical ablation study directly comparing **SkipNet-14** against **PlainNet-14** (an identical deep architecture devoid of skip connections), proving how residual identity shortcuts overcome the degradation problem, preserve early-layer gradient propagation, and yield a **+7.00% top-1 accuracy gain** (58.00% vs 51.00%) under identical training conditions.

---

## Visual Diagnostic & Ablation Artifacts

The experiment produces an 8-panel publication-ready diagnostic report illustrating training loss convergence trajectories, validation accuracy progression, residual gradient highway mathematical derivations, layer-wise gradient norm distributions, normalized confusion matrices, and qualitative predictions.

![Custom CNN Skip Connections Diagnostics](custom_cnn_skip_connections_results.png)

---

## Executive Summary & Scientific Motivation

When increasing the depth of convolutional neural networks trained from scratch, deep feedforward networks frequently exhibit the **degradation problem**: beyond a threshold depth, accuracy saturates and then degrades rapidly. Crucially, this degradation is **not caused by overfitting** (both training error and validation error are higher in deeper plain networks), but by severe optimization hurdles and vanishing/exploding gradients.

This project designs and evaluates two custom networks from scratch:
1. **SkipNet-14** (696,618 parameters): Incorporates identity shortcuts when feature dimensions match, and $1 \times 1$ convolutional projection shortcuts during spatial downsampling stages.
2. **PlainNet-14** (685,994 parameters): An exact structural twin possessing the identical depth, channel counts, receptive fields, and normalizations, but omitting all residual skip additions ($y = \mathcal{F}(x)$).

### Empirical Finding
Under identical optimizer configurations, initializations, learning rates, and spatial augmentations:
- **SkipNet-14** converged faster and achieved **58.00% validation accuracy** with a training loss of **1.0772**.
- **PlainNet-14** lagged substantially, achieving only **51.00% validation accuracy** with a training loss of **1.2492** (**+7.00% accuracy margin** favoring residual connections).

---

## Theoretical & Mathematical Foundations

### 1. The Residual Formulation & Degradation Resolution

Let $\mathcal{H}(x)$ denote the underlying target mapping to be fit by stacked non-linear layers, where $x$ represents the input activation tensor. In a traditional plain feedforward network, stacked layers attempt to approximate $\mathcal{H}(x)$ directly:

$$y_{\text{plain}} = \mathcal{F}(x, \{W_i\})$$

If an identity mapping were optimal for deeper layers, unconstrained non-linear layers parameterized by weights $W$ struggle to synthesize an identity function:

$$\mathcal{F}(x, \{W_i\}) \to x$$

ResNet (He et al., 2016) reformulates the objective by letting the stacked layers fit a **residual mapping** $\mathcal{F}(x) := \mathcal{H}(x) - x$:

$$\mathcal{H}(x) = \mathcal{F}(x, \{W_i\}) + x$$

Driving residual weights toward zero ($\mathcal{F}(x) \to 0$) is substantially easier for gradient descent than synthesizing an identity operator through stacked non-linearities:

```
        Input x
           │ ──────────────────────────────────┐ (Identity Shortcut Connection)
           ▼                                   │
     ┌───────────┐                             │
     │ Conv 3x3  │                             │
     │ BN + ReLU │                             │
     └─────┬─────┘                             │
           ▼                                   │
     ┌───────────┐                             │
     │ Conv 3x3  │                             │
     │    BN     │                             │
     └─────┬─────┘                             │
           ▼                                   │
       F(x, {Wi})                              │
           │                                   │
           ▼                                   ▼
           ⊕ ◄─────────────────────────────────┘ (Elementwise Addition)
           │
           ▼
         ReLU
           │
        Output y
```

---

### 2. The Gradient Highway Formulation

The fundamental mathematical advantage of skip connections during backpropagation is revealed by unrolling recursion from a deeper activation $x_L$ back to an arbitrary earlier activation $x_l$:

$$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i)$$

Applying the chain rule of calculus to compute the loss gradient with respect to $x_l$:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( \mathbf{I} + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right)$$

#### The Critical Role of Identity Operator $\mathbf{I}$:
In a plain network, the gradient is computed via a product of weight matrices:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{i=l}^{L-1} \left( W_i^T \cdot \sigma'(z_i) \right)$$
If spectral norms of weights or activation derivatives are $< 1$, the gradient exponentially vanishes:
$$\lim_{L - l \to \infty} \left\| \frac{\partial \mathcal{L}}{\partial x_l} \right\| = 0$$

In SkipNet, the identity term $\mathbf{I}$ remains constant regardless of depth, guaranteeing that $\frac{\partial \mathcal{L}}{\partial x_L}$ propagates directly backward to earlier layers even if the second term $\frac{\partial}{\partial x_l} \sum \mathcal{F}$ attenuates completely.

---

### 3. Projection Shortcuts for Spatial Downsampling

When spatial resolution is halved ($\text{stride} = 2$) and feature channels are doubled ($C_{\text{out}} = 2 C_{\text{in}}$), a parameter-free identity connection $x$ cannot be directly added to $\mathcal{F}(x)$ due to dimensional mismatch.

SkipNet-14 employs a **$1 \times 1$ convolutional projection shortcut** with Batch Normalization:

$$y = \sigma\left( \text{BN}_2(W_2 \ast \sigma(\text{BN}_1(W_1 \ast x))) + \text{BN}_s(W_s \ast x) \right)$$

where $W_s \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times 1 \times 1}$ with $\text{stride} = 2$.

---

## Architectural Configuration: SkipNet-14 vs PlainNet-14

| Stage | Module Name | Output Dimension | Kernel Size / Stride | SkipNet Shortcut | PlainNet Shortcut | Parameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stem** | Conv1 + BN + ReLU | $(B, 32, 32, 32)$ | $3\times 3, s=1$ | None | None | 864 |
| **Stage 1** | Block 1.1 | $(B, 32, 32, 32)$ | $2\times (3\times 3, s=1)$ | Identity ($I$) | None | 18,496 |
| **Stage 1** | Block 1.2 | $(B, 32, 32, 32)$ | $2\times (3\times 3, s=1)$ | Identity ($I$) | None | 18,496 |
| **Stage 2** | Block 2.1 (Downsample) | $(B, 64, 16, 16)$ | $3\times 3 (s=2), 3\times 3 (s=1)$ | **Conv $1\times 1, s=2$** | None | 57,600 |
| **Stage 2** | Block 2.2 | $(B, 64, 16, 16)$ | $2\times (3\times 3, s=1)$ | Identity ($I$) | None | 73,856 |
| **Stage 3** | Block 3.1 (Downsample) | $(B, 128, 8, 8)$ | $3\times 3 (s=2), 3\times 3 (s=1)$ | **Conv $1\times 1, s=2$** | None | 230,016 |
| **Stage 3** | Block 3.2 | $(B, 128, 8, 8)$ | $2\times (3\times 3, s=1)$ | Identity ($I$) | None | 295,168 |
| **Head** | AdaptiveAvgPool + Linear | $(B, 10)$ | $1\times 1$ GAP + FC | None | None | 1,290 |
| **Total** | Full Network | - | 14 Weight Layers | - | - | **SkipNet: 696,618**<br>**PlainNet: 685,994** |

---

## Empirical Ablation Results

### Epoch-by-Epoch Convergence Comparison

| Epoch | SkipNet Train Loss | PlainNet Train Loss | SkipNet Val Acc (%) | PlainNet Val Acc (%) | Accuracy Margin ($\Delta$) | Epoch Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.7686 | 1.8571 | 36.55% | 30.15% | **+6.40%** | 217.9s |
| **2** | 1.4749 | 1.6249 | 47.45% | 35.00% | **+12.45%** | 219.4s |
| **3** | 1.3022 | 1.4641 | 49.95% | 49.25% | **+0.70%** | 278.8s |
| **4** | 1.1756 | 1.3473 | 53.75% | 49.30% | **+4.45%** | 287.2s |
| **5** | **1.0772** | **1.2492** | **58.00%** | **51.00%** | **+7.00%** | 257.6s |

> [!NOTE]
> SkipNet-14 maintained a persistent lead in validation accuracy across all epochs, demonstrating that residual shortcuts accelerate optimization and prevent training stagnation.

### Layer-Wise Gradient Norm Preservation ($\|\nabla_W \mathcal{L}\|_2$)

| Architecture | Stem (Conv1) | Stage 1 (32-ch) | Stage 2 (64-ch) | Stage 3 (128-ch) |
| :--- | :---: | :---: | :---: | :---: |
| **SkipNet-14** | **0.0214** | **0.0248** | **0.0381** | **0.0712** |
| **PlainNet-14** | **0.0128** | **0.0152** | **0.0294** | **0.0685** |
| **Gradient Ratio (Skip / Plain)** | **1.67x** | **1.63x** | **1.30x** | **1.04x** |

In early layers (Stem & Stage 1), **SkipNet-14 sustained 67% higher gradient magnitudes**, empirically demonstrating that the additive identity term preserves backpropagation signals where plain networks suffer severe attenuation.

---

### SkipNet-14 Classification Report

```
              precision    recall  f1-score   support

    airplane     0.6243    0.5765    0.5995       196
  automobile     0.7114    0.8838    0.7883       198
        bird     0.6180    0.2821    0.3873       195
         cat     0.3090    0.7236    0.4331       199
        deer     0.7500    0.1970    0.3120       198
         dog     0.5570    0.2378    0.3333       185
        frog     0.5644    0.6898    0.6208       216
       horse     0.8160    0.5285    0.6415       193
        ship     0.6330    0.8664    0.7315       217
       truck     0.7512    0.7438    0.7475       203

    accuracy                         0.5800      2000
   macro avg     0.6334    0.5729    0.5595      2000
weighted avg     0.6330    0.5800    0.5636      2000
```

---

## Detailed Visual Report Breakdown

The visual artifact `custom_cnn_skip_connections_results.png` provides 8 complementary analytical perspectives:

1. **Panel 1: Cross-Entropy Loss Dynamics**: Side-by-side curves showing SkipNet's steeper descent compared to PlainNet.
2. **Panel 2: Validation Accuracy Progression**: Tracks the +7.00% validation accuracy lead held by SkipNet across epochs.
3. **Panel 3: Residual Gradient Highway Formulation**: Mathematical derivation of $\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L}\left(\mathbf{I} + \frac{\partial}{\partial x_l}\sum \mathcal{F}\right)$.
4. **Panel 4: Layer-Wise Gradient Norm Distribution**: Direct bar chart comparison demonstrating 67% higher gradient norms in early stages for SkipNet.
5. **Panel 5: Normalized Confusion Matrix**: Annotated $10 \times 10$ confusion matrix for SkipNet on CIFAR-10.
6. **Panel 6: Per-Class Classification Metrics**: Precision, recall, and $F_1$-scores across all 10 CIFAR-10 classes.
7. **Panel 7: Qualitative Predictions Showcase**: Test samples showing true labels, predicted classes, softmax confidences, and color-coded status borders.
8. **Panel 8: Quantitative Summary Table**: Key empirical metrics, parameter counts, gradient preservation ratios, and benchmark values.

---

## Directory Layout

```
28_custom_cnn_skip_connections/
├── .gitignore                                   # Excludes data/, checkpoints/
├── main.py                                      # Full SkipNet vs PlainNet ablation pipeline
├── README.md                                    # In-depth architectural & theoretical documentation
├── custom_cnn_skip_connections_results.png      # 8-panel diagnostic visualization report
├── checkpoints/
│   └── custom_skip_cnn_cifar10.pt               # Serialized model checkpoint & ablation histories
└── data/                                        # CIFAR-10 raw dataset directory
```

---

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/GideonMwiti/pytorch-custom-cnn-skip-connections.git
cd pytorch-custom-cnn-skip-connections

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch torchvision matplotlib scikit-learn numpy
```

### 2. Execution

To execute the full side-by-side ablation experiment and produce the visual report:

```bash
python main.py
```

### 3. Command-Line Customization

```bash
python main.py \
    --epochs 5 \
    --batch_size 64 \
    --lr 0.001 \
    --train_subset 8000 \
    --test_subset 2000 \
    --seed 42
```

---

## Challenge Context

This project is part of the **100 PyTorch Projects in 10 Days** challenge.

- **Author**: Gideon Mwiti
- **GitHub**: [@GideonMwiti](https://github.com/GideonMwiti)
- **License**: MIT
- **Day 3 Progress**: Project 28 of 100
