"""
Custom Deep CNN with Skip Connections vs Plain CNN on CIFAR-10
==============================================================
This project designs and trains a custom deep convolutional neural network
incorporating residual skip connections from scratch on CIFAR-10.

It empirically investigates the foundational breakthrough of residual learning:
- Skip Connections provide an uninterrupted mathematical gradient highway:
  dH/dx = dF/dx + I
  guaranteeing non-vanishing gradient propagation across arbitrary depth.
- In contrast, Plain Deep CNNs (lacking skip connections) suffer from the
  degradation problem and severe gradient attenuation in early layers.

The pipeline implements an ablation study training SkipNet-14 against PlainNet-14
under identical hyperparameters, logs layer-wise gradient norms, and produces
an 8-panel publication-grade diagnostic report.
"""

import os
import sys
import types
import io
import time
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Ensure standard compression modules are stubbed if missing on Windows environments
if "bz2" not in sys.modules:
    sys.modules["bz2"] = types.SimpleNamespace(open=io.open, BZ2File=io.BytesIO)
if "lzma" not in sys.modules:
    sys.modules["lzma"] = types.SimpleNamespace(
        open=io.open,
        LZMAFile=io.BytesIO,
        FORMAT_ALONE=1,
        FORMAT_XZ=2,
        CHECK_NONE=0,
        CHECK_CRC32=1,
        CHECK_CRC64=4,
        CHECK_SHA256=10,
    )

SCRIPT_DIR = Path(__file__).resolve().parent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def set_seed(seed: int = 42) -> None:
    """Set random seeds across libraries for strict deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    train_subset: int = 8000,
    test_subset: int = 2000,
) -> Tuple[DataLoader, DataLoader, torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """
    Construct optimized CIFAR-10 data loaders with standard spatial augmentations.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=test_transform,
    )

    if train_subset and train_subset < len(train_dataset):
        train_indices = list(range(train_subset))
        active_train_dataset = Subset(train_dataset, train_indices)
    else:
        active_train_dataset = train_dataset

    if test_subset and test_subset < len(test_dataset):
        test_indices = list(range(test_subset))
        active_test_dataset = Subset(test_dataset, test_indices)
    else:
        active_test_dataset = test_dataset

    train_loader = DataLoader(
        active_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        active_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, train_dataset, test_dataset


class CustomBlock(nn.Module):
    """
    Modular building block configurable with or without identity/projection skip connections.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_skip: bool = True):
        super().__init__()
        self.use_skip = use_skip

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_skip and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif use_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip:
            out = out + self.shortcut(residual)
        return self.relu(out)


class CustomCNN(nn.Module):
    """
    14-Weight-Layer Deep Convolutional Architecture for CIFAR-10.
    Can be instantiated with skip connections (SkipNet-14) or without (PlainNet-14).
    """
    def __init__(self, num_classes: int = 10, use_skip: bool = True):
        super().__init__()
        self.use_skip = use_skip

        # Initial stem convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32 channels, 32x32 feature maps (2 blocks)
        self.stage1 = nn.Sequential(
            CustomBlock(32, 32, stride=1, use_skip=use_skip),
            CustomBlock(32, 32, stride=1, use_skip=use_skip)
        )

        # Stage 2: 64 channels, 16x16 feature maps (2 blocks, downsampling stride 2)
        self.stage2 = nn.Sequential(
            CustomBlock(32, 64, stride=2, use_skip=use_skip),
            CustomBlock(64, 64, stride=1, use_skip=use_skip)
        )

        # Stage 3: 128 channels, 8x8 feature maps (2 blocks, downsampling stride 2)
        self.stage3 = nn.Sequential(
            CustomBlock(64, 128, stride=2, use_skip=use_skip),
            CustomBlock(128, 128, stride=1, use_skip=use_skip)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def compute_layer_gradient_norms(model: CustomCNN) -> Dict[str, float]:
    """Compute average gradient Frobenius norms across the 4 network stages."""
    def get_norm(module: nn.Module) -> float:
        norms = [p.grad.data.norm(2).item() for p in module.parameters() if p.grad is not None]
        return float(np.mean(norms)) if norms else 0.0

    return {
        "Stem (Conv1)": get_norm(model.init_conv),
        "Stage 1 (32-ch)": get_norm(model.stage1),
        "Stage 2 (64-ch)": get_norm(model.stage2),
        "Stage 3 (128-ch)": get_norm(model.stage3),
    }


def train_epoch_with_grad_logging(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """Train for one epoch, returning loss, accuracy, and stage gradient norms."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    accumulated_grads = {"Stem (Conv1)": [], "Stage 1 (32-ch)": [], "Stage 2 (64-ch)": [], "Stage 3 (128-ch)": []}

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Sample gradient norms on the final batch
        if batch_idx == len(loader) - 1:
            grads = compute_layer_gradient_norms(model)
            for k, v in grads.items():
                accumulated_grads[k].append(v)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = total_loss / total
    epoch_acc = (correct / total) * 100.0
    mean_grads = {k: float(np.mean(v)) if v else 0.0 for k, v in accumulated_grads.items()}
    return epoch_loss, epoch_acc, mean_grads


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model, returning metrics, targets, predictions, and probabilities."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        probs = torch.softmax(outputs, dim=1)

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = total_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc, np.array(all_targets), np.array(all_preds), np.array(all_probs)


def generate_visual_artifacts(
    history_skip: Dict[str, List[float]],
    history_plain: Dict[str, List[float]],
    grad_skip: Dict[str, float],
    grad_plain: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    test_dataset: Any,
    params_skip: int,
    params_plain: int,
    save_path: Path,
) -> None:
    """
    Generate an 8-panel publication-grade ablation and diagnostic report.
    """
    fig = plt.figure(figsize=(22, 16), dpi=300)
    plt.subplots_adjust(hspace=0.38, wspace=0.30, top=0.94, bottom=0.06, left=0.06, right=0.95)
    fig.suptitle(
        "Custom Deep CNN with Skip Connections (SkipNet-14) vs Plain CNN (PlainNet-14) on CIFAR-10\n"
        "Empirical Investigation of Residual Gradient Highways, Vanishing Gradients, and Convergence Acceleration",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    epochs = list(range(1, len(history_skip["train_loss"]) + 1))

    # --- Panel 1: Training & Validation Loss Comparison ---
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.plot(epochs, history_skip["train_loss"], marker="o", color="#1f77b4", linewidth=2.2, label="SkipNet (Train)")
    ax1.plot(epochs, history_skip["val_loss"], marker="s", color="#3470a3", linestyle="--", linewidth=2.0, label="SkipNet (Val)")
    ax1.plot(epochs, history_plain["train_loss"], marker="^", color="#d62728", linewidth=2.2, label="PlainNet (Train)")
    ax1.plot(epochs, history_plain["val_loss"], marker="v", color="#e76f51", linestyle="--", linewidth=2.0, label="PlainNet (Val)")
    ax1.set_title("Cross-Entropy Loss Dynamics", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # --- Panel 2: Validation Accuracy Progression ---
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.plot(epochs, history_skip["val_acc"], marker="o", color="#2ca02c", linewidth=2.5, label="SkipNet (Validation)")
    ax2.plot(epochs, history_plain["val_acc"], marker="s", color="#9467bd", linewidth=2.5, label="PlainNet (Validation)")
    ax2.set_title("Top-1 Validation Accuracy Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8)

    # Annotate accuracy difference
    gap = history_skip["val_acc"][-1] - history_plain["val_acc"][-1]
    ax2.annotate(
        f"SkipNet Lead: +{gap:.2f}%",
        xy=(epochs[-1], history_skip["val_acc"][-1]),
        xytext=(epochs[-1] - 1.5, history_skip["val_acc"][-1] - 6.0),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf7ec", edgecolor="#2ca02c"),
    )

    # --- Panel 3: Skip Connection Gradient Highway Formulation ---
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.axis("off")
    ax3.set_title("Residual Gradient Highway Formulation", fontsize=12, fontweight="bold")

    theory_text = (
        "Residual Learning Framework (He et al., 2016):\n"
        "════════════════════════════════════════════════════\n"
        "• Target Mapping:     H(x)\n"
        "• Learned Residual:   F(x) := H(x) - x\n"
        "• Reconstructed Map:  H(x) = F(x, {Wi}) + x\n\n"
        "Mathematical Backpropagation Gradient Highway:\n"
        "Let deeper unit L and shallower unit l be related by:\n"
        "    x_L = x_l + sum_{i=l}^{L-1} F(x_i, W_i)\n\n"
        "Applying the chain rule of calculus:\n"
        "    dL/dx_l = dL/dx_L * [ I + d/dx_l ( sum F ) ]\n\n"
        "Key Significance of the Identity Term I:\n"
        "• The identity operator I guarantees that gradient\n"
        "  dL/dx_L propagates directly back to layer l WITHOUT\n"
        "  attenuation, completely circumventing vanishing\n"
        "  gradients even across arbitrarily deep networks!"
    )
    ax3.text(
        0.02, 0.95, theory_text,
        transform=ax3.transAxes,
        fontsize=7.8,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f9f4", edgecolor="#2ca02c", alpha=0.9),
    )

    # --- Panel 4: Layer-Wise Gradient Norms Comparison ---
    ax4 = fig.add_subplot(2, 4, 4)
    stages = list(grad_skip.keys())
    x_pos = np.arange(len(stages))
    width = 0.35

    skip_norms = [grad_skip[s] for s in stages]
    plain_norms = [grad_plain[s] for s in stages]

    ax4.bar(x_pos - width / 2, skip_norms, width, label="SkipNet-14", color="#2ca02c")
    ax4.bar(x_pos + width / 2, plain_norms, width, label="PlainNet-14", color="#d62728")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(stages, rotation=25, ha="right", fontsize=8.5)
    ax4.set_ylabel("Average Gradient Norm ||dL/dW||_2", fontsize=10)
    ax4.set_title("Layer-Wise Gradient Norm Distribution", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=8)

    # --- Panel 5: Normalized Confusion Matrix ---
    ax5 = fig.add_subplot(2, 4, 5)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    im = ax5.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    tick_marks = np.arange(len(CIFAR10_CLASSES))
    ax5.set_xticks(tick_marks)
    ax5.set_yticks(tick_marks)
    ax5.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax5.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax5.set_title("Normalized Confusion Matrix (SkipNet)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Predicted Label", fontsize=10)
    ax5.set_ylabel("Ground Truth Label", fontsize=10)

    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(CIFAR10_CLASSES)):
            val = cm_norm[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax5.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=6.5)

    # --- Panel 6: Per-Class Precision & Recall ---
    ax6 = fig.add_subplot(2, 4, 6)
    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, output_dict=True)
    precisions = [report[c]["precision"] * 100 for c in CIFAR10_CLASSES]
    recalls = [report[c]["recall"] * 100 for c in CIFAR10_CLASSES]
    f1_scores = [report[c]["f1-score"] * 100 for c in CIFAR10_CLASSES]

    x = np.arange(len(CIFAR10_CLASSES))
    w = 0.25
    ax6.bar(x - w, precisions, w, label="Precision", color="#3470a3")
    ax6.bar(x, recalls, w, label="Recall", color="#e76f51")
    ax6.bar(x + w, f1_scores, w, label="F1-Score", color="#2a9d8f")
    ax6.set_xticks(x)
    ax6.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax6.set_ylabel("Score (%)", fontsize=10)
    ax6.set_ylim(0, 105)
    ax6.set_title("Per-Class Classification Metrics (SkipNet)", fontsize=12, fontweight="bold")
    ax6.grid(axis="y", linestyle=":", alpha=0.6)
    ax6.legend(loc="lower right", fontsize=8)

    # --- Panel 7: Qualitative Predictions Showcase ---
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.axis("off")
    ax7.set_title("Sample Test Predictions (SkipNet-14)", fontsize=12, fontweight="bold")

    sample_indices = np.random.choice(len(y_true), size=min(6, len(y_true)), replace=False)
    sub_cols = 3
    for k, idx in enumerate(sample_indices):
        img_tensor, _ = test_dataset[idx]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
        img_np = np.clip(img_np, 0.0, 1.0)

        true_cls = CIFAR10_CLASSES[y_true[idx]]
        pred_cls = CIFAR10_CLASSES[y_pred[idx]]
        conf = y_prob[idx, y_pred[idx]] * 100
        is_correct = (y_true[idx] == y_pred[idx])

        sub_ax = fig.add_axes([
            0.57 + (k % sub_cols) * 0.125,
            0.12 + (1 - (k // sub_cols)) * 0.18,
            0.10,
            0.14
        ])
        sub_ax.imshow(img_np)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        color = "#2a9d8f" if is_correct else "#e76f51"
        sub_ax.set_title(
            f"T: {true_cls}\nP: {pred_cls} ({conf:.1f}%)",
            fontsize=7.5,
            color=color,
            fontweight="bold"
        )
        for spine in sub_ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # --- Panel 8: Quantitative Ablation Summary Table ---
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis("off")
    ax8.set_title("Quantitative Ablation & Architecture Summary", fontsize=12, fontweight="bold")

    table_data = [
        ["Architecture Type", "Custom 14-Layer Deep CNN"],
        ["SkipNet-14 Parameters", f"{params_skip:,}"],
        ["PlainNet-14 Parameters", f"{params_plain:,}"],
        ["SkipNet Final Val Loss", f"{history_skip['val_loss'][-1]:.4f}"],
        ["PlainNet Final Val Loss", f"{history_plain['val_loss'][-1]:.4f}"],
        ["SkipNet Peak Val Accuracy", f"{max(history_skip['val_acc']):.2f}%"],
        ["PlainNet Peak Val Accuracy", f"{max(history_plain['val_acc']):.2f}%"],
        ["Skip Connection Gain", f"+{max(history_skip['val_acc']) - max(history_plain['val_acc']):.2f}%"],
        ["Stem Gradient (Skip vs Plain)", f"{grad_skip['Stem (Conv1)']:.4f} vs {grad_plain['Stem (Conv1)']:.4f}"],
    ]

    table = ax8.table(
        cellText=table_data,
        colLabels=["Property / Metric", "Empirical Value / Comparison"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2b5c8f")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#f1f5f9")
            else:
                cell.set_facecolor("#ffffff")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Artifact] Saved comprehensive visual report to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom CNN with Skip Connections on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_subset", type=int, default=8000, help="Train samples")
    parser.add_argument("--test_subset", type=int, default=2000, help="Test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")

    data_dir = SCRIPT_DIR / "data"
    print(f"[Data] Loading CIFAR-10 from: {data_dir}")
    train_loader, test_loader, full_train_dataset, full_test_dataset = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )
    print(f"[Data] Loaded {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} validation samples.")

    # 1. Instantiate Models
    print("\n[Models] Instantiating SkipNet-14 (with residual connections) and PlainNet-14 (without)...")
    model_skip = CustomCNN(num_classes=10, use_skip=True).to(device)
    model_plain = CustomCNN(num_classes=10, use_skip=False).to(device)

    params_skip = sum(p.numel() for p in model_skip.parameters())
    params_plain = sum(p.numel() for p in model_plain.parameters())
    print(f"SkipNet-14 Parameters: {params_skip:,} | PlainNet-14 Parameters: {params_plain:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer_skip = optim.AdamW(model_skip.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_plain = optim.AdamW(model_plain.parameters(), lr=args.lr, weight_decay=1e-4)

    history_skip = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    history_plain = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    latest_grads_skip = {}
    latest_grads_plain = {}

    print(f"\n--- Training SkipNet-14 vs PlainNet-14 for {args.epochs} Epochs ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train SkipNet
        tr_loss_s, tr_acc_s, grads_s = train_epoch_with_grad_logging(model_skip, train_loader, criterion, optimizer_skip, device)
        val_loss_s, val_acc_s, _, _, _ = evaluate(model_skip, test_loader, criterion, device)
        latest_grads_skip = grads_s

        # Train PlainNet
        tr_loss_p, tr_acc_p, grads_p = train_epoch_with_grad_logging(model_plain, train_loader, criterion, optimizer_plain, device)
        val_loss_p, val_acc_p, _, _, _ = evaluate(model_plain, test_loader, criterion, device)
        latest_grads_plain = grads_p

        dt = time.time() - t0

        history_skip["train_loss"].append(tr_loss_s)
        history_skip["train_acc"].append(tr_acc_s)
        history_skip["val_loss"].append(val_loss_s)
        history_skip["val_acc"].append(val_acc_s)

        history_plain["train_loss"].append(tr_loss_p)
        history_plain["train_acc"].append(tr_acc_p)
        history_plain["val_loss"].append(val_loss_p)
        history_plain["val_acc"].append(val_acc_p)

        print(
            f"Epoch [{epoch}/{args.epochs}] ({dt:.1f}s) | "
            f"SkipNet: Train Loss {tr_loss_s:.4f}, Val Acc {val_acc_s:.2f}% | "
            f"PlainNet: Train Loss {tr_loss_p:.4f}, Val Acc {val_acc_p:.2f}%"
        )

    # Detailed final evaluation on SkipNet
    print("\n[Evaluation] Computing comprehensive validation metrics for SkipNet-14...")
    final_val_loss_s, final_val_acc_s, y_true, y_pred, y_prob = evaluate(model_skip, test_loader, criterion, device)
    print(f"SkipNet Final Validation Accuracy: {final_val_acc_s:.2f}% | Final Loss: {final_val_loss_s:.4f}")
    print("\nSkipNet Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, digits=4))

    # Save Checkpoint
    checkpoint_dir = SCRIPT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "custom_skip_cnn_cifar10.pt"
    torch.save({
        "model_state_dict": model_skip.state_dict(),
        "final_accuracy": final_val_acc_s,
        "classes": CIFAR10_CLASSES,
        "history_skip": history_skip,
        "history_plain": history_plain,
    }, checkpoint_path)
    print(f"[Checkpoint] Model saved to {checkpoint_path}")

    # Generate Visual Artifacts
    artifact_path = SCRIPT_DIR / "custom_cnn_skip_connections_results.png"
    generate_visual_artifacts(
        history_skip=history_skip,
        history_plain=history_plain,
        grad_skip=latest_grads_skip,
        grad_plain=latest_grads_plain,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        test_dataset=full_test_dataset,
        params_skip=params_skip,
        params_plain=params_plain,
        save_path=artifact_path,
    )
    print("[Done] Custom CNN with Skip Connections pipeline executed successfully.")


if __name__ == "__main__":
    main()

