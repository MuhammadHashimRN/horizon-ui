"""Generate beautiful confusion matrix heatmaps and metric charts for presentation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "metrics_output"

# ---------- Color palette ----------
BG_COLOR = "#0F1117"
CARD_BG = "#1A1B26"
TEXT_COLOR = "#E0E0E0"
ACCENT_BLUE = "#00D4FF"
ACCENT_GREEN = "#00E676"
ACCENT_RED = "#FF5252"
ACCENT_ORANGE = "#FFB74D"
GRID_COLOR = "#2A2B36"


def setup_dark_style():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "font.family": "Segoe UI",
        "font.size": 11,
    })


def create_custom_cmap():
    """Dark blue to cyan colormap."""
    colors = ["#0F1117", "#0A2647", "#144272", "#205295", "#2C7DA0", "#00D4FF"]
    return mcolors.LinearSegmentedColormap.from_list("horizon", colors, N=256)


def plot_confusion_matrix(cm, labels, title, filename, figsize=None):
    """Generate a polished confusion matrix heatmap."""
    n = len(labels)
    if figsize is None:
        figsize = (max(10, n * 0.9), max(8, n * 0.75))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cmap = create_custom_cmap()

    # Normalize for color intensity (row-wise)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    im = ax.imshow(cm_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate cells with counts
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val == 0:
                continue
            color = "#FFFFFF" if cm_norm[i, j] > 0.5 else "#AAAAAA"
            weight = "bold" if i == j else "normal"
            fontsize = 9 if n > 15 else 11
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=fontsize, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    # Shorten long labels
    short = [l[:12] + "." if len(l) > 13 else l for l in labels]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15, color=ACCENT_BLUE)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalized)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Highlight diagonal
    for i in range(n):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor=ACCENT_GREEN, linewidth=1.5, alpha=0.6)
        ax.add_patch(rect)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metric_comparison(gru_metrics, cnn_metrics, filename):
    """Bar chart comparing GRU vs CNN1D metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    gru_vals = [gru_metrics["accuracy"], gru_metrics["precision"],
                gru_metrics["recall"], gru_metrics["f1_weighted"]]
    cnn_vals = [cnn_metrics["accuracy"], cnn_metrics["precision"],
                cnn_metrics["recall"], cnn_metrics["f1_weighted"]]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, gru_vals, width, label="GRU",
                   color="#2C7DA0", edgecolor="#00D4FF", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, cnn_vals, width, label="CNN1D",
                   color="#00D4FF", edgecolor="#FFFFFF", linewidth=0.8)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2%}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#2C7DA0")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2%}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=ACCENT_BLUE)

    # Target line
    ax.axhline(y=0.85, color=ACCENT_GREEN, linestyle="--", linewidth=1.5, alpha=0.7, label="SRS Target (F1 >= 0.85)")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Gesture Recognition: GRU vs CNN1D", fontsize=16, fontweight="bold",
                 pad=15, color=ACCENT_BLUE)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_latency_benchmarks(latency_data, filename):
    """Horizontal bar chart for latency benchmarks."""
    fig, ax = plt.subplots(figsize=(10, 5))

    components = list(latency_data.keys())
    values = list(latency_data.values())
    targets = {
        "EMA Filter": 1.0, "Adaptive EMA Filter": 1.0, "Kalman Filter 2D": 1.0,
        "Feature Extraction": 10.0, "EventBus Publish (3 subs)": 1.0,
        "GRU Model Inference": 5.0, "CNN1D Model Inference": 5.0,
    }

    colors = []
    for comp, val in zip(components, values):
        target = targets.get(comp, 10.0)
        colors.append(ACCENT_GREEN if val < target * 0.5 else ACCENT_BLUE if val < target else ACCENT_RED)

    bars = ax.barh(range(len(components)), values, color=colors, edgecolor="#333", height=0.6)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + max(values) * 0.02, i,
                f"{val:.4f} ms", va="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=10)
    ax.set_xlabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Component Latency Benchmarks", fontsize=16, fontweight="bold",
                 pad=15, color=ACCENT_BLUE)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_voice_per_category(category_data, filename):
    """Bar chart for voice recognition per-category accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))

    categories = sorted(category_data.keys())
    accuracies = []
    for cat in categories:
        d = category_data[cat]
        accuracies.append(d["correct"] / d["total"] if d["total"] > 0 else 0)

    colors = [ACCENT_GREEN if a >= 0.9 else ACCENT_BLUE if a >= 0.75 else ACCENT_ORANGE if a >= 0.5 else ACCENT_RED
              for a in accuracies]

    bars = ax.bar(range(len(categories)), accuracies, color=colors, edgecolor="#333", width=0.7)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    ax.axhline(y=0.90, color=ACCENT_GREEN, linestyle="--", linewidth=1.5, alpha=0.5, label="90% Target")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Voice Command Recognition: Per-Category Accuracy", fontsize=16,
                 fontweight="bold", pad=15, color=ACCENT_BLUE)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_voice_metrics_summary(metrics, filename):
    """Single summary card for voice metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    items = [
        ("Accuracy", metrics["accuracy"], 0.90),
        ("Precision", metrics["precision"], 0.85),
        ("Recall", metrics["recall"], 0.85),
        ("F1 Score (Weighted)", metrics["f1_weighted"], 0.85),
    ]

    ax.set_title("Voice Command Recognition Metrics", fontsize=18, fontweight="bold",
                 pad=20, color=ACCENT_BLUE)

    for i, (name, value, target) in enumerate(items):
        y = 0.80 - i * 0.20
        status_color = ACCENT_GREEN if value >= target else ACCENT_ORANGE

        ax.text(0.05, y, name, fontsize=14, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.55, y, f"{value:.2%}", fontsize=20, fontweight="bold",
                transform=ax.transAxes, va="center", color=status_color)
        ax.text(0.80, y, f"(target: {target:.0%})", fontsize=10,
                transform=ax.transAxes, va="center", color="#888888")

        # Progress bar
        bar_y = y - 0.06
        ax.barh(bar_y, value, height=0.03, color=status_color, alpha=0.7, transform=ax.transAxes)
        ax.barh(bar_y, 1.0, height=0.03, color=GRID_COLOR, alpha=0.3, transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dataset_breakdown(gesture_data, voice_data, filename):
    """Side-by-side pie/donut charts for dataset composition."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gesture dataset
    g_labels = ["Training (80%)", "Validation (20%)"]
    g_sizes = [gesture_data["training_split"]["train"], gesture_data["training_split"]["val"]]
    g_colors = [ACCENT_BLUE, "#2C7DA0"]
    wedges1, texts1, autotexts1 = ax1.pie(g_sizes, labels=g_labels, colors=g_colors,
                                           autopct="%1.0f%%", startangle=90,
                                           textprops={"color": TEXT_COLOR, "fontsize": 11},
                                           wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2})
    for t in autotexts1:
        t.set_fontweight("bold")
    ax1.set_title(f"Gesture Dataset\n{gesture_data['total_samples']} samples, {gesture_data['num_classes']} classes",
                  fontsize=14, fontweight="bold", color=ACCENT_BLUE, pad=15)

    # Voice dataset
    v_labels = [f"Exact ({voice_data['exact_transcripts']})",
                f"Fuzzy ({voice_data['fuzzy_typo']})",
                f"ASR Noisy ({voice_data['asr_noisy']})",
                f"Negative ({voice_data['negative_samples']})"]
    v_sizes = [voice_data["exact_transcripts"], voice_data["fuzzy_typo"],
               voice_data["asr_noisy"], voice_data["negative_samples"]]
    v_colors = [ACCENT_GREEN, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_RED]
    wedges2, texts2, autotexts2 = ax2.pie(v_sizes, labels=v_labels, colors=v_colors,
                                           autopct="%1.0f%%", startangle=90,
                                           textprops={"color": TEXT_COLOR, "fontsize": 10},
                                           wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2})
    for t in autotexts2:
        t.set_fontweight("bold")
    ax2.set_title(f"Voice Test Dataset\n{voice_data['total_samples']} samples, {voice_data['command_count']} commands",
                  fontsize=14, fontweight="bold", color=ACCENT_BLUE, pad=15)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    setup_dark_style()
    os.makedirs(output_dir, exist_ok=True)

    # Load gesture metrics
    gesture_path = output_dir / "metrics_results.json"
    with open(gesture_path) as f:
        gesture_data = json.load(f)

    # Load voice metrics
    voice_path = output_dir / "voice_metrics_results.json"
    with open(voice_path) as f:
        voice_data = json.load(f)

    print("=" * 60)
    print("GENERATING PRESENTATION VISUALIZATIONS")
    print("=" * 60)
    print()

    # 1. Gesture confusion matrices
    print("[1/8] Gesture Confusion Matrix - GRU")
    gru_cm = np.array(gesture_data["gru_model"]["confusion_matrix"])
    g_labels = gesture_data["gru_model"]["class_names"]
    plot_confusion_matrix(gru_cm, g_labels,
                          "Gesture Recognition: GRU Confusion Matrix",
                          "gesture_cm_gru.png")

    print("[2/8] Gesture Confusion Matrix - CNN1D")
    cnn_cm = np.array(gesture_data["cnn_model"]["confusion_matrix"])
    plot_confusion_matrix(cnn_cm, g_labels,
                          "Gesture Recognition: CNN1D Confusion Matrix",
                          "gesture_cm_cnn1d.png")

    # 2. Voice confusion matrix
    print("[3/8] Voice Confusion Matrix")
    voice_cm = np.array(voice_data["confusion_matrix"])
    v_labels = voice_data["labels"]
    plot_confusion_matrix(voice_cm, v_labels,
                          "Voice Command Recognition: Confusion Matrix",
                          "voice_cm.png",
                          figsize=(16, 13))

    # 3. Model comparison chart
    print("[4/8] GRU vs CNN1D Comparison")
    gru_m = gesture_data["gru_model"]
    cnn_m = gesture_data["cnn_model"]
    plot_metric_comparison(
        {"accuracy": gru_m["accuracy"], "precision": gru_m["precision"],
         "recall": gru_m["recall"], "f1_weighted": gru_m["f1_weighted"]},
        {"accuracy": cnn_m["accuracy"], "precision": cnn_m["precision"],
         "recall": cnn_m["recall"], "f1_weighted": cnn_m["f1_weighted"]},
        "gesture_model_comparison.png")

    # 4. Latency benchmarks
    print("[5/8] Latency Benchmarks")
    plot_latency_benchmarks(gesture_data["latency_benchmarks"], "latency_benchmarks.png")

    # 5. Voice per-category accuracy
    print("[6/8] Voice Per-Category Accuracy")
    plot_voice_per_category(voice_data["per_category"], "voice_per_category.png")

    # 6. Voice metrics summary
    print("[7/8] Voice Metrics Summary Card")
    plot_voice_metrics_summary(voice_data["metrics"], "voice_metrics_summary.png")

    # 7. Dataset breakdown
    print("[8/8] Dataset Breakdown")
    gesture_ds = {
        "total_samples": gesture_data["dataset"]["total_samples"],
        "num_classes": gesture_data["dataset"]["num_classes"],
        "training_split": {
            "train": gesture_data["dataset"]["train_size"],
            "val": gesture_data["dataset"]["val_size"],
        },
    }
    voice_ds = voice_data["dataset"]
    plot_dataset_breakdown(gesture_ds, voice_ds, "dataset_breakdown.png")

    print()
    print("=" * 60)
    print(f"All 8 visualizations saved to: {output_dir}")
    print("=" * 60)
    print()
    print("Files generated:")
    for f in sorted(output_dir.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
