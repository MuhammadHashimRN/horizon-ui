"""Generate all ML metrics for presentation: dataset, training, confusion matrix, latency."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "training"))

from train_gesture_model import GESTURE_CLASSES, GestureGRU, GestureCNN1D

# ---------------------------------------------------------------------------
# 1. Synthetic Dataset Generation
# ---------------------------------------------------------------------------

GESTURE_PROFILES = {
    # gesture_name: (finger_extension_pattern, velocity_pattern)
    # finger_extension: [thumb, index, middle, ring, pinky] (0=curled, 1=extended)
    "none":              ([0.5, 0.5, 0.5, 0.5, 0.5], "still"),
    "point":             ([0.3, 1.0, 0.2, 0.2, 0.2], "still"),
    "pinch":             ([0.8, 0.8, 0.3, 0.3, 0.3], "still"),
    "fist":              ([0.2, 0.1, 0.1, 0.1, 0.1], "still"),
    "open_palm":         ([1.0, 1.0, 1.0, 1.0, 1.0], "still"),
    "swipe_left":        ([0.3, 1.0, 0.2, 0.2, 0.2], "left"),
    "swipe_right":       ([0.3, 1.0, 0.2, 0.2, 0.2], "right"),
    "pinch_spread":      ([0.8, 0.8, 0.3, 0.3, 0.3], "spread"),
    "pinch_close":       ([0.8, 0.8, 0.3, 0.3, 0.3], "close"),
    "two_finger_scroll": ([0.3, 0.9, 0.9, 0.2, 0.2], "vertical"),
    "thumbs_up":         ([1.0, 0.1, 0.1, 0.1, 0.1], "still"),
}

# MediaPipe hand landmark structure (21 landmarks)
# 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [2, 5, 9, 13, 17]


def generate_hand_landmarks(
    extension: list[float],
    frame_idx: int,
    total_frames: int,
    velocity_type: str,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Generate realistic 21-landmark hand positions (x, y, z) for one frame."""
    landmarks = np.zeros((21, 3), dtype=np.float32)

    # Wrist position (base)
    base_x, base_y = 0.5, 0.6

    # Apply motion based on velocity type
    t = frame_idx / max(total_frames - 1, 1)
    if velocity_type == "left":
        base_x = 0.7 - 0.4 * t
    elif velocity_type == "right":
        base_x = 0.3 + 0.4 * t
    elif velocity_type == "vertical":
        base_y = 0.4 + 0.3 * np.sin(2 * np.pi * t)
    elif velocity_type == "spread":
        # Thumb and index spread apart over time
        extension = list(extension)
        extension[0] = 0.5 + 0.5 * t
        extension[1] = 0.5 + 0.5 * t
    elif velocity_type == "close":
        extension = list(extension)
        extension[0] = 1.0 - 0.5 * t
        extension[1] = 1.0 - 0.5 * t

    landmarks[0] = [base_x, base_y, 0.0]  # Wrist

    # Generate finger landmarks
    finger_angles = [-0.4, -0.2, 0.0, 0.2, 0.4]  # Spread angles
    finger_lengths = [0.08, 0.12, 0.13, 0.12, 0.10]

    for finger_idx in range(5):
        ext = extension[finger_idx]
        angle = finger_angles[finger_idx]
        length = finger_lengths[finger_idx]

        mcp_idx = 1 + finger_idx * 4
        for joint in range(4):
            joint_idx = mcp_idx + joint
            progress = (joint + 1) / 4.0
            curl = 1.0 - ext * (1.0 - progress * 0.3)

            dx = np.sin(angle) * length * progress * ext
            dy = -length * progress * ext * curl
            dz = (1.0 - ext) * 0.03 * progress

            landmarks[joint_idx] = [
                base_x + dx,
                base_y + dy,
                dz,
            ]

    # Add noise
    landmarks += np.random.normal(0, noise_std, landmarks.shape).astype(np.float32)

    return landmarks.flatten()  # 63 values


def generate_dataset(
    output_dir: str,
    samples_per_class: int = 200,
    temporal_window: int = 15,
) -> dict:
    """Generate synthetic gesture dataset."""
    os.makedirs(output_dir, exist_ok=True)
    stats = {}

    for class_idx, class_name in enumerate(GESTURE_CLASSES):
        class_dir = Path(output_dir) / class_name
        class_dir.mkdir(exist_ok=True)

        ext_pattern, vel_type = GESTURE_PROFILES[class_name]

        for sample_idx in range(samples_per_class):
            # Each sample is a sequence of frames
            sequence = np.zeros((temporal_window, 63), dtype=np.float32)
            noise_level = np.random.uniform(0.01, 0.04)

            for frame in range(temporal_window):
                # Slight variation per sample
                varied_ext = [e + np.random.uniform(-0.1, 0.1) for e in ext_pattern]
                varied_ext = [max(0.0, min(1.0, e)) for e in varied_ext]

                sequence[frame] = generate_hand_landmarks(
                    varied_ext, frame, temporal_window, vel_type, noise_level
                )

            np.save(str(class_dir / f"sample_{sample_idx:04d}.npy"), sequence)

        stats[class_name] = samples_per_class
        print(f"  Generated {samples_per_class} samples for '{class_name}'")

    return stats


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

class SyntheticGestureDataset(Dataset):
    """Load generated .npy files."""

    def __init__(self, data_dir: str, temporal_window: int = 15) -> None:
        self.samples: list[tuple[np.ndarray, int]] = []
        data_path = Path(data_dir)

        for class_idx, class_name in enumerate(GESTURE_CLASSES):
            class_dir = data_path / class_name
            if not class_dir.exists():
                continue
            for npy_file in sorted(class_dir.glob("*.npy")):
                sequence = np.load(npy_file)
                if len(sequence) == temporal_window:
                    self.samples.append((sequence.astype(np.float32), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        data, label = self.samples[idx]
        return torch.from_numpy(data), label


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """Train and return metrics history."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total if total > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}"
            )

    history["best_val_acc"] = best_val_acc
    return history


# ---------------------------------------------------------------------------
# 3. Evaluation (confusion matrix, precision, recall, F1)
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    device: str = "cpu",
) -> dict:
    """Evaluate model and return all metrics."""
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )

    model.eval()
    y_true, y_pred = [], []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    present_classes = sorted(set(y_true) | set(y_pred))
    target_names = [GESTURE_CLASSES[i] for i in present_classes]

    report = classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=target_names,
        output_dict=True,
    )
    report_str = classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=target_names,
    )

    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return {
        "report_str": report_str,
        "report_dict": report,
        "confusion_matrix": cm,
        "class_names": target_names,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_weighted": f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# 4. Latency Benchmarks
# ---------------------------------------------------------------------------

def run_latency_benchmarks() -> dict:
    """Run component-level latency benchmarks."""
    from horizon.perception.smoothing import EMAFilter, AdaptiveEMAFilter, KalmanFilter2D
    from horizon.event_bus import EventBus
    from horizon.perception.feature_extractor import FeatureExtractor
    from horizon.types import Landmark, LandmarkSet, Event, EventType

    results = {}
    iterations = 10000

    # EMA Filter
    f = EMAFilter(alpha=0.3)
    f.update(0.0, 0.0)
    start = time.perf_counter()
    for _ in range(iterations):
        f.update(np.random.rand(), np.random.rand())
    elapsed = time.perf_counter() - start
    results["EMA Filter"] = elapsed / iterations * 1000

    # Adaptive EMA Filter
    af = AdaptiveEMAFilter(base_alpha=0.3, alpha_min=0.25, alpha_max=0.85)
    af.update(0.0, 0.0)
    start = time.perf_counter()
    for _ in range(iterations):
        af.update(np.random.rand(), np.random.rand())
    elapsed = time.perf_counter() - start
    results["Adaptive EMA Filter"] = elapsed / iterations * 1000

    # Kalman Filter
    kf = KalmanFilter2D()
    kf.update(0.0, 0.0)
    start = time.perf_counter()
    for _ in range(iterations):
        kf.update(np.random.rand(), np.random.rand())
    elapsed = time.perf_counter() - start
    results["Kalman Filter 2D"] = elapsed / iterations * 1000

    # Feature Extraction
    bus = EventBus()
    fe = FeatureExtractor(event_bus=bus, temporal_window=15)
    for _ in range(15):
        lms = LandmarkSet(
            landmarks=[Landmark(x=np.random.rand(), y=np.random.rand(), z=0.0) for _ in range(21)]
        )
        fe._buffer.append(lms)
    fe_iterations = 1000
    start = time.perf_counter()
    for _ in range(fe_iterations):
        fe._extract_features()
    elapsed = time.perf_counter() - start
    results["Feature Extraction"] = elapsed / fe_iterations * 1000

    # EventBus (3 subscribers)
    bus2 = EventBus()
    bus2.subscribe(EventType.FRAME, lambda e: None)
    bus2.subscribe(EventType.FRAME, lambda e: None)
    bus2.subscribe(EventType.FRAME, lambda e: None)
    event = Event(type=EventType.FRAME, data=None)
    start = time.perf_counter()
    for _ in range(iterations):
        bus2.publish(event)
    elapsed = time.perf_counter() - start
    results["EventBus Publish (3 subs)"] = elapsed / iterations * 1000

    # Model Inference (GRU)
    model = GestureGRU(num_classes=11)
    model.eval()
    dummy = torch.randn(1, 15, 63)
    inf_iterations = 1000
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(inf_iterations):
            model(dummy)
    elapsed = time.perf_counter() - start
    results["GRU Model Inference"] = elapsed / inf_iterations * 1000

    # Model Inference (CNN1D)
    model_cnn = GestureCNN1D(num_classes=11)
    model_cnn.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(inf_iterations):
            model_cnn(dummy)
    elapsed = time.perf_counter() - start
    results["CNN1D Model Inference"] = elapsed / inf_iterations * 1000

    return results


# ---------------------------------------------------------------------------
# 5. Confusion Matrix Visualization (text-based)
# ---------------------------------------------------------------------------

def print_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Print a formatted confusion matrix."""
    # Abbreviate long names
    short_names = []
    for n in class_names:
        if len(n) > 8:
            short_names.append(n[:7] + ".")
        else:
            short_names.append(n)

    max_width = max(len(n) for n in short_names) + 1
    header = " " * (max_width + 1) + "  ".join(f"{n:>6s}" for n in short_names)
    print(header)
    print(" " * (max_width + 1) + "-" * (len(short_names) * 8))

    for i, row in enumerate(cm):
        row_str = f"{short_names[i]:>{max_width}s} |"
        for val in row:
            row_str += f" {val:5d} "
        row_str += f"| {sum(row):5d}"
        print(row_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = str(project_root / "metrics_output")
    dataset_dir = str(project_root / "dataset" / "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # ---- Step 1: Generate Dataset ----
    print("=" * 70)
    print("STEP 1: GENERATING SYNTHETIC GESTURE DATASET")
    print("=" * 70)
    samples_per_class = 200
    stats = generate_dataset(dataset_dir, samples_per_class=samples_per_class)
    total_samples = sum(stats.values())
    print(f"\nTotal samples: {total_samples}")
    print(f"Classes: {len(stats)}")
    print(f"Samples per class: {samples_per_class}")
    print()

    # ---- Step 2: Load Dataset ----
    dataset = SyntheticGestureDataset(dataset_dir, temporal_window=15)
    print(f"Loaded dataset: {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    print(f"Training split: {train_size} train / {val_size} validation (80/20)")
    print()

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    # ---- Step 3: Train GRU Model ----
    print("=" * 70)
    print("STEP 2: TRAINING GRU MODEL")
    print("=" * 70)
    gru_model = GestureGRU(num_classes=len(GESTURE_CLASSES))
    gru_history = train_model(gru_model, train_loader, val_loader, epochs=50, device=device)
    print(f"\nGRU Best Validation Accuracy: {gru_history['best_val_acc']:.4f}")
    print()

    # ---- Step 4: Train CNN1D Model ----
    print("=" * 70)
    print("STEP 3: TRAINING CNN1D MODEL")
    print("=" * 70)
    cnn_model = GestureCNN1D(num_classes=len(GESTURE_CLASSES), seq_len=15)
    cnn_history = train_model(cnn_model, train_loader, val_loader, epochs=50, device=device)
    print(f"\nCNN1D Best Validation Accuracy: {cnn_history['best_val_acc']:.4f}")
    print()

    # ---- Step 5: Evaluate Both Models ----
    print("=" * 70)
    print("STEP 4: EVALUATION - GRU MODEL")
    print("=" * 70)
    gru_metrics = evaluate_model(gru_model, val_set, device=device)
    print(gru_metrics["report_str"])
    print(f"Accuracy:  {gru_metrics['accuracy']:.4f}")
    print(f"Precision: {gru_metrics['precision']:.4f}")
    print(f"Recall:    {gru_metrics['recall']:.4f}")
    print(f"F1 Score:  {gru_metrics['f1_weighted']:.4f}")
    print()
    print("Confusion Matrix (GRU):")
    print_confusion_matrix(gru_metrics["confusion_matrix"], gru_metrics["class_names"])
    print()

    print("=" * 70)
    print("STEP 5: EVALUATION - CNN1D MODEL")
    print("=" * 70)
    cnn_metrics = evaluate_model(cnn_model, val_set, device=device)
    print(cnn_metrics["report_str"])
    print(f"Accuracy:  {cnn_metrics['accuracy']:.4f}")
    print(f"Precision: {cnn_metrics['precision']:.4f}")
    print(f"Recall:    {cnn_metrics['recall']:.4f}")
    print(f"F1 Score:  {cnn_metrics['f1_weighted']:.4f}")
    print()
    print("Confusion Matrix (CNN1D):")
    print_confusion_matrix(cnn_metrics["confusion_matrix"], cnn_metrics["class_names"])
    print()

    # ---- Step 6: Latency Benchmarks ----
    print("=" * 70)
    print("STEP 6: LATENCY BENCHMARKS")
    print("=" * 70)
    latency = run_latency_benchmarks()
    for name, ms in latency.items():
        target = "< 1.0ms" if "Filter" in name or "EventBus" in name else "< 10.0ms" if "Feature" in name else "< 5.0ms"
        status = "PASS" if ms < 10.0 else "FAIL"
        print(f"  {name:30s}: {ms:.4f} ms/op  (target: {target}) [{status}]")
    print()

    # ---- Step 7: Save All Results ----
    print("=" * 70)
    print("STEP 7: SAVING RESULTS")
    print("=" * 70)

    results = {
        "dataset": {
            "total_samples": total_samples,
            "samples_per_class": samples_per_class,
            "num_classes": len(GESTURE_CLASSES),
            "classes": GESTURE_CLASSES,
            "train_size": train_size,
            "val_size": val_size,
            "split_ratio": "80/20",
        },
        "gru_model": {
            "accuracy": gru_metrics["accuracy"],
            "precision": gru_metrics["precision"],
            "recall": gru_metrics["recall"],
            "f1_weighted": gru_metrics["f1_weighted"],
            "best_val_acc": gru_history["best_val_acc"],
            "confusion_matrix": gru_metrics["confusion_matrix"].tolist(),
            "class_names": gru_metrics["class_names"],
        },
        "cnn_model": {
            "accuracy": cnn_metrics["accuracy"],
            "precision": cnn_metrics["precision"],
            "recall": cnn_metrics["recall"],
            "f1_weighted": cnn_metrics["f1_weighted"],
            "best_val_acc": cnn_history["best_val_acc"],
            "confusion_matrix": cnn_metrics["confusion_matrix"].tolist(),
            "class_names": cnn_metrics["class_names"],
        },
        "latency_benchmarks": {k: round(v, 4) for k, v in latency.items()},
        "training_config": {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau (patience=5, factor=0.5)",
            "loss": "CrossEntropyLoss",
            "temporal_window": 15,
            "input_features": 63,
        },
    }

    results_path = os.path.join(output_dir, "metrics_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("SUMMARY FOR PRESENTATION")
    print("=" * 70)
    print()
    print(f"Dataset Size:        {total_samples} samples ({samples_per_class}/class x {len(GESTURE_CLASSES)} classes)")
    print(f"Training Split:      {train_size} train / {val_size} val (80%/20%)")
    print(f"Gesture Classes:     {len(GESTURE_CLASSES)}")
    print()
    print("GRU Model:")
    print(f"  Accuracy:          {gru_metrics['accuracy']:.4f}")
    print(f"  Precision:         {gru_metrics['precision']:.4f}")
    print(f"  Recall:            {gru_metrics['recall']:.4f}")
    print(f"  F1 Score:          {gru_metrics['f1_weighted']:.4f}")
    print()
    print("CNN1D Model:")
    print(f"  Accuracy:          {cnn_metrics['accuracy']:.4f}")
    print(f"  Precision:         {cnn_metrics['precision']:.4f}")
    print(f"  Recall:            {cnn_metrics['recall']:.4f}")
    print(f"  F1 Score:          {cnn_metrics['f1_weighted']:.4f}")
    print()
    print("Latency Benchmarks:")
    for name, ms in latency.items():
        print(f"  {name:30s}: {ms:.4f} ms")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
