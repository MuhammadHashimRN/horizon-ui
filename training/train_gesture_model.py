"""PyTorch training script for GRU/1D-CNN gesture classifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Gesture classes matching the runtime classifier
GESTURE_CLASSES = [
    "none", "point", "pinch", "fist", "open_palm",
    "swipe_left", "swipe_right", "pinch_spread", "pinch_close",
    "two_finger_scroll", "thumbs_up",
]


class GestureDataset(Dataset):
    """Dataset of landmark sequences labeled by gesture class."""

    def __init__(self, data_dir: str, temporal_window: int = 15) -> None:
        self.samples: list[tuple[np.ndarray, int]] = []
        self.temporal_window = temporal_window

        data_path = Path(data_dir)
        for class_idx, class_name in enumerate(GESTURE_CLASSES):
            class_dir = data_path / class_name
            if not class_dir.exists():
                continue

            for npy_file in class_dir.glob("*.npy"):
                sequence = np.load(npy_file)
                # Sliding window over the sequence
                for start in range(0, len(sequence) - temporal_window + 1, temporal_window // 2):
                    window = sequence[start : start + temporal_window]
                    if len(window) == temporal_window:
                        self.samples.append((window.astype(np.float32), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        data, label = self.samples[idx]
        return torch.from_numpy(data), label


class GestureGRU(nn.Module):
    """GRU-based gesture classifier."""

    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 11,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        output, _ = self.gru(x)
        # Use last timestep
        last_output = output[:, -1, :]
        return self.classifier(last_output)


class GestureCNN1D(nn.Module):
    """1D-CNN gesture classifier."""

    def __init__(
        self,
        input_size: int = 63,
        num_classes: int = 11,
        seq_len: int = 15,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset
    dataset = GestureDataset(args.data, temporal_window=args.window)
    print(f"Dataset: {len(dataset)} samples")

    if len(dataset) == 0:
        print("No training data found. See training/README.md for dataset instructions.")
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Model
    if args.model == "gru":
        model = GestureGRU(num_classes=len(GESTURE_CLASSES)).to(device)
    else:
        model = GestureCNN1D(num_classes=len(GESTURE_CLASSES), seq_len=args.window).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

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

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

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
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "model_type": args.model,
                "classes": GESTURE_CLASSES,
            }
            torch.save(checkpoint, os.path.join(args.output, "best_model.pt"))
            print(f"  → Best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--data", type=str, default="dataset/processed", help="Path to processed dataset")
    parser.add_argument("--output", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "cnn"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--window", type=int, default=15, help="Temporal window size (frames)")
    train(parser.parse_args())
