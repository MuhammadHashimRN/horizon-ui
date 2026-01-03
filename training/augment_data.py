"""Data augmentation utilities for gesture training data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def add_noise(sequence: np.ndarray, std: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to landmark coordinates."""
    noise = np.random.normal(0, std, sequence.shape)
    return sequence + noise


def scale(sequence: np.ndarray, factor_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Scale landmark coordinates by a random factor."""
    factor = np.random.uniform(*factor_range)
    return sequence * factor


def time_warp(sequence: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """Apply random time warping by interpolating along the time axis."""
    T = len(sequence)
    orig_steps = np.arange(T)
    warp = np.cumsum(np.random.normal(1, sigma, T))
    warp = warp * (T - 1) / warp[-1]

    warped = np.zeros_like(sequence)
    for col in range(sequence.shape[1]):
        warped[:, col] = np.interp(orig_steps, warp, sequence[:, col])
    return warped


def mirror(sequence: np.ndarray) -> np.ndarray:
    """Mirror hand landmarks horizontally (swap left/right hand)."""
    mirrored = sequence.copy()
    # Flip x coordinates (every 3rd starting from 0)
    mirrored[:, 0::3] = 1.0 - mirrored[:, 0::3]
    return mirrored


def augment_dataset(input_dir: str, output_dir: str, augmentations_per_sample: int = 3) -> None:
    """Augment all samples in a dataset directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    total_original = 0
    total_augmented = 0

    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue

        out_class = output_path / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)

        for npy_file in class_dir.glob("*.npy"):
            sequence = np.load(npy_file)
            total_original += 1

            # Copy original
            np.save(out_class / npy_file.name, sequence)

            # Generate augmentations
            for i in range(augmentations_per_sample):
                augmented = sequence.copy()

                # Randomly apply augmentations
                if np.random.rand() < 0.7:
                    augmented = add_noise(augmented)
                if np.random.rand() < 0.5:
                    augmented = scale(augmented)
                if np.random.rand() < 0.3:
                    augmented = time_warp(augmented)
                if np.random.rand() < 0.3:
                    augmented = mirror(augmented)

                aug_name = f"{npy_file.stem}_aug{i}.npy"
                np.save(out_class / aug_name, augmented.astype(np.float32))
                total_augmented += 1

    print(f"Augmentation complete: {total_original} originals → {total_original + total_augmented} total samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment gesture training data")
    parser.add_argument("--input", type=str, default="dataset/processed", help="Input dataset directory")
    parser.add_argument("--output", type=str, default="dataset/augmented", help="Output directory")
    parser.add_argument("--count", type=int, default=3, help="Augmentations per sample")
    args = parser.parse_args()
    augment_dataset(args.input, args.output, args.count)
