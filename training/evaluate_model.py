"""Evaluate gesture classifier model with metrics and confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from train_gesture_model import GESTURE_CLASSES, GestureDataset


def evaluate(args: argparse.Namespace) -> None:
    import onnxruntime as ort
    from sklearn.metrics import classification_report, confusion_matrix

    # Load model
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Model loaded: {args.model}")

    # Load dataset
    dataset = GestureDataset(args.data, temporal_window=args.window)
    print(f"Dataset: {len(dataset)} samples")

    if len(dataset) == 0:
        print("No evaluation data found.")
        return

    y_true = []
    y_pred = []

    for i in range(len(dataset)):
        data, label = dataset[i]
        input_data = data.numpy().reshape(1, args.window, -1).astype(np.float32)

        outputs = session.run(None, {input_name: input_data})
        probs = outputs[0][0]

        # Softmax if raw logits
        if np.any(probs < 0):
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / exp_p.sum()

        predicted = int(np.argmax(probs))
        y_true.append(label)
        y_pred.append(predicted)

    # Metrics
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    present_classes = sorted(set(y_true) | set(y_pred))
    target_names = [GESTURE_CLASSES[i] for i in present_classes]
    print(classification_report(y_true, y_pred, labels=present_classes, target_names=target_names))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    print(cm)

    # Overall metrics
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Per-class F1 (weighted)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Weighted F1 Score: {f1:.4f}")

    # Check against SRS target
    target_f1 = 0.85
    if f1 >= target_f1:
        print(f"✓ F1 score meets SRS target (>= {target_f1})")
    else:
        print(f"✗ F1 score below SRS target ({f1:.4f} < {target_f1})")

    # Save results
    if args.output:
        results = {
            "accuracy": float(accuracy),
            "f1_weighted": float(f1),
            "confusion_matrix": cm.tolist(),
            "classes": GESTURE_CLASSES,
        }
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gesture classifier")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--data", type=str, default="dataset/processed", help="Path to processed dataset")
    parser.add_argument("--window", type=int, default=15, help="Temporal window size")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    evaluate(parser.parse_args())
