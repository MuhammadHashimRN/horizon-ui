"""Export trained PyTorch gesture model to ONNX format with quantization."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from train_gesture_model import GestureCNN1D, GestureGRU, GESTURE_CLASSES


def export(args: argparse.Namespace) -> None:
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_type = checkpoint.get("model_type", "gru")

    if model_type == "gru":
        model = GestureGRU(num_classes=len(GESTURE_CLASSES))
    else:
        model = GestureCNN1D(num_classes=len(GESTURE_CLASSES), seq_len=args.window)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded {model_type} model from {args.checkpoint}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")

    # Create dummy input
    dummy_input = torch.randn(1, args.window, 63)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"ONNX model exported to {args.output}")

    # Quantize if requested
    if args.quantize:
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_output = args.output.replace(".onnx", "_int8.onnx")
            quantize_dynamic(
                args.output,
                quantized_output,
                weight_type=QuantType.QInt8,
            )
            print(f"Quantized model saved to {quantized_output}")
        except ImportError:
            print("onnxruntime.quantization not available, skipping quantization")

    # Verify
    import onnxruntime as ort

    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, args.window, 63).astype(np.float32)
    result = session.run(None, {"input": test_input})
    print(f"Verification: output shape = {result[0].shape}")
    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export gesture model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="../models/gesture_classifier.onnx", help="ONNX output path")
    parser.add_argument("--window", type=int, default=15, help="Temporal window size")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 dynamic quantization")
    export(parser.parse_args())
