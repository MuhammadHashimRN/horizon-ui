"""Utility for recording gesture training data from webcam."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def collect(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils

    print(f"Recording gesture: {args.gesture}")
    print(f"Output: {output_dir}")
    print(f"Duration per sample: {args.duration}s")
    print(f"Samples to collect: {args.samples}")
    print("\nPress 'r' to start recording, 'q' to quit.")

    sample_idx = 0
    recording = False
    frames: list[np.ndarray] = []
    start_time = 0.0

    while sample_idx < args.samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Status display
        if recording:
            elapsed = time.time() - start_time
            remaining = args.duration - elapsed
            cv2.putText(frame, f"RECORDING ({remaining:.1f}s)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Extract landmarks
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                coords = []
                for point in lm.landmark:
                    coords.extend([point.x, point.y, point.z])
                frames.append(coords)

            if elapsed >= args.duration:
                recording = False
                if len(frames) > 0:
                    sequence = np.array(frames, dtype=np.float32)
                    filename = f"{args.gesture}_{sample_idx:04d}.npy"
                    np.save(output_dir / filename, sequence)
                    print(f"  Saved {filename} ({len(frames)} frames)")
                    sample_idx += 1
                else:
                    print("  No landmarks detected, sample discarded")
                frames = []
        else:
            cv2.putText(frame, f"Sample {sample_idx + 1}/{args.samples} | Press 'r' to record",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {args.gesture}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Gesture Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r") and not recording:
            recording = True
            frames = []
            start_time = time.time()
            print(f"  Recording sample {sample_idx + 1}...")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"\nCollection complete: {sample_idx} samples saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect gesture training data")
    parser.add_argument("--gesture", type=str, required=True, help="Gesture class name (e.g., pinch)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .npy files")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to collect")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration per sample (seconds)")
    collect(parser.parse_args())
