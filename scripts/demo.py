"""Quick demo launcher to test gesture detection without full UI."""

from __future__ import annotations

import argparse

import cv2
import mediapipe as mp
import numpy as np


def run_demo(args: argparse.Namespace) -> None:
    """Run a simple gesture detection demo with OpenCV window."""
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
    mp_styles = mp.solutions.drawing_styles

    print("Horizon UI Demo — Gesture Detection")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                # Display index finger tip position
                index_tip = hand_landmarks.landmark[8]
                h, w = frame.shape[:2]
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"({index_tip.x:.2f}, {index_tip.y:.2f})",
                           (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Simple gesture heuristic
                gesture = _detect_simple_gesture(hand_landmarks)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            handedness = results.multi_handedness[0].classification[0].label
            cv2.putText(frame, f"Hand: {handedness}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

        cv2.imshow("Horizon UI Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def _detect_simple_gesture(landmarks) -> str:
    """Simple rule-based gesture detection (for demo only)."""
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    extended = []
    for tip, pip_joint in zip(tips, pips):
        extended.append(landmarks.landmark[tip].y < landmarks.landmark[pip_joint].y)

    # All fingers extended
    if all(extended):
        return "open_palm"

    # Only index extended
    if extended[1] and not any(extended[i] for i in [2, 3, 4]):
        return "point"

    # No fingers extended
    if not any(extended):
        return "fist"

    # Thumb and index close together (pinch)
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    if dist < 0.05:
        return "pinch"

    # Thumb up
    if extended[0] and not any(extended[1:]):
        return "thumbs_up"

    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Horizon UI gesture detection demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    run_demo(parser.parse_args())
