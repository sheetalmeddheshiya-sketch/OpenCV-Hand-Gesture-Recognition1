"""Real-time OpenCV hand gesture recognition application."""

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from gesture_utils import extract_features, hand_present


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/gesture_model.joblib"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model}. Run `python train_model.py` first."
        )

    model = joblib.load(args.model)
    # DirectShow is more reliable than MSMF for many Windows webcams.
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Camera started. Press Q or Esc to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            side = min(height, width) * 2 // 3
            x1, y1 = (width - side) // 2, (height - side) // 2
            x2, y2 = x1 + side, y1 + side
            roi = frame[y1:y2, x1:x2]

            if hand_present(roi):
                features = extract_features(roi).reshape(1, -1)
                prediction = model.predict(features)[0]
                decision = model.decision_function(features)
                confidence = float(np.max(decision))
                label = prediction if confidence >= args.confidence else "unknown"
                status = f"Gesture: {label}  score: {confidence:.2f}"
                status_color = (255, 255, 255)
            else:
                status = "No hand detected - show your hand in the box"
                status_color = (80, 220, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
            cv2.rectangle(frame, (0, 0), (width, 52), (20, 20, 20), -1)
            cv2.putText(
                frame,
                status,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("OpenCV Hand Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()