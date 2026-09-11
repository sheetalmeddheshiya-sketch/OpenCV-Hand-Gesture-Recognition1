"""Train a HOG + SVM hand gesture classifier from leapGestRecog."""

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from gesture_utils import extract_features, iter_image_paths


def load_dataset(dataset_dir: Path, max_per_class: int | None):
    features = []
    labels = []
    counts = {}

    for class_name, image_path in iter_image_paths(dataset_dir):
        if max_per_class is not None and counts.get(class_name, 0) >= max_per_class:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        features.append(extract_features(image))
        labels.append(class_name)
        counts[class_name] = counts.get(class_name, 0) + 1

    if not features:
        raise RuntimeError(f"No PNG images found in {dataset_dir}")
    return np.asarray(features), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/leapGestRecog"))
    parser.add_argument("--output", type=Path, default=Path("models/gesture_model.joblib"))
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Maximum images per class (use 0 for the complete dataset).",
    )
    args = parser.parse_args()

    max_per_class = None if args.max_per_class == 0 else args.max_per_class
    print(f"Loading images from {args.dataset}...")
    x, y = load_dataset(args.dataset, max_per_class)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=2.0, class_weight="balanced", max_iter=5000)),
        ]
    )
    print(f"Training on {len(x_train)} images and validating on {len(x_test)} images...")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, zero_division=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    metadata = {
        "classes": sorted(set(y.tolist())),
        "validation_accuracy": round(float(accuracy), 4),
        "samples": int(len(x)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
