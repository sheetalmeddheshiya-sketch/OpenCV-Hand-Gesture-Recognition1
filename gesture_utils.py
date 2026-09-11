"""Shared image preprocessing and feature extraction for gesture recognition."""

from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = (128, 128)
FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert an image into the binary hand silhouette used by the model."""
    if image is None:
        raise ValueError("Could not read image")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def hand_present(image: np.ndarray) -> bool:
    """Reject empty frames and obvious face images before classification."""
    if image is None or image.size == 0:
        return False

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # A dark/flat frame cannot contain the bright hand silhouettes used for
    # training, so never let the classifier label it as a gesture.
    if float(np.mean(gray)) < 20.0 or float(np.std(gray)) < 12.0:
        return False

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        return False

    mask = preprocess_image(gray)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    frame_area = float(mask.shape[0] * mask.shape[1])
    contour_area = cv2.contourArea(largest) / frame_area
    x, y, width, height = cv2.boundingRect(largest)
    extent = (width * height) / frame_area
    return 0.015 <= contour_area <= 0.70 and 0.04 <= extent <= 0.85


def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract normalized HOG features from an image or camera ROI."""
    silhouette = preprocess_image(image)
    hog = cv2.HOGDescriptor(
        _winSize=IMAGE_SIZE,
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9,
    )
    return hog.compute(silhouette).flatten().astype(np.float32)


def iter_image_paths(dataset_dir: Path):
    """Yield (class name, image path) pairs from the dataset directory."""
    for image_path in sorted(dataset_dir.rglob("*.png")):
        # leapGestRecog stores images as <subject>/<gesture>/frame.png.
        yield image_path.parent.name, image_path
