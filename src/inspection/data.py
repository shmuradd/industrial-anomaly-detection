"""Read MVTec bottle images without mixing training and test data."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def split_normal(root, seed=42):
    paths = sorted((Path(root) / "train/good").glob("*.png"))
    if len(paths) < 100:
        raise ValueError("Expected the complete MVTec AD bottle training set")
    order = np.random.default_rng(seed).permutation(len(paths))
    count = int(np.ceil(0.2 * len(paths)))
    calibration = [paths[i] for i in order[:count]]
    fitting = [paths[i] for i in order[count:]]
    return fitting, calibration


def test_images(root):
    paths = sorted((Path(root) / "test").glob("*/*.png"))
    if not paths:
        raise ValueError("No test images found")
    return paths


def mask_for(root, path, size=256):
    if path.parent.name == "good":
        return np.zeros((size, size), dtype=np.uint8)
    mask = Path(root) / "ground_truth" / path.parent.name / (path.stem + "_mask.png")
    with Image.open(mask) as image:
        return np.asarray(image.resize((size, size), Image.Resampling.NEAREST)) > 0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
