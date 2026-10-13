"""Fit normal feature statistics and score unusual regions."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

SIZE = 256


class Encoder(torch.nn.Module):
    """Take aligned features from two frozen ResNet18 stages."""

    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.eval().requires_grad_(False)

    def forward(self, images):
        first = self.layer1(self.stem(images))
        second = self.layer2(first)
        first = F.adaptive_avg_pool2d(first, second.shape[-2:])
        return torch.cat([first, second], dim=1)


def image_tensor(path):
    transform = Compose(
        [
            Resize((SIZE, SIZE)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


@torch.inference_mode()
def extract(encoder, paths, batch_size=16):
    if not paths:
        raise ValueError("No images supplied")
    batches = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([image_tensor(p) for p in paths[start : start + batch_size]])
        batches.append(encoder(batch).numpy())
    return np.concatenate(batches)


def fit_statistics(features, regularization=0.01):
    """Keep a mean and diagonal variance at each feature position."""
    if features.ndim != 4 or len(features) < 2 or regularization <= 0:
        raise ValueError("Need at least two feature maps and positive regularization")
    mean = features.mean(axis=0)
    variance = features.var(axis=0, ddof=1)
    # Keep weak channels from dominating the distance
    floor = regularization * np.maximum(variance.mean(axis=(1, 2), keepdims=True), 1e-6)
    return mean.astype(np.float32), (variance + floor).astype(np.float32)


def score_maps(features, mean, variance):
    if features.shape[1:] != mean.shape or mean.shape != variance.shape:
        raise ValueError("Feature and model shapes do not match")
    if np.any(variance <= 0):
        raise ValueError("Variances must be positive")
    distance = np.sqrt(np.mean((features - mean) ** 2 / variance, axis=1))
    resized = F.interpolate(
        torch.from_numpy(distance[:, None]), size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[:, 0].numpy()
    return np.stack([gaussian_filter(item, sigma=4) for item in resized])


def image_scores(maps):
    # Average the strongest one percent to reduce single-pixel noise
    flat = maps.reshape(len(maps), -1)
    count = max(1, math.ceil(flat.shape[1] * 0.01))
    return np.partition(flat, -count, axis=1)[:, -count:].mean(axis=1)


def calibrate(scores, alpha=0.05):
    """Choose a finite-sample upper quantile using held-out normal images."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration requires finite scores")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between zero and one")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("Not enough calibration images for this alpha")
    return float(np.sort(scores)[rank - 1])


def load_statistics(path):
    with np.load(path, allow_pickle=False) as saved:
        return saved["mean"], saved["variance"], float(saved["threshold"])
