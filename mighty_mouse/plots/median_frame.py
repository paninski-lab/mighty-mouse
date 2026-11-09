"""Median frame of a session folder, and a frame minus that median (static background removed)."""

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg")


def session_median(folder: Path, n: int, cache: dict) -> tuple[np.ndarray, int]:
    """Per-pixel median of every frame of one session folder, or of n evenly spaced ones if n > 0
    (cached, a few sessions kept)."""
    if folder not in cache:
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        k = len(files) if n <= 0 else min(n, len(files))
        take = np.unique(np.linspace(0, len(files) - 1, k).round().astype(int))
        imgs = [cv2.imread(str(files[t])) for t in take]
        imgs = [im for im in imgs if im is not None]
        imgs = [im for im in imgs if im.shape == imgs[0].shape]
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[folder] = (np.median(np.stack(imgs), 0).astype(np.float32), len(imgs))
    return cache[folder]


def median_subtracted(img: np.ndarray, med: np.ndarray) -> np.ndarray:
    """Frame minus its session median, gray = unchanged; the frame's global brightness offset is
    removed and the difference stretched per frame (99.5th percentile -> full range, at most x16),
    so dim frames stay readable."""
    if med.shape != img.shape:
        med = cv2.resize(med, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    d = img.astype(np.float32).mean(2) - med.mean(2)
    d -= np.median(d)
    gain = 127.0 / max(float(np.percentile(np.abs(d), 99.5)), 8.0)
    return cv2.cvtColor(np.clip(128 + d * gain, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
