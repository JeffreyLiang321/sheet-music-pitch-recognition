"""Notehead detection with a small fully-convolutional network.

The morphological pipeline in segment.py loses hollow heads and merged
chords, which on piano music is most of the page. This network instead
predicts a heatmap that peaks at every notehead centre. It is a three-level
U-Net with a few thousand filters in total, run on the page at half the
canonical resolution (staff lines ~10 px apart).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from .staff import UNIT

DOWN = 2  # the detector works at 1/DOWN of the canonical scale
SIGMA = 2.0  # heatmap peak width in detector pixels
PEAK_THRESHOLD = 0.4
# two heads a second apart are ~0.5 staff space apart vertically
PEAK_RADIUS = int(round(UNIT / DOWN * 0.35))


def build_detector(base: int = 16):
    from tensorflow.keras import layers, Model

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    inp = layers.Input((None, None, 1))
    c1 = block(inp, base)
    c2 = block(layers.MaxPooling2D(2)(c1), base * 2)
    c3 = block(layers.MaxPooling2D(2)(c2), base * 4)
    u2 = layers.Concatenate()([layers.UpSampling2D(2)(c3), c2])
    c4 = block(u2, base * 2)
    u1 = layers.Concatenate()([layers.UpSampling2D(2)(c4), c1])
    c5 = block(u1, base)
    out = layers.Conv2D(1, 1, activation="sigmoid")(c5)
    return Model(inp, out, name="notehead_detector")


def heatmap_target(shape: tuple[int, int], centers: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """Gaussian bump at every (x, y) centre, combined by max."""
    h, w = shape
    target = np.zeros((h, w), dtype=np.float32)
    r = int(3 * sigma)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    bump = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
    for x, y in centers:
        x, y = int(round(x)), int(round(y))
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        patch = bump[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)]
        target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], patch)
    return target


def pad_to_multiple(img: np.ndarray, m: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = (-h) % m, (-w) % m
    if ph == 0 and pw == 0:
        return img
    return np.pad(img, ((0, ph), (0, pw)), mode="constant")


def find_peaks(heat: np.ndarray, threshold: float = PEAK_THRESHOLD,
               radius: int = PEAK_RADIUS) -> list[tuple[float, float, float]]:
    """Local maxima of the heatmap above threshold: [(x, y, score)] in heatmap px."""
    pooled = maximum_filter(heat, size=2 * radius + 1, mode="constant")
    ys, xs = np.nonzero((heat >= threshold) & (heat == pooled))
    peaks = []
    for x, y in zip(xs, ys):
        # sub-pixel refinement from the 3x3 neighbourhood
        y0, y1 = max(0, y - 1), min(heat.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(heat.shape[1], x + 2)
        patch = heat[y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        wsum = patch.sum()
        peaks.append((float((gx * patch).sum() / wsum), float((gy * patch).sum() / wsum), float(heat[y, x])))
    return peaks


def detect(model, binary: np.ndarray, y0: int, y1: int) -> list[tuple[float, float, float]]:
    """Run the detector on rows y0:y1 of a canonical-scale binary page.

    Returns notehead centres (x, y, score) in canonical full-resolution px.
    """
    import cv2

    band = binary[y0:y1]
    small = cv2.resize(band, (band.shape[1] // DOWN, band.shape[0] // DOWN), interpolation=cv2.INTER_AREA)
    small = pad_to_multiple((small > 127).astype(np.float32))
    heat = model.predict(small[None, ..., None], verbose=0)[0, ..., 0]
    heat = heat[:band.shape[0] // DOWN, :band.shape[1] // DOWN]
    return [((x + 0.5) * DOWN, (y + 0.5) * DOWN + y0, s) for x, y, s in find_peaks(heat)]
