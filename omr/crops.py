"""Fixed-size windows cut around symbols for the CNN classifiers.

Both training (from ground-truth boxes) and inference (from detections) go
through the same function so the networks see identical framing. Windows
are in canonical-scale pixels (UNIT px per staff space) and cut from the
binarised page, ink = 1.
"""
from __future__ import annotations

import cv2
import numpy as np

# (height, width). Noteheads are about 24x28 px at canonical scale; the
# extra margin lets the half/whole classifier see whether a stem is attached.
NOTE_WINDOW = (32, 40)

# Clefs and accidentals are much taller. These are cut at full size then
# shrunk by CLEF_SHRINK / ACC_SHRINK before going into the network.
CLEF_WINDOW = (176, 96)
CLEF_SHRINK = 2
ACC_WINDOW = (80, 40)
ACC_SHRINK = 2


def window(binary: np.ndarray, cx: float, cy: float, size: tuple[int, int],
           shrink: int = 1) -> np.ndarray:
    """Cut a (h, w) window centred on (cx, cy), zero padded at the borders."""
    h, w = size
    y0 = int(round(cy - h / 2))
    x0 = int(round(cx - w / 2))
    out = np.zeros((h, w), dtype=np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(binary.shape[0], y0 + h), min(binary.shape[1], x0 + w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = binary[sy0:sy1, sx0:sx1]
    if shrink > 1:
        out = cv2.resize(out, (w // shrink, h // shrink), interpolation=cv2.INTER_AREA)
    return out


def note_window(binary, cx, cy):
    return window(binary, cx, cy, NOTE_WINDOW)


def clef_window(binary, cx, cy, line_left=None):
    """Window around a clef. Everything left of `line_left` (where the staff
    lines begin) is blanked so braces, brackets and the system barline do
    not end up in the crop; a bracket bar next to a bass clef makes it look
    like a C clef otherwise."""
    win = window(binary, cx, cy, CLEF_WINDOW)
    if line_left is not None:
        cut = int(round(line_left - (cx - CLEF_WINDOW[1] / 2)))
        if cut > 0:
            win[:, :min(cut, win.shape[1])] = 0
    h, w = CLEF_WINDOW
    return cv2.resize(win, (w // CLEF_SHRINK, h // CLEF_SHRINK), interpolation=cv2.INTER_AREA)


def accidental_window(binary, cx, cy):
    return window(binary, cx, cy, ACC_WINDOW, ACC_SHRINK)


def to_batch(windows: list[np.ndarray]) -> np.ndarray:
    """Stack uint8 windows into a float32 (n, h, w, 1) batch in [0, 1]."""
    x = np.stack(windows).astype(np.float32)
    if x.max() > 1:
        x /= 255.0
    return x[..., None]
