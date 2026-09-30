"""Notehead segmentation.

Ported from the notebook more or less as-is: remove staff lines, close the
gaps that leaves in noteheads, fill hollow heads, open away stems and thin
artifacts, then find contours. Anything too big to be one notehead is split
with a watershed, and each candidate has an ellipse fitted to it and is
rejected if the fit does not look like a notehead.

All sizes assume the page has been rescaled so staff lines are UNIT px apart
(see staff.normalize_scale).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ELLIPSE_SIZE = 20.5  # roughly one staff space; kernels are sized off this
WIDTH_FACTOR = 0.5
HEIGHT_FACTOR = 0.4
MIN_AREA = 100  # px, anything smaller is a speck or a fragment
MAX_AREA = 600  # a single filled head is ~500 px; bigger means merged heads
MAX_HEIGHT_WIDTH_RATIO = 2.0
MIN_WIDTH_HEIGHT_RATIO = 1.0
PAD = 4


@dataclass
class Blob:
    cx: float
    cy: float
    w: float  # ellipse axes as returned by fitEllipse
    h: float
    angle: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (int(round(self.cx - self.w / 2)), int(round(self.cy - self.h / 2)),
                int(round(self.cx + self.w / 2)), int(round(self.cy + self.h / 2)))


def remove_staff_lines(binary: np.ndarray) -> np.ndarray:
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal)
    return cv2.subtract(binary, lines)


def remove_staff_lines_at(binary: np.ndarray, line_ys: list[float], thickness: int = 3) -> np.ndarray:
    """Erase staff lines only where the ink is no taller than the line.

    Uses the detected line positions instead of a morphological opening, so
    a notehead sitting on a line keeps its full outline. A pixel in the line
    band is kept if the vertical run of ink through it extends past the band.
    """
    out = binary.copy()
    h = binary.shape[0]
    reach = thickness + 1
    for y in line_ys:
        y = int(round(y))
        y0, y1 = max(0, y - reach), min(h, y + reach + 1)
        band = binary[y0:y1] > 0
        # a column survives if there is ink just above or just below the band
        above = binary[max(0, y0 - 2):y0].any(axis=0) if y0 > 0 else np.zeros(binary.shape[1], bool)
        below = binary[y1:min(h, y1 + 2)].any(axis=0) if y1 < h else np.zeros(binary.shape[1], bool)
        keep = above | below
        band_out = np.where(keep[None, :], band, False)
        out[y0:y1] = band_out.astype(np.uint8) * 255
    return out


def selective_hole_fill(binary: np.ndarray, min_hole_area: int, max_hole_area: int) -> np.ndarray:
    """Fill enclosed holes in the size range of a hollow notehead's centre."""
    inv = cv2.bitwise_not(binary)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = binary.copy()
    for lbl in range(1, num):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if min_hole_area <= area <= max_hole_area:
            filled[labels == lbl] = 255
    return filled


def notehead_mask(binary: np.ndarray, line_ys: list[float] | None = None,
                  legacy: bool = False) -> np.ndarray:
    """Binary mask of candidate noteheads (staff lines out, heads solid).

    legacy=True reproduces the notebook exactly: morphological line removal
    and merged blobs dropped before they could be split.
    """
    if line_ys is None or legacy:
        processed = remove_staff_lines(binary)
    else:
        processed = remove_staff_lines_at(binary, line_ys)

    close_krn = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, round(ELLIPSE_SIZE * 0.2)), max(1, round(ELLIPSE_SIZE * 0.4))))
    closed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, close_krn)
    filled = selective_hole_fill(closed, min_hole_area=150, max_hole_area=250)

    open_krn = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (round(ELLIPSE_SIZE * WIDTH_FACTOR), round(ELLIPSE_SIZE * HEIGHT_FACTOR)))
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, open_krn)

    # tall thin kernel merges vertically adjacent fragments (time signatures
    # etc.) so they get thrown out together in the size filter below
    close_krn = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, round(ELLIPSE_SIZE * 0.1)), max(1, round(ELLIPSE_SIZE * 0.8))))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_krn)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Blobs bigger than one head are kept here so the watershed in
        # fit_blobs gets a chance to split them (chords). The notebook
        # dropped them at this point, which meant the splitting never ran.
        if area > MAX_AREA and area <= 4 * MAX_AREA and not legacy:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            continue
        if area > MAX_AREA:
            continue
        if h / float(w) > MAX_HEIGHT_WIDTH_RATIO or w / float(h) < MIN_WIDTH_HEIGHT_RATIO:
            continue
        cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask


def _roi(mask: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(mask)
    return (max(0, x - PAD), max(0, y - PAD),
            min(mask.shape[1], x + w + PAD), min(mask.shape[0], y + h + PAD))


def split_horizontal_blob(mask: np.ndarray) -> list[np.ndarray]:
    """Cut a tall blob (two stacked heads) at its vertical midpoint."""
    x, y, w, h = cv2.boundingRect(mask)
    mid = y + h // 2
    top, bottom = np.zeros_like(mask), np.zeros_like(mask)
    top[y:mid, x:x + w] = mask[y:mid, x:x + w]
    bottom[mid:y + h, x:x + w] = mask[mid:y + h, x:x + w]
    if cv2.countNonZero(top) == 0 or cv2.countNonZero(bottom) == 0:
        return [mask]
    # open each half to round off the straight cut edge
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return [cv2.morphologyEx(p, cv2.MORPH_OPEN, kern) for p in (top, bottom)]


def split_blob(mask: np.ndarray) -> list[np.ndarray]:
    """Watershed a merged blob into pieces of at most one notehead each."""
    if cv2.countNonZero(mask) <= MAX_AREA:
        return [mask]

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
    fg = fg.astype(np.uint8)

    num, markers = cv2.connectedComponents(fg)
    if num < 2:
        return [mask]
    markers = markers + 1
    markers[cv2.subtract(mask, fg) == 255] = 0
    cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)

    pieces = []
    for lbl in range(2, num + 1):
        piece = np.zeros_like(mask)
        piece[markers == lbl] = 255
        if cv2.countNonZero(piece) == 0:
            continue
        x, y, w, h = cv2.boundingRect(piece)
        # still taller than wide means two heads a second apart, on top of
        # each other, which the watershed can't separate
        if h > w * 1.2:
            pieces.extend(split_horizontal_blob(piece))
        else:
            pieces.append(piece)
    return pieces or [mask]


def fit_blobs(mask: np.ndarray,
              min_area: int = MIN_AREA,
              min_ellipse_area: int = 300,
              min_minor_axis: int = 10,
              min_axis_ratio: float = 0.5,
              min_area_ratio: float = 0.6,
              max_area_ratio: float = 1.3) -> tuple[list[Blob], list[Blob]]:
    """Fit an ellipse to every blob in the mask. Returns (accepted, rejected)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted, rejected = [], []
    for cnt in contours:
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [cnt], -1, 255, -1)
        pieces = split_blob(blob) if cv2.contourArea(cnt) > MAX_AREA else [blob]

        for piece in pieces:
            x0, y0, x1, y1 = _roi(piece)
            local, _ = cv2.findContours(piece[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for lc in local:
                if len(lc) < 5:
                    continue
                area = cv2.contourArea(lc)
                (cx, cy), (ma, mb), ang = cv2.fitEllipse(lc)
                b = Blob(cx + x0, cy + y0, ma, mb, ang)
                ell_area = np.pi * (ma / 2) * (mb / 2)
                ok = (min_area <= area <= MAX_AREA
                      and ell_area >= min_ellipse_area
                      and min(ma, mb) >= min_minor_axis
                      and min(ma, mb) / max(ma, mb) >= min_axis_ratio
                      and min_area_ratio <= area / ell_area <= max_area_ratio)
                (accepted if ok else rejected).append(b)
    return accepted, rejected


def find_noteheads(binary: np.ndarray, line_ys: list[float] | None = None,
                   legacy: bool = False) -> tuple[list[Blob], list[Blob]]:
    mask = notehead_mask(binary, line_ys, legacy)
    return fit_blobs(mask)
