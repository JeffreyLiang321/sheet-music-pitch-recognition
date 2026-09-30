"""Staff line detection and scale normalisation.

Staff lines are found from the horizontal projection of the binarised page
(rows that are almost entirely ink), then grouped into staves of five
equally spaced lines. Everything downstream is tuned for the DoReMi
rendering scale, where lines are 21 px apart, so pages are rescaled to that
spacing before the rest of the pipeline runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np
from scipy.signal import find_peaks

UNIT = 21.0  # canonical distance between staff lines, in px

# How far above/below its outer lines a staff "owns" noteheads (ledger
# lines). 4.5 spaces covers up to roughly the 4th ledger line.
STAFF_PAD = 4.5


@dataclass
class Staff:
    line_ys: list[float]
    left: int  # leftmost ink on the line rows (may be a brace or bracket)
    right: int
    line_left: int = 0  # where the staff lines themselves begin
    system: int = 0
    index_in_system: int = 0
    clef: str = "gClef"
    key_fifths: int = 0  # +sharps / -flats, filled in by the key signature step
    clef_prob: float = 0.0
    accidentals: list = field(default_factory=list)

    @property
    def spacing(self) -> float:
        return (self.line_ys[-1] - self.line_ys[0]) / 4

    @property
    def center(self) -> float:
        return self.line_ys[2]

    @property
    def top(self) -> float:
        return self.line_ys[0]

    @property
    def bottom(self) -> float:
        return self.line_ys[-1]

    @property
    def upper_bound(self) -> float:
        return self.top - STAFF_PAD * self.spacing

    @property
    def lower_bound(self) -> float:
        return self.bottom + STAFF_PAD * self.spacing


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold, ink = 255."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _line_rows(binary: np.ndarray, min_frac: float = 0.35) -> np.ndarray:
    """Rows that hold a staff line.

    Only ink that forms a run at least a fifth of the page wide counts, so
    beams, hairpins and text do not show up as peaks; staff lines are the
    only thing on a page that long.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, binary.shape[1] // 5), 1))
    long_runs = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    projection = long_runs.sum(axis=1) / 255.0
    if projection.max() == 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(projection, height=projection.max() * min_frac, distance=4)
    return peaks


def _group_lines(rows: np.ndarray, max_spacing: float = 80.0) -> list[list[float]]:
    """Group line rows into sets of five with (nearly) equal spacing.

    The notebook took every consecutive five rows, which breaks as soon as a
    beam or a title rule shows up as an extra peak. Here each candidate
    staff starts at some row and may skip up to two spurious rows among the
    next six; the first five-row subset whose gaps agree within 20% wins.
    """
    rows = sorted(float(r) for r in rows)
    staffs = []
    i = 0
    while i + 4 < len(rows):
        found = None
        for combo in combinations(range(i + 1, min(i + 7, len(rows))), 4):
            group = [rows[i]] + [rows[j] for j in combo]
            gaps = np.diff(group)
            if gaps.max() <= max_spacing and gaps.max() / gaps.min() < 1.2:
                found = (group, combo[-1])
                break
        if found:
            staffs.append(found[0])
            i = found[1] + 1
        else:
            i += 1
    return staffs


def _extent(binary: np.ndarray, line_ys: list[float]) -> tuple[int, int, int]:
    """(leftmost ink, start of the lines proper, rightmost ink) on the line rows.

    A brace or bracket sits to the left of the lines and shows up on the
    line rows as a short run of ink; the lines themselves are the first run
    that is longer than a couple of staff spaces.
    """
    lefts, starts, rights = [], [], []
    for y in line_ys:
        row = binary[int(round(y))] > 0
        cols = np.flatnonzero(row)
        if not len(cols):
            continue
        lefts.append(cols[0])
        rights.append(cols[-1])
        # run starts: ink pixels whose left neighbour is blank
        run_start = cols[np.r_[True, np.diff(cols) > 1]]
        run_end = cols[np.r_[np.diff(cols) > 1, True]]
        long = run_start[(run_end - run_start) >= 2 * UNIT]
        starts.append(long[0] if len(long) else cols[0])
    if not lefts:
        return 0, 0, binary.shape[1] - 1
    return int(np.median(lefts)), int(np.median(starts)), int(np.median(rights))


def _same_system(binary: np.ndarray, upper: Staff, lower: Staff) -> bool:
    # Staves that belong to one system are joined by a barline drawn down
    # their left edge. Look for a solid vertical run of ink there. On piano
    # music the leftmost ink is the brace, and the barline sits a little to
    # the right of it, so the strip is a couple of staff spaces wide.
    x0 = max(0, min(upper.left, lower.left) - 2)
    x1 = min(binary.shape[1], int(max(upper.left, lower.left) + 2.5 * UNIT))
    y0, y1 = int(upper.bottom), int(lower.top)
    if y1 <= y0 or x1 <= x0:
        return False
    strip = binary[y0:y1, x0:x1] > 0
    return strip.mean(axis=0).max() > 0.9


def detect_staffs(binary: np.ndarray) -> list[Staff]:
    rows = _line_rows(binary)
    staffs = [Staff(line_ys=g, left=0, right=0) for g in _group_lines(rows)]
    for s in staffs:
        s.left, s.line_left, s.right = _extent(binary, s.line_ys)

    system = 0
    for i, s in enumerate(staffs):
        if i > 0 and not _same_system(binary, staffs[i - 1], s):
            system += 1
        s.system = system
        s.index_in_system = sum(1 for t in staffs[:i] if t.system == system)
    return staffs


def normalize_scale(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Rescale so the median staff spacing equals UNIT. Returns (image, scale)."""
    staffs = detect_staffs(binarize(gray))
    if not staffs:
        return gray, 1.0
    spacing = float(np.median([s.spacing for s in staffs]))
    scale = UNIT / spacing
    # line centres are only found to +-1 px, so don't resample for less than
    # about a pixel of spacing difference
    if abs(scale - 1) < 0.06:
        return gray, 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    return resized, scale


def staff_for_y(staffs: list[Staff], y: float) -> Staff:
    """Pick the staff a notehead at height y belongs to."""
    inside = [s for s in staffs if s.upper_bound <= y <= s.lower_bound]
    if len(inside) == 1:
        return inside[0]
    pool = inside or staffs
    return min(pool, key=lambda s: abs(s.center - y))
