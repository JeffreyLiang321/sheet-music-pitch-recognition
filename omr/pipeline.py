"""Run the whole thing on one page image.

    result = Pipeline().run(gray_page)

The page is rescaled to the canonical staff spacing, staves are found, then
noteheads are detected (CNN heatmap, or the morphological pipeline as a
baseline), classified, assigned to a staff and given a pitch from the
staff's clef and key signature. Timing is worked out afterwards in score.py.
All coordinates in the result are in the canonical-scale image; divide by
`scale` to get back to the input image.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import detector as det
from . import segment
from .crops import note_window, clef_window, accidental_window, to_batch
from .models import load_model, MODEL_DIR, NOTE_CLASSES, CLEF_CLASSES, ACC_CLASSES
from .pitch import Pitch, pitch_at, key_from_accidentals, staff_position
from .staff import UNIT, Staff, binarize, detect_staffs, normalize_scale, staff_for_y

DURATION_BEATS = {"noteheadBlack": 1.0, "noteheadHalf": 2.0, "noteheadWhole": 4.0}


@dataclass
class Note:
    cx: float
    cy: float
    score: float
    cls: str = "noteheadBlack"
    p_hollow: float = 0.0
    p_whole: float = 0.0
    staff: int = 0  # index into PageResult.staffs
    position: int = 0  # half-spaces above the middle line
    pitch: Pitch | None = None
    accidental: int | None = None
    onset: float = 0.0
    duration: float = 1.0
    column: int = 0

    @property
    def midi(self) -> int:
        return self.pitch.midi if self.pitch else 60

    @property
    def box(self) -> tuple[int, int, int, int]:
        w, h = 0.65 * UNIT, 0.55 * UNIT
        return int(self.cx - w), int(self.cy - h), int(self.cx + w), int(self.cy + h)


@dataclass
class PageResult:
    width: int
    height: int
    scale: float
    staffs: list[Staff] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    barlines: dict[int, list[float]] = field(default_factory=dict)  # staff index -> x positions

    @property
    def systems(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for i, s in enumerate(self.staffs):
            out.setdefault(s.system, []).append(i)
        return [out[k] for k in sorted(out)]


def _components(binary: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """Connected components inside a window, as (bx0, by0, bx1, by1) in page px."""
    region = binary[max(0, y0):y1, max(0, x0):x1]
    if region.size == 0:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        out.append((x + max(0, x0), y + max(0, y0), x + w + max(0, x0), y + h + max(0, y0)))
    return out


def _merge_boxes(boxes, gap_x: float, gap_y: float):
    """Union boxes that (nearly) touch.

    Removing the staff lines cuts a glyph wherever one of its curves runs
    along a line, so a flat can come back as a stem plus a bowl and a bass
    clef as two halves. Gluing pieces that are within gap_x / gap_y of each
    other gets the whole symbol back.
    """
    boxes = sorted(boxes)
    merged = []
    for b in boxes:
        for i, m in enumerate(merged):
            if b[0] <= m[2] + gap_x and b[2] >= m[0] - gap_x and b[1] <= m[3] + gap_y and b[3] >= m[1] - gap_y:
                merged[i] = (min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3]))
                break
        else:
            merged.append(b)
    return merged


class Pipeline:
    def __init__(self, model_dir=MODEL_DIR, use_detector: bool = True, legacy: bool = False):
        """legacy=True runs the notebook's segmentation exactly as it was,
        without the two fixes in segment.py; only meaningful with
        use_detector=False."""
        self.use_detector = use_detector
        self.legacy = legacy
        self.detector = load_model("detector", model_dir) if use_detector else None
        self.filled = load_model("note_filled", model_dir)
        self.hollow = load_model("note_hollow", model_dir)
        self.clef = load_model("clef", model_dir)
        self.accidental = load_model("accidental", model_dir)

    # -- noteheads ---------------------------------------------------------

    def _detect(self, binary: np.ndarray, staffs: list[Staff]) -> list[Note]:
        if self.use_detector:
            y0 = int(max(0, min(s.top for s in staffs) - 6 * UNIT))
            y1 = int(min(binary.shape[0], max(s.bottom for s in staffs) + 6 * UNIT))
            y1 -= (y1 - y0) % det.DOWN
            peaks = det.detect(self.detector, binary, y0, y1)
            return [Note(x, y, s) for x, y, s in peaks]
        if self.legacy:
            accepted, _ = segment.find_noteheads(binary, None, legacy=True)
        else:
            line_ys = [y for s in staffs for y in s.line_ys]
            accepted, _ = segment.find_noteheads(binary, line_ys)
        return [Note(b.cx, b.cy, 1.0) for b in accepted]

    def _classify(self, binary: np.ndarray, notes: list[Note]) -> None:
        if not notes:
            return
        x = to_batch([note_window(binary, n.cx, n.cy) for n in notes])
        p_hollow = self.filled.predict(x, verbose=0)[:, 0]
        p_whole = self.hollow.predict(x, verbose=0)[:, 0]
        for n, ph, pw in zip(notes, p_hollow, p_whole):
            n.p_hollow, n.p_whole = float(ph), float(pw)
            n.cls = "noteheadBlack" if ph < 0.5 else ("noteheadWhole" if pw > 0.5 else "noteheadHalf")
            n.duration = DURATION_BEATS[n.cls]

    # -- clefs, key signatures, barlines ------------------------------------

    def _find_clef(self, clean: np.ndarray, staff: Staff):
        """Left-most tall glyph at the start of the staff.

        Falls back to a window a couple of spaces in from the staff's left
        edge, which is where a clef sits, if no component looks like one.
        """
        # Only glyphs that start where the staff lines start can be a clef;
        # braces and brackets begin further left. Components are taken over
        # the wider region so those get excluded rather than clipped.
        x0, x1 = int(staff.line_left - 0.3 * UNIT), int(staff.line_left + 6 * UNIT)
        y0, y1 = int(staff.top - 3 * UNIT), int(staff.bottom + 3 * UNIT)
        boxes = [b for b in _components(clean, min(x0, int(staff.left) - 2), y0, x1, y1)
                 if b[3] - b[1] > 0.5 * UNIT and b[0] >= x0]
        # the barline at the left edge is tall and thin; keep it out of the
        # merge so it doesn't swallow the clef
        boxes = [b for b in boxes if b[2] - b[0] > 0.3 * UNIT and b[3] - b[1] < 9 * UNIT]
        cands = []
        # a tight gap: enough to re-join a glyph the line removal cut in two,
        # not enough to pull in the key signature that follows
        for bx0, by0, bx1, by1 in _merge_boxes(boxes, 0.12 * UNIT, 0.3 * UNIT):
            h, w = by1 - by0, bx1 - bx0
            if 2.5 * UNIT <= h <= 9 * UNIT and w >= 1.2 * UNIT:
                # no clef is wider than ~3 spaces; clip anything glued on
                cands.append((bx0, by0, min(bx1, int(bx0 + 3.2 * UNIT)), by1))
        if cands:
            return min(cands, key=lambda c: c[0])
        cx = staff.line_left + 2 * UNIT
        return (int(cx - 1.5 * UNIT), int(staff.top - UNIT), int(cx + 1.5 * UNIT), int(staff.bottom + UNIT))

    def _key_signature(self, clean: np.ndarray, staff: Staff, clef_box, first_note_x: float) -> list[str]:
        x0 = clef_box[2] - 2
        x1 = int(min(first_note_x - 0.8 * UNIT, clef_box[2] + 9 * UNIT))
        y0, y1 = int(staff.top - 2 * UNIT), int(staff.bottom + 2 * UNIT)
        boxes = [b for b in _components(clean, x0, y0, x1, y1) if b[3] - b[1] > 0.3 * UNIT]
        cands = []
        for bx0, by0, bx1, by1 in _merge_boxes(boxes, 0.2 * UNIT, 0.2 * UNIT):
            h, w = by1 - by0, bx1 - bx0
            if 1.7 * UNIT <= h <= 3.6 * UNIT and 0.4 * UNIT <= w <= 1.6 * UNIT:
                cands.append((bx0, by0, bx1, by1))
        cands.sort()
        if not cands:
            return []
        x = to_batch([accidental_window(clean, (c[0] + c[2]) / 2, (c[1] + c[3]) / 2) for c in cands])
        probs = self.accidental.predict(x, verbose=0)
        kinds = []
        for c, p in zip(cands, probs):
            kind = ACC_CLASSES[int(p.argmax())]
            if kind == "other":
                break
            kinds.append(kind)
        return kinds

    def _barlines(self, clean: np.ndarray, staff: Staff) -> list[float]:
        """x positions of barlines: columns of solid ink across the staff.

        Done per column rather than per component because slurs and ties
        often cross a barline and would fuse with it. A stem can also run
        the full staff height, so a column is dropped if there is a notehead
        sized blob at either end of its run.
        """
        top, bottom = int(round(staff.top)), int(round(staff.bottom))
        x0, x1 = int(staff.left + 2 * UNIT), int(min(clean.shape[1], staff.right + UNIT))
        band = clean[top - 2:bottom + 3, x0:x1] > 0
        solid = band.mean(axis=0) >= 0.9
        xs = np.flatnonzero(solid)
        if len(xs) == 0:
            return []

        def blob_near(x, y):
            # ink density in a notehead-sized box just left/right of (x, y)
            h, w = int(0.6 * UNIT), int(1.4 * UNIT)
            best = 0.0
            for bx in (x - w - 2, x + 3):
                patch = clean[max(0, y - h // 2):y + h // 2, max(0, bx):bx + w]
                if patch.size:
                    best = max(best, float((patch > 0).mean()))
            return best > 0.35

        bars, run = [], [xs[0]]
        for x in list(xs[1:]) + [None]:
            if x is not None and x == run[-1] + 1:
                run.append(x)
                continue
            cx = x0 + int(np.mean(run))
            if len(run) <= 0.4 * UNIT and not blob_near(cx, top) and not blob_near(cx, bottom):
                bars.append(float(cx))
            run = [x]
        return bars

    def _local_accidental(self, clean: np.ndarray, note: Note) -> int | None:
        """Sharp/flat/natural drawn right in front of the notehead, if any."""
        x0, x1 = int(note.cx - 2.4 * UNIT), int(note.cx - 0.6 * UNIT)
        y0, y1 = int(note.cy - 2 * UNIT), int(note.cy + 2 * UNIT)
        best = None
        for bx0, by0, bx1, by1 in _components(clean, x0, y0, x1, y1):
            h, w = by1 - by0, bx1 - bx0
            cy = (by0 + by1) / 2
            # an accidental is centred on its note's line or space; the
            # neighbouring note of a chord is only half a space away, so
            # the vertical tolerance has to be tight
            if 1.7 * UNIT <= h <= 3.6 * UNIT and 0.4 * UNIT <= w <= 1.6 * UNIT and abs(cy - note.cy) < 0.45 * UNIT:
                if best is None or bx1 > best[2]:
                    best = (bx0, by0, bx1, by1)
        if best is None:
            return None
        p = self.accidental.predict(to_batch([accidental_window(clean, (best[0] + best[2]) / 2, (best[1] + best[3]) / 2)]), verbose=0)[0]
        kind = ACC_CLASSES[int(p.argmax())]
        return {"accidentalSharp": 1, "accidentalFlat": -1, "accidentalNatural": 0}.get(kind)

    # -- main ---------------------------------------------------------------

    def run(self, gray: np.ndarray) -> PageResult:
        height, width = gray.shape[:2]
        gray, scale = normalize_scale(gray)
        binary = binarize(gray)
        staffs = detect_staffs(binary)
        result = PageResult(width=width, height=height, scale=scale, staffs=staffs)
        if not staffs:
            return result

        notes = self._detect(binary, staffs)
        # drop anything outside the staves' horizontal extent
        left = min(s.left for s in staffs) - 2 * UNIT
        right = max(s.right for s in staffs) + 2 * UNIT
        notes = [n for n in notes if left <= n.cx <= right]
        self._classify(binary, notes)

        for n in notes:
            staff = staff_for_y(staffs, n.cy)
            n.staff = staffs.index(staff)
            n.position = staff_position(n.cy, staff.center, staff.spacing)

        clean = segment.remove_staff_lines_at(binary, [y for s in staffs for y in s.line_ys])
        for i, staff in enumerate(staffs):
            clef_box = self._find_clef(clean, staff)
            if clef_box is not None:
                x = to_batch([clef_window(clean, (clef_box[0] + clef_box[2]) / 2, (clef_box[1] + clef_box[3]) / 2,
                                          line_left=staff.line_left)])
                p = self.clef.predict(x, verbose=0)[0]
                staff.clef = CLEF_CLASSES[int(p.argmax())]
                staff.clef_prob = float(p.max())
                # the C clef glyph is centred on the line it names: middle
                # line for alto, fourth line for tenor
                if staff.clef == "cClef" and (clef_box[1] + clef_box[3]) / 2 < staff.center - 0.5 * UNIT:
                    staff.clef = "tenorClef"
                on_staff = [n.cx for n in notes if n.staff == i]
                first = min(on_staff) if on_staff else staff.right
                staff.accidentals = self._key_signature(clean, staff, clef_box, first)
                staff.key_fifths = key_from_accidentals(staff.accidentals)
            result.barlines[i] = self._barlines(clean, staff)

        # Accidentals carry through to the end of the measure for the same
        # letter and octave, so walk each staff left to right resetting at
        # every barline.
        for i, staff in enumerate(staffs):
            on_staff = sorted((n for n in notes if n.staff == i), key=lambda n: n.cx)
            bars = result.barlines[i]
            active: dict[tuple[str, int], int] = {}
            next_bar = 0
            for n in on_staff:
                while next_bar < len(bars) and bars[next_bar] < n.cx:
                    active.clear()
                    next_bar += 1
                base = pitch_at(n.cy, staff.center, staff.spacing, staff.clef, staff.key_fifths)
                key = (base.letter, base.octave)
                local = self._local_accidental(clean, n)
                if local is not None:
                    active[key] = local
                n.accidental = active.get(key)
                n.pitch = pitch_at(n.cy, staff.center, staff.spacing, staff.clef, staff.key_fifths, n.accidental)

        result.notes = sorted(notes, key=lambda n: (staffs[n.staff].system, n.cx, n.cy))
        return result
