"""Score pipeline output against DoReMi ground truth.

Detections are matched to labelled noteheads greedily: a ground-truth box
claims the nearest unclaimed detection whose centre lies inside it. Every
other metric (class, pitch, staff, timing order) is computed over the
matched pairs, so a note has to be found before it can be judged.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .doremi import GTPage
from .pipeline import PageResult, Note
from .score import assign_timing


@dataclass
class Tally:
    gt: int = 0
    det: int = 0
    tp: int = 0
    cls_ok: int = 0
    filled_ok: int = 0
    hollow_gt: int = 0
    hollow_ok: int = 0
    step_ok: int = 0
    midi_ok: int = 0
    staff_ok: int = 0
    clef_gt: int = 0
    clef_ok: int = 0
    staff_count_ok: int = 0
    pages: int = 0
    order_pairs: int = 0
    order_ok: int = 0
    simul_pairs: int = 0
    simul_ok: int = 0
    confusion: dict = field(default_factory=lambda: defaultdict(int))
    gt_by_class: dict = field(default_factory=lambda: defaultdict(int))
    tp_by_class: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, o: "Tally"):
        for k in vars(self):
            if k in ("confusion", "gt_by_class", "tp_by_class"):
                for kk, v in getattr(o, k).items():
                    getattr(self, k)[kk] += v
            else:
                setattr(self, k, getattr(self, k) + getattr(o, k))

    def summary(self) -> dict:
        p = self.tp / max(self.det, 1)
        r = self.tp / max(self.gt, 1)
        return {
            "pages": self.pages,
            "gt_notes": self.gt,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "class_acc": round(self.cls_ok / max(self.tp, 1), 4),
            "filled_vs_hollow_acc": round(self.filled_ok / max(self.tp, 1), 4),
            "half_vs_whole_acc": round(self.hollow_ok / max(self.hollow_gt, 1), 4),
            "hollow_matched": self.hollow_gt,
            "staff_assign_acc": round(self.staff_ok / max(self.tp, 1), 4),
            "pitch_step_acc": round(self.step_ok / max(self.tp, 1), 4),
            "pitch_midi_acc": round(self.midi_ok / max(self.tp, 1), 4),
            "clef_acc": round(self.clef_ok / max(self.clef_gt, 1), 4),
            "staff_count_acc": round(self.staff_count_ok / max(self.pages, 1), 4),
            "onset_order_acc": round(self.order_ok / max(self.order_pairs, 1), 4),
            "simultaneity_acc": round(self.simul_ok / max(self.simul_pairs, 1), 4),
            "recall_by_class": {c: round(self.tp_by_class[c] / max(n, 1), 4) for c, n in self.gt_by_class.items()},
        }


def match(pred: list[Note], gt: GTPage, scale: float) -> list[tuple[int, int]]:
    """Pairs of (gt index, pred index)."""
    used, pairs = set(), []
    for gi, g in enumerate(gt.notes):
        best, best_d = None, None
        for pi, n in enumerate(pred):
            if pi in used:
                continue
            x, y = n.cx / scale, n.cy / scale
            if g.box.contains(x, y):
                d = (x - g.box.cx) ** 2 + (y - g.box.cy) ** 2
                if best is None or d < best_d:
                    best, best_d = pi, d
        if best is not None:
            used.add(best)
            pairs.append((gi, best))
    return pairs


def score_page(result: PageResult, gt: GTPage) -> Tally:
    t = Tally(pages=1, gt=len(gt.notes), det=len(result.notes))
    assign_timing([result])
    pairs = match(result.notes, gt, result.scale)
    t.tp = len(pairs)
    for g in gt.notes:
        t.gt_by_class[g.cls] += 1
    for gi, _ in pairs:
        t.tp_by_class[gt.notes[gi].cls] += 1

    if len(result.staffs) == len(gt.staffs):
        t.staff_count_ok = 1
    for i, gs in enumerate(gt.staffs):
        clef = gt.clef_for_staff(gs.staff_id)
        if clef is None or i >= len(result.staffs):
            continue
        t.clef_gt += 1
        # alto and tenor share the C clef glyph, which is all DoReMi labels
        t.clef_ok += result.staffs[i].clef.replace("tenorClef", "cClef") == clef.cls

    # staff ids are not in reading order (a quartet page can go 3, 0, 1, 2
    # top to bottom), so compare positions instead
    staff_index = {s.staff_id: i for i, s in enumerate(gt.staffs)}
    matched = []
    for gi, pi in pairs:
        g, n = gt.notes[gi], result.notes[pi]
        matched.append((g, n))
        t.confusion[(g.cls, n.cls)] += 1
        t.cls_ok += g.cls == n.cls
        t.filled_ok += (g.cls == "noteheadBlack") == (n.cls == "noteheadBlack")
        if g.cls != "noteheadBlack":
            t.hollow_gt += 1
            t.hollow_ok += g.cls == n.cls
        t.staff_ok += n.staff == staff_index.get(g.staff_id, -1)
        if n.pitch is not None:
            # DoReMi's step carries the accidental ("Ab"); the letter alone
            # tests staff position and clef, the MIDI number tests the rest
            t.step_ok += (n.pitch.letter, n.pitch.octave) == (g.step[0], g.octave)
            t.midi_ok += n.pitch.midi == g.midi

    # timing: does the left-to-right order agree, and are simultaneous
    # notes still simultaneous? Sample pairs so dense pages stay cheap.
    rng = np.random.default_rng(0)
    idx = list(range(len(matched)))
    if len(idx) > 60:
        idx = list(rng.choice(idx, 60, replace=False))
    for a, b in combinations(idx, 2):
        (ga, na), (gb, nb) = matched[a], matched[b]
        if abs(ga.onset - gb.onset) < 1e-6:
            t.simul_pairs += 1
            t.simul_ok += abs(na.onset - nb.onset) < 1e-6
        elif ga.staff_id == gb.staff_id:
            t.order_pairs += 1
            t.order_ok += (ga.onset < gb.onset) == (na.onset < nb.onset) and na.onset != nb.onset
    return t
