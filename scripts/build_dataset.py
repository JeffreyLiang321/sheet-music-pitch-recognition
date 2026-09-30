"""Cut training data out of DoReMi.

Writes to data/cache:
  noteheads_{split}.npz    windows around every notehead, labelled black/half/whole
  clefs_{split}.npz        windows around every clef
  accidentals_{split}.npz  windows around accidentals, plus time signature
                           digits and rests as the "other" class
  bands/                   half-resolution binary strip of each page's system
                           and the notehead centres in it, for the detector

Run with --limit N to only process N pages per song while iterating.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from omr import doremi
from omr.crops import note_window, clef_window, accidental_window
from omr.detector import DOWN
from omr.models import NOTE_CLASSES, CLEF_CLASSES, ACC_CLASSES
from omr.staff import UNIT, binarize, normalize_scale

OUT = doremi.CACHE_DIR
BAND_DIR = OUT / "bands"
BAND_PAD = 6 * UNIT

# the three big synthetic "beam groups 12 ..." songs are ~4000 near identical
# pages; one in ten is plenty
SYNTHETIC_STRIDE = 10
MAX_BLACK_PER_PAGE = 40
MAX_ACC_PER_PAGE = 12
MAX_OTHER_PER_PAGE = 8
JITTER = 3  # px, training crops only


def jitter(rng, split):
    if split != "train":
        return 0.0, 0.0
    return rng.uniform(-JITTER, JITTER), rng.uniform(-JITTER, JITTER)


def band_bounds(page, shape):
    top = min(s.line_ys[0] for s in page.staffs) - BAND_PAD
    bottom = max(s.line_ys[-1] for s in page.staffs) + BAND_PAD
    left = min(s.left for s in page.staffs) - 3 * UNIT
    right = max(s.right for s in page.staffs) + 3 * UNIT
    y0, y1 = int(max(0, top)), int(min(shape[0], bottom))
    x0, x1 = int(max(0, left)), int(min(shape[1], right))
    # keep dimensions even so the half-res band maps back cleanly
    return y0, y1 - (y1 - y0) % DOWN, x0, x1 - (x1 - x0) % DOWN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(0)
    BAND_DIR.mkdir(parents=True, exist_ok=True)
    notes = defaultdict(lambda: {"x": [], "y": [], "song": []})
    clefs = defaultdict(lambda: {"x": [], "y": [], "song": []})
    accs = defaultdict(lambda: {"x": [], "y": [], "song": []})
    band_index = []
    skipped = 0

    for song in doremi.songs():
        split = doremi.split_of(song)
        pages = doremi.load_song(song)
        if song.startswith("beam groups 12"):
            pages = pages[::SYNTHETIC_STRIDE]
        if args.limit:
            pages = pages[:args.limit]

        for page in tqdm(pages, desc=song[:30].ljust(30), leave=False):
            if not page.staffs or not page.notes:
                continue
            gray = cv2.imread(str(page.image_path), cv2.IMREAD_GRAYSCALE)
            gray, scale = normalize_scale(gray)
            if scale != 1.0:
                # DoReMi is rendered at exactly the canonical spacing, so
                # anything else means staff detection went wrong on the page
                skipped += 1
                continue
            binary = binarize(gray)

            black = [n for n in page.notes if n.cls == "noteheadBlack"]
            rng.shuffle(black)
            for n in black[:MAX_BLACK_PER_PAGE] + [n for n in page.notes if n.cls != "noteheadBlack"]:
                dx, dy = jitter(rng, split)
                notes[split]["x"].append(note_window(binary, n.box.cx + dx, n.box.cy + dy))
                notes[split]["y"].append(NOTE_CLASSES.index(n.cls))
                notes[split]["song"].append(song)

            staff_left = {s.staff_id: s.left for s in page.staffs}
            for c in page.clefs:
                # clef boxes come from the detector's own rough localisation
                # at inference, so jitter these twice as much as noteheads
                dx, dy = (2 * v for v in jitter(rng, split))
                clefs[split]["x"].append(clef_window(binary, c.box.cx + dx, c.box.cy + dy,
                                                     line_left=staff_left.get(c.staff_id)))
                clefs[split]["y"].append(CLEF_CLASSES.index(c.cls))
                clefs[split]["song"].append(song)

            acc = list(page.accidentals)
            rng.shuffle(acc)
            other = list(page.others)
            rng.shuffle(other)
            for a in acc[:MAX_ACC_PER_PAGE] + other[:MAX_OTHER_PER_PAGE]:
                dx, dy = jitter(rng, split)
                accs[split]["x"].append(accidental_window(binary, a.box.cx + dx, a.box.cy + dy))
                accs[split]["y"].append(ACC_CLASSES.index(a.cls) if a.cls in ACC_CLASSES else ACC_CLASSES.index("other"))
                accs[split]["song"].append(song)

            y0, y1, x0, x1 = band_bounds(page, binary.shape)
            band = binary[y0:y1, x0:x1]
            small = cv2.resize(band, (band.shape[1] // DOWN, band.shape[0] // DOWN), interpolation=cv2.INTER_AREA)
            small = ((small > 127) * 255).astype(np.uint8)
            name = f"{song}-{page.page_index:03d}"
            cv2.imwrite(str(BAND_DIR / f"{name}.png"), small)
            band_index.append({
                "name": name, "song": song, "split": split, "offset": [x0, y0],
                "centers": [[(n.box.cx - x0) / DOWN, (n.box.cy - y0) / DOWN, NOTE_CLASSES.index(n.cls)]
                            for n in page.notes],
            })

    for label, store in (("noteheads", notes), ("clefs", clefs), ("accidentals", accs)):
        for split, d in store.items():
            np.savez_compressed(OUT / f"{label}_{split}.npz", x=np.stack(d["x"]), y=np.array(d["y"], dtype=np.int8),
                                song=np.array(d["song"]))
            counts = np.bincount(np.array(d["y"]))
            print(f"{label:12s} {split:5s} n={len(d['y']):6d} per class={counts.tolist()}")
    with open(OUT / "bands.json", "w") as f:
        json.dump(band_index, f)
    print(f"bands: {len(band_index)} pages ({skipped} skipped)")


if __name__ == "__main__":
    main()
