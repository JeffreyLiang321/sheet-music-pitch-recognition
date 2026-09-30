"""Run the pipeline on one image or PDF from the command line.

    python scripts/run_page.py score.png --midi out.mid --overlay out.png

Prints one line per staff (clef, key) and a summary of the notes found.
"""
import argparse

import cv2
import numpy as np

from omr.pipeline import Pipeline
from omr.score import assign_timing, write_midi, write_musicxml

COLOR = {"noteheadBlack": (0, 0, 220), "noteheadHalf": (0, 160, 60), "noteheadWhole": (220, 100, 0)}


def load_pages(path: str) -> list[np.ndarray]:
    if path.lower().endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(path)
        out = []
        for page in doc:
            pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
            out.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width))
        return out
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE)]


def draw_overlay(gray, result, path):
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    s = result.scale
    for i, st in enumerate(result.staffs):
        cv2.putText(canvas, f"{st.clef} key={st.key_fifths:+d}", (int(st.left / s), int(st.top / s) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
        for x in result.barlines.get(i, []):
            cv2.line(canvas, (int(x / s), int(st.top / s)), (int(x / s), int(st.bottom / s)), (180, 180, 0), 2)
    for n in result.notes:
        x0, y0, x1, y1 = [int(v / s) for v in n.box]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR[n.cls], 2)
        cv2.putText(canvas, n.pitch.name if n.pitch else "?", (x0, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 0, 120), 1)
    cv2.imwrite(path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--midi")
    ap.add_argument("--musicxml")
    ap.add_argument("--overlay", help="write an annotated PNG (first page only)")
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--bpm", type=int, default=90)
    args = ap.parse_args()

    pipe = Pipeline(use_detector=not args.classical)
    pages = load_pages(args.path)
    results = [pipe.run(g) for g in pages]
    events = assign_timing(results)

    for p, r in enumerate(results):
        for i, st in enumerate(r.staffs):
            n = sum(1 for x in r.notes if x.staff == i)
            print(f"page {p} staff {i} (system {st.system}): {st.clef} p={st.clef_prob:.2f} "
                  f"key={st.key_fifths:+d} {st.accidentals} notes={n} barlines={len(r.barlines.get(i, []))}")
    print(f"{len(events)} notes, {events[-1].onset + events[-1].duration if events else 0:.1f} beats")

    if args.overlay:
        draw_overlay(pages[0], results[0], args.overlay)
    if args.midi:
        write_midi(events, args.midi, args.bpm)
    if args.musicxml:
        write_musicxml(events, args.musicxml, args.bpm)


if __name__ == "__main__":
    main()
