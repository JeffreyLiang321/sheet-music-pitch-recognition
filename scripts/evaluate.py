"""Evaluate the pipeline on held-out DoReMi songs.

    python scripts/evaluate.py --split test
    python scripts/evaluate.py --split test --classical     # morphological baseline
    python scripts/evaluate.py --split val --limit 5        # quick check

Prints a per-song table and writes results/<split>_<detector>.json.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from omr import doremi
from omr.evaluate import Tally, score_page
from omr.pipeline import Pipeline

COLS = ["pages", "gt_notes", "precision", "recall", "f1", "class_acc", "half_vs_whole_acc",
        "staff_assign_acc", "clef_acc", "pitch_step_acc", "pitch_midi_acc", "onset_order_acc", "simultaneity_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--classical", action="store_true", help="use the morphological detector")
    ap.add_argument("--legacy", action="store_true", help="with --classical: the notebook's version, no fixes")
    ap.add_argument("--limit", type=int, default=None, help="pages per song")
    ap.add_argument("--songs", nargs="*", default=None)
    args = ap.parse_args()

    pipe = Pipeline(use_detector=not args.classical, legacy=args.legacy)
    per_song = {}
    total = Tally()
    seconds = 0.0
    for song in doremi.songs():
        if doremi.split_of(song) != args.split or (args.songs and song not in args.songs):
            continue
        pages = doremi.load_song(song)[:args.limit]
        tally = Tally()
        for page in tqdm(pages, desc=song[:32].ljust(32), leave=False):
            gray = cv2.imread(str(page.image_path), cv2.IMREAD_GRAYSCALE)
            t0 = time.time()
            result = pipe.run(gray)
            seconds += time.time() - t0
            tally.add(score_page(result, page))
        per_song[song] = tally.summary()
        total.add(tally)

    per_song["ALL"] = total.summary()
    header = f"{'song':36s}" + "".join(f"{c[:10]:>11s}" for c in COLS)
    print("\n" + header)
    for song, s in per_song.items():
        print(f"{song[:36]:36s}" + "".join(f"{s[c]:>11}" for c in COLS))
    print(f"\n{seconds / max(total.pages, 1):.2f} s per page")
    conf = {f"{a}->{b}": v for (a, b), v in sorted(total.confusion.items())}
    print("class confusion (true->pred):", conf)

    Path("results").mkdir(exist_ok=True)
    detector = "legacy" if args.legacy else "classical" if args.classical else "cnn"
    name = f"results/{args.split}_{detector}.json"
    with open(name, "w") as f:
        json.dump({"per_song": per_song, "confusion": conf, "seconds_per_page": seconds / max(total.pages, 1)}, f, indent=2)
    print("wrote", name)


if __name__ == "__main__":
    main()
