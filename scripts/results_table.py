"""Print the README results table from results/test_*.json.

    python scripts/results_table.py
"""
import json
from pathlib import Path

COLUMNS = [
    ("legacy", "notebook as-is"),
    ("classical", "notebook + fixes"),
    ("cnn", "heatmap detector"),
]
ROWS = [
    ("detection precision / recall / F1", lambda s: f"{s['precision']:.3f} / {s['recall']:.3f} / {s['f1']:.3f}"),
    ("quarter / half / whole class accuracy", lambda s: f"{s['class_acc']:.3f}"),
    ("half vs whole (hollow heads only)", lambda s: f"{s['half_vs_whole_acc']:.3f}"),
    ("staff assignment", lambda s: f"{s['staff_assign_acc']:.3f}"),
    ("clef accuracy (per staff)", lambda s: f"{s['clef_acc']:.3f}"),
    ("pitch: letter + octave", lambda s: f"{s['pitch_step_acc']:.3f}"),
    ("pitch: exact MIDI (key sig. + accidentals)", lambda s: f"{s['pitch_midi_acc']:.3f}"),
    ("left-to-right order of notes on a staff", lambda s: f"{s['onset_order_acc']:.3f}"),
    ("simultaneous notes kept simultaneous", lambda s: f"{s['simultaneity_acc']:.3f}"),
]


def main():
    results = {}
    for key, _ in COLUMNS:
        path = Path(f"results/test_{key}.json")
        if path.exists():
            with open(path) as f:
                results[key] = json.load(f)
    cols = [(k, name) for k, name in COLUMNS if k in results]
    first = results[cols[0][0]]["per_song"]["ALL"]
    print(f"{first['pages']} pages, {first['gt_notes']} labelled noteheads\n")
    print("| | " + " | ".join(name for _, name in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, fn in ROWS:
        print(f"| {label} | " + " | ".join(fn(results[k]["per_song"]["ALL"]) for k, _ in cols) + " |")
    print("| time per page (CPU) | " + " | ".join(f"{results[k]['seconds_per_page']:.2f} s" for k, _ in cols) + " |")
    print("\nrecall by class:")
    for k, name in cols:
        print(f"  {name}: {results[k]['per_song']['ALL']['recall_by_class']}")


if __name__ == "__main__":
    main()
