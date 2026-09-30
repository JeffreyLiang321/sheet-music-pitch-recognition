"""Reader for the DoReMi dataset ground truth.

DoReMi ships one OMR_XML file per song with a <Page> per rendered image.
Every object on the page is a <Node> with a class name, a bounding box and
(for the classes we care about) a <Data> block with the musical meaning:
noteheads carry their MIDI pitch, onset and duration, clefs carry their
type, and staff lines carry the id of the staff they belong to.

Parsing the big XML files is slow, so the parsed pages are cached as JSON
under data/cache the first time a song is loaded.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from pathlib import Path

DOREMI_DIR = Path(os.environ.get("DOREMI_DIR", "data/DoReMi_v1"))
CACHE_DIR = Path(os.environ.get("DOREMI_CACHE", "data/cache"))

NOTEHEAD_CLASSES = ("noteheadBlack", "noteheadHalf", "noteheadWhole")
CLEF_CLASSES = ("gClef", "fClef", "cClef")
ACCIDENTAL_CLASSES = ("accidentalSharp", "accidentalFlat", "accidentalNatural")

# Fixed song-level split, chosen before any model was trained on this
# machine. Test songs are mostly piano (the target use case) plus one
# string quartet so the C clef gets exercised. The "beam groups ..." and
# "accidentals ..." songs are synthetic notation exercises, not real pieces,
# so they only ever appear in training.
TEST_SONGS = (
    "Chopin - Etude Op 10 no 1",
    "Satie - Gnossienne 1",
    "Schumann - String Quartet 1 mvt 3",
    "Bach - Goldberg Variation 16",
    "Reger - Improvisationen 4",
    "Mendelssohn - Songs Without Words 17",
)
VAL_SONGS = (
    "Bartok - Mikrokosmos 144",
    "Webern - Variations Op 27 III",
)


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class GTNote:
    cls: str
    box: Box
    staff_id: int
    midi: int
    step: str
    octave: int
    onset: float
    duration: float


@dataclass
class GTClef:
    cls: str
    box: Box
    staff_id: int
    hotspot: str  # e.g. "G4" for a treble clef


@dataclass
class GTStaff:
    staff_id: int
    line_ys: list[int]  # 5 line centres, top to bottom
    left: int
    right: int

    @property
    def spacing(self) -> float:
        return (self.line_ys[-1] - self.line_ys[0]) / 4

    @property
    def center(self) -> float:
        return self.line_ys[2]


@dataclass
class GTSymbol:
    cls: str
    box: Box
    staff_id: int | None = None


@dataclass
class GTPage:
    song: str
    page_index: int
    notes: list[GTNote] = field(default_factory=list)
    clefs: list[GTClef] = field(default_factory=list)
    staffs: list[GTStaff] = field(default_factory=list)
    accidentals: list[GTSymbol] = field(default_factory=list)
    # time signature digits and rests; used as negatives when training the
    # accidental classifier, since they sit in the same part of the staff
    others: list[GTSymbol] = field(default_factory=list)

    @property
    def image_path(self) -> Path:
        return DOREMI_DIR / "Images" / f"{self.song}-{self.page_index + 1:03d}.png"

    def clef_for_staff(self, staff_id: int) -> GTClef | None:
        # First clef on the staff reading left to right.
        clefs = [c for c in self.clefs if c.staff_id == staff_id]
        return min(clefs, key=lambda c: c.box.x0) if clefs else None


def songs() -> list[str]:
    files = sorted((DOREMI_DIR / "OMR_XML").glob("*.xml"))
    return [f.name.replace("-layout-0-muscima.xml", "") for f in files]


def split_of(song: str) -> str:
    if song in TEST_SONGS:
        return "test"
    if song in VAL_SONGS:
        return "val"
    return "train"


def _data(node) -> dict:
    d = node.find("Data")
    return {} if d is None else {item.get("key"): item.text for item in d}


def _box(node) -> Box:
    t, l = int(node.findtext("Top")), int(node.findtext("Left"))
    w, h = int(node.findtext("Width")), int(node.findtext("Height"))
    return Box(l, t, l + w, t + h)


def _parse_page(song: str, page) -> GTPage:
    gt = GTPage(song=song, page_index=int(page.get("pageIndex")))
    lines: dict[int, list] = {}
    seen: set = set()
    for node in page.iter("Node"):
        cls = node.findtext("ClassName")
        box = _box(node)
        data = _data(node)
        # Some pages contain each object twice, the second copy stripped of
        # its Data block. Keep whichever copy came first with data attached.
        key = (cls, box.x0, box.y0, box.x1, box.y1)
        if key in seen:
            continue
        if cls in NOTEHEAD_CLASSES:
            if "midi_pitch_code" not in data:
                continue
            seen.add(key)
            gt.notes.append(GTNote(
                cls=cls, box=box, staff_id=int(data["staff_id"]),
                midi=int(data["midi_pitch_code"]),
                step=data["normalized_pitch_step"],
                octave=int(data["pitch_octave"]),
                onset=float(data["onset_beats"]),
                duration=float(data["duration_beats"]),
            ))
        elif cls in CLEF_CLASSES:
            if "staff_id" not in data:
                continue
            seen.add(key)
            gt.clefs.append(GTClef(cls=cls, box=box, staff_id=int(data["staff_id"]),
                                   hotspot=data.get("clef_hotspot", "")))
        elif cls == "kStaffLine":
            if "staff_id" not in data:
                continue
            seen.add(key)
            lines.setdefault(int(data["staff_id"]), []).append(box)
        elif cls in ACCIDENTAL_CLASSES:
            seen.add(key)
            sid = data.get("staff_id")
            gt.accidentals.append(GTSymbol(cls=cls, box=box, staff_id=int(sid) if sid else None))
        elif cls.startswith("timeSig") or cls.startswith("rest"):
            seen.add(key)
            sid = data.get("staff_id")
            gt.others.append(GTSymbol(cls=cls, box=box, staff_id=int(sid) if sid else None))

    for sid, boxes in lines.items():
        ys = sorted({int(b.cy) for b in boxes})
        if len(ys) != 5:
            continue
        gt.staffs.append(GTStaff(staff_id=sid, line_ys=ys,
                                 left=min(b.x0 for b in boxes), right=max(b.x1 for b in boxes)))
    gt.staffs.sort(key=lambda s: s.line_ys[0])
    return gt


def _from_dict(d: dict) -> GTPage:
    def box(b):
        return Box(**b)
    return GTPage(
        song=d["song"], page_index=d["page_index"],
        notes=[GTNote(**{**n, "box": box(n["box"])}) for n in d["notes"]],
        clefs=[GTClef(**{**c, "box": box(c["box"])}) for c in d["clefs"]],
        staffs=[GTStaff(**s) for s in d["staffs"]],
        accidentals=[GTSymbol(**{**a, "box": box(a["box"])}) for a in d["accidentals"]],
        others=[GTSymbol(**{**a, "box": box(a["box"])}) for a in d.get("others", [])],
    )


def load_song(song: str) -> list[GTPage]:
    cache = CACHE_DIR / f"{song}.json"
    if cache.exists():
        with open(cache) as f:
            return [_from_dict(d) for d in json.load(f)]

    root = ET.parse(DOREMI_DIR / "OMR_XML" / f"{song}-layout-0-muscima.xml").getroot()
    pages = [_parse_page(song, p) for p in root.findall("Page")]
    pages = [p for p in pages if p.image_path.exists()]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump([asdict(p) for p in pages], f)
    return pages


def load_split(split: str) -> list[GTPage]:
    pages = []
    for song in songs():
        if split_of(song) == split:
            pages.extend(load_song(song))
    return pages
