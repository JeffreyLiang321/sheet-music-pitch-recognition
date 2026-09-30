"""Turn detected notes into something playable.

Noteheads are grouped into columns (things that sound together) by their
x position within a system. Each column lasts as long as its shortest
note, so a whole note in the left hand keeps ringing under four quarters
in the right. Systems, and then pages, are played one after another.

The result is a flat list of note events plus MIDI / MusicXML export via
music21.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .pipeline import PageResult, Note
from .staff import UNIT

DEFAULT_BPM = 90
COLUMN_GAP = 0.6 * UNIT


@dataclass
class Event:
    onset: float  # beats
    duration: float
    midi: int
    name: str
    cls: str
    part: int  # staff index within its system
    page: int
    x0: int
    y0: int
    x1: int
    y1: int


def _columns(notes: list[Note]) -> list[list[Note]]:
    notes = sorted(notes, key=lambda n: n.cx)
    cols: list[list[Note]] = []
    for n in notes:
        if cols and n.cx - cols[-1][-1].cx <= COLUMN_GAP:
            cols[-1].append(n)
        else:
            cols.append([n])

    # Seconds in a chord are drawn on opposite sides of the stem, which
    # puts them slightly apart in x. Merge neighbouring columns that hold
    # such a pair on the same staff.
    merged: list[list[Note]] = []
    for col in cols:
        if merged and any(a.staff == b.staff and abs(a.position - b.position) == 1
                          and abs(a.cx - b.cx) < 1.4 * UNIT
                          for a in merged[-1] for b in col):
            merged[-1].extend(col)
        else:
            merged.append(col)
    return merged


def assign_timing(pages: list[PageResult]) -> list[Event]:
    events: list[Event] = []
    t = 0.0
    for page_no, page in enumerate(pages):
        for system in page.systems:
            notes = [n for n in page.notes if n.staff in system]
            for col_no, col in enumerate(_columns(notes)):
                for n in col:
                    n.onset, n.column = t, col_no
                    x0, y0, x1, y1 = n.box
                    s = page.scale
                    events.append(Event(
                        onset=t, duration=n.duration, midi=n.midi,
                        name=n.pitch.name if n.pitch else "", cls=n.cls,
                        part=page.staffs[n.staff].index_in_system,
                        page=page_no, x0=int(x0 / s), y0=int(y0 / s), x1=int(x1 / s), y1=int(y1 / s)))
                t += min(n.duration for n in col)
    return events


def to_music21(events: list[Event], bpm: int = DEFAULT_BPM):
    from music21 import stream, note, chord, tempo, instrument

    score = stream.Score()
    score.insert(0, tempo.MetronomeMark(number=bpm))
    parts = sorted({e.part for e in events})
    for p in parts:
        part = stream.Part(id=f"staff{p}")
        part.insert(0, instrument.Piano())
        by_onset: dict[float, list[Event]] = {}
        for e in events:
            if e.part == p:
                by_onset.setdefault(e.onset, []).append(e)
        for onset, group in sorted(by_onset.items()):
            pitches = sorted({e.midi for e in group})
            dur = max(e.duration for e in group)
            obj = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            obj.quarterLength = dur
            part.insert(onset, obj)
        score.insert(0, part)
    return score


def write_midi(events: list[Event], path, bpm: int = DEFAULT_BPM) -> None:
    to_music21(events, bpm).write("midi", fp=str(path))


def write_musicxml(events: list[Event], path, bpm: int = DEFAULT_BPM) -> None:
    to_music21(events, bpm).write("musicxml", fp=str(path))


def events_json(events: list[Event]) -> list[dict]:
    return [asdict(e) for e in events]
