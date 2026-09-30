"""Vertical position -> pitch.

A notehead's pitch is read off its distance from the middle staff line in
half-spaces, which the clef turns into a letter and octave. The key
signature (and any accidental in front of the head) then decides the
chromatic alteration, giving a MIDI number.
"""
from __future__ import annotations

from dataclasses import dataclass

LETTERS = "CDEFGAB"
SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# letter and octave of the middle line for each clef. "cClef" is the alto
# clef; the same glyph one line higher is the tenor clef.
MIDDLE_LINE = {"gClef": ("B", 4), "fClef": ("D", 3), "cClef": ("C", 4), "tenorClef": ("A", 3)}

SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"


@dataclass
class Pitch:
    letter: str
    octave: int
    alter: int  # -1 flat, 0 natural, +1 sharp

    @property
    def midi(self) -> int:
        return 12 * (self.octave + 1) + SEMITONES[self.letter] + self.alter

    @property
    def name(self) -> str:
        return f"{self.letter}{'#' if self.alter > 0 else 'b' if self.alter < 0 else ''}{self.octave}"


def staff_position(cy: float, center: float, spacing: float) -> int:
    """Half-spaces above the middle line (negative = below)."""
    return int(round((center - cy) / (spacing / 2)))


def letter_octave(position: int, clef: str) -> tuple[str, int]:
    letter, octave = MIDDLE_LINE.get(clef, MIDDLE_LINE["gClef"])
    diatonic = octave * 7 + LETTERS.index(letter) + position
    return LETTERS[diatonic % 7], diatonic // 7


def key_alteration(letter: str, fifths: int) -> int:
    if fifths > 0 and letter in SHARP_ORDER[:fifths]:
        return 1
    if fifths < 0 and letter in FLAT_ORDER[:-fifths]:
        return -1
    return 0


def pitch_at(cy: float, center: float, spacing: float, clef: str,
             fifths: int = 0, accidental: int | None = None) -> Pitch:
    letter, octave = letter_octave(staff_position(cy, center, spacing), clef)
    alter = key_alteration(letter, fifths) if accidental is None else accidental
    return Pitch(letter, octave, alter)


def key_from_accidentals(kinds: list[str]) -> int:
    """Key signature in fifths from the run of accidentals after the clef."""
    sharps = sum(k == "accidentalSharp" for k in kinds)
    flats = sum(k == "accidentalFlat" for k in kinds)
    if sharps and not flats:
        return min(sharps, 7)
    if flats and not sharps:
        return -min(flats, 7)
    return 0
