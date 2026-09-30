from omr.pipeline import Note, PageResult
from omr.pitch import Pitch
from omr.score import assign_timing, to_music21
from omr.staff import Staff, UNIT


def _staff(top, system=0, idx=0):
    return Staff(line_ys=[top + i * UNIT for i in range(5)], left=100, right=2000, system=system, index_in_system=idx)


def _note(cx, cy, staff, cls="noteheadBlack", midi=60):
    dur = {"noteheadBlack": 1.0, "noteheadHalf": 2.0, "noteheadWhole": 4.0}[cls]
    n = Note(cx=cx, cy=cy, score=1.0, cls=cls, staff=staff, duration=dur)
    n.pitch = Pitch("C", 4, 0)
    n.position = 0
    return n


def test_whole_note_holds_under_quarters():
    page = PageResult(width=1000, height=1000, scale=1.0, staffs=[_staff(100, idx=0), _staff(400, idx=1)])
    page.notes = [_note(200 + i * 60, 140, 0) for i in range(4)] + [_note(200, 440, 1, "noteheadWhole")]
    events = assign_timing([page])
    quarters = sorted(e.onset for e in events if e.part == 0)
    whole = [e for e in events if e.part == 1][0]
    assert quarters == [0.0, 1.0, 2.0, 3.0]
    assert whole.onset == 0.0 and whole.duration == 4.0


def test_chord_notes_share_an_onset():
    page = PageResult(width=1000, height=1000, scale=1.0, staffs=[_staff(100)])
    page.notes = [_note(200, 140, 0), _note(203, 160, 0), _note(300, 140, 0)]
    events = assign_timing([page])
    assert [e.onset for e in events] == [0.0, 0.0, 1.0]


def test_seconds_in_a_chord_are_merged():
    page = PageResult(width=1000, height=1000, scale=1.0, staffs=[_staff(100)])
    a, b = _note(200, 140, 0), _note(200 + 1.1 * UNIT, 150, 0)
    a.position, b.position = 2, 1
    page.notes = [a, b]
    events = assign_timing([page])
    assert events[0].onset == events[1].onset == 0.0


def test_systems_play_in_order():
    s1, s2 = _staff(100, system=0), _staff(800, system=1)
    page = PageResult(width=1000, height=1000, scale=1.0, staffs=[s1, s2])
    page.notes = [_note(200, 140, 0), _note(200, 840, 1)]
    events = assign_timing([page])
    assert [e.onset for e in events] == [0.0, 1.0]


def test_music21_export_round_trip(tmp_path):
    page = PageResult(width=1000, height=1000, scale=1.0, staffs=[_staff(100)])
    page.notes = [_note(200, 140, 0, midi=60), _note(260, 140, 0, "noteheadHalf")]
    events = assign_timing([page])
    score = to_music21(events)
    assert len(score.parts) == 1
    assert [n.quarterLength for n in score.parts[0].notes] == [1.0, 2.0]
    score.write("midi", fp=str(tmp_path / "out.mid"))
    assert (tmp_path / "out.mid").stat().st_size > 0
