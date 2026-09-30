from omr.pitch import pitch_at, letter_octave, staff_position, key_from_accidentals, key_alteration


def test_middle_line_per_clef():
    assert letter_octave(0, "gClef") == ("B", 4)
    assert letter_octave(0, "fClef") == ("D", 3)
    assert letter_octave(0, "cClef") == ("C", 4)


def test_positions_walk_the_scale():
    # treble: bottom line E4 is four half-spaces below the middle line
    assert letter_octave(-4, "gClef") == ("E", 4)
    assert letter_octave(-6, "gClef") == ("C", 4)  # first ledger line below
    assert letter_octave(4, "gClef") == ("F", 5)   # top line
    # bass: top line A3, first ledger line above is C4
    assert letter_octave(4, "fClef") == ("A", 3)
    assert letter_octave(6, "fClef") == ("C", 4)


def test_staff_position_rounds_to_nearest_half_space():
    center, spacing = 100.0, 20.0
    assert staff_position(100.0, center, spacing) == 0
    assert staff_position(90.0, center, spacing) == 1   # one space up
    assert staff_position(121.0, center, spacing) == -2
    assert staff_position(96.0, center, spacing) == 0   # within rounding


def test_midi_numbers():
    assert pitch_at(100.0, 100.0, 20.0, "gClef").midi == 71  # B4
    assert pitch_at(100.0, 100.0, 20.0, "fClef").midi == 50  # D3
    # C4 sits on the first ledger line below treble
    assert pitch_at(160.0, 100.0, 20.0, "gClef").midi == 60


def test_key_signature_alters_the_right_letters():
    assert key_from_accidentals(["accidentalSharp"] * 2) == 2
    assert key_from_accidentals(["accidentalFlat"] * 3) == -3
    assert key_from_accidentals([]) == 0
    assert key_alteration("F", 1) == 1 and key_alteration("C", 1) == 0
    assert key_alteration("B", -1) == -1 and key_alteration("E", -1) == 0
    p = pitch_at(100.0, 100.0, 20.0, "gClef", fifths=5)  # B major has B natural? no: 5 sharps = F C G D A
    assert p.alter == 0 and p.name == "B4"
    p = pitch_at(100.0, 100.0, 20.0, "gClef", fifths=-1)
    assert p.name == "Bb4" and p.midi == 70


def test_local_accidental_overrides_key():
    p = pitch_at(100.0, 100.0, 20.0, "gClef", fifths=-2, accidental=0)  # natural cancels the flat
    assert p.name == "B4"
