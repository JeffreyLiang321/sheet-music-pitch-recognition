import cv2
import numpy as np

from omr.staff import UNIT, binarize, detect_staffs, normalize_scale, staff_for_y


def draw_page(spacing=UNIT, systems=2, staves_per_system=2):
    """White page with `systems` systems, each `staves_per_system` staves joined by a left barline."""
    h, w = 1400, 1000
    page = np.full((h, w), 255, dtype=np.uint8)
    y = 120
    tops = []
    for _ in range(systems):
        first = y
        for _ in range(staves_per_system):
            tops.append(y)
            for i in range(5):
                cv2.line(page, (100, int(y + i * spacing)), (900, int(y + i * spacing)), 0, 3)
            y += 4 * spacing + 6 * spacing  # staff height + gap inside the system
        # barline joining the staves of this system
        cv2.line(page, (100, int(first)), (100, int(tops[-1] + 4 * spacing)), 0, 4)
        y += 8 * spacing  # bigger gap between systems
    return page, tops


def test_detects_staves_and_groups_systems():
    page, tops = draw_page()
    staffs = detect_staffs(binarize(page))
    assert len(staffs) == 4
    assert [round(s.top) for s in staffs] == [round(t) for t in tops]
    assert [s.system for s in staffs] == [0, 0, 1, 1]
    assert [s.index_in_system for s in staffs] == [0, 1, 0, 1]
    assert all(abs(s.spacing - UNIT) < 0.6 for s in staffs)
    assert all(90 <= s.left <= 110 and 890 <= s.right <= 910 for s in staffs)


def test_ignores_stray_horizontal_rule():
    page, _ = draw_page(systems=1)
    cv2.line(page, (100, 60), (900, 60), 0, 3)  # a title rule above the music
    staffs = detect_staffs(binarize(page))
    assert len(staffs) == 2


def test_normalize_scale_rescales_to_unit():
    page, _ = draw_page(spacing=UNIT * 1.5, systems=1)
    scaled, scale = normalize_scale(page)
    assert abs(scale - 1 / 1.5) < 0.03
    staffs = detect_staffs(binarize(scaled))
    assert len(staffs) == 2 and all(abs(s.spacing - UNIT) < 1.0 for s in staffs)


def test_staff_for_y_prefers_containing_staff():
    page, tops = draw_page(systems=1)
    staffs = detect_staffs(binarize(page))
    assert staff_for_y(staffs, tops[0] + 2 * UNIT) is staffs[0]
    assert staff_for_y(staffs, tops[1] - 2 * UNIT) is staffs[1]  # ledger line above the second staff
