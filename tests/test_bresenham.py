from __future__ import annotations

import random

import pytest

from atlas.bresenham import is_eight_connected, line
from atlas.errors import Invalid


def _rounded(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return [(x0, y0)]
    if abs(dx) >= abs(dy):
        sx = 1 if dx > 0 else -1
        return [(x0 + i * sx, y0 + round(dy * (i * sx) / dx)) for i in range(abs(dx) + 1)]
    sy = 1 if dy > 0 else -1
    return [(x0 + round(dx * (i * sy) / dy), y0 + i * sy) for i in range(abs(dy) + 1)]


class TestKnownLines:
    def test_a_diagonal(self):
        assert line(0, 0, 3, 3) == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_a_shallow_line(self):
        assert line(0, 0, 6, 2) == [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2)]

    def test_a_steep_line(self):
        assert line(0, 0, 2, 6) == [(0, 0), (0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

    def test_a_single_cell(self):
        assert line(4, 4, 4, 4) == [(4, 4)]


class TestInvariants:
    def test_eight_connected_with_exactly_max_delta_plus_one_cells(self):
        rng = random.Random(83)
        for _ in range(20000):
            x0, y0, x1, y1 = (rng.randint(-40, 40) for _ in range(4))
            cells = line(x0, y0, x1, y1)
            assert is_eight_connected(cells)
            assert len(cells) == max(abs(x1 - x0), abs(y1 - y0)) + 1
            assert cells[0] == (x0, y0)
            assert cells[-1] == (x1, y1)

    def test_it_never_strays_more_than_a_tie_from_real_rounding(self):
        rng = random.Random(83)
        exact = total = 0
        for _ in range(5000):
            x0, y0, x1, y1 = (rng.randint(-40, 40) for _ in range(4))
            cells = line(x0, y0, x1, y1)
            ref = _rounded(x0, y0, x1, y1)
            total += 1
            if cells == ref:
                exact += 1
            for a, b in zip(cells, ref, strict=True):
                assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
        # measured 77% identical; the rest differ only at half-step ties
        assert exact / total > 0.7


class TestRefusals:
    def test_non_integer_coordinates_are_refused(self):
        with pytest.raises(Invalid):
            line(0.5, 0, 3, 3)
