from __future__ import annotations

import math

import pytest

from atlas.errors import Invalid
from atlas.marchingsquares import chain_loops, encloses, is_closed, isoline_segments

N = 41
SUMMIT = (20, 20)
HILL = [[math.exp(-((r - 20) ** 2 + (c - 20) ** 2) / 60.0) for c in range(N)] for r in range(N)]


class TestOnePeak:
    @pytest.mark.parametrize("level", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_each_interior_level_is_one_closed_loop_around_the_summit(self, level):
        loops = chain_loops(isoline_segments(HILL, level))
        assert len(loops) == 1
        assert is_closed(loops[0])
        assert encloses(loops[0], SUMMIT)

    def test_higher_rings_are_smaller(self):
        counts = [len(isoline_segments(HILL, level)) for level in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert counts == [92, 68, 52, 36, 20]

    def test_levels_outside_the_field_produce_nothing(self):
        assert isoline_segments(HILL, 1.5) == []
        assert isoline_segments(HILL, -0.1) == []


class TestCrossings:
    def test_every_endpoint_lies_on_a_cell_edge(self):
        for a, b in isoline_segments(HILL, 0.5):
            for x, y in (a, b):
                assert abs(x - round(x)) < 1e-12 or abs(y - round(y)) < 1e-12

    def test_a_crossing_interpolates_to_exactly_the_level(self):
        (x, y), _ = isoline_segments(HILL, 0.5)[0]
        c0, r0 = math.floor(x), math.floor(y)
        fx, fy = x - c0, y - r0
        if fx < 1e-12:
            value = HILL[r0][c0] * (1 - fy) + HILL[r0 + 1][c0] * fy
        else:
            value = HILL[r0][c0] * (1 - fx) + HILL[r0][c0 + 1] * fx
        assert value == pytest.approx(0.5)


class TestSaddle:
    def test_a_saddle_cell_yields_two_segments(self):
        grid = [[1.0, 0.0], [0.0, 1.0]]  # diagonal corners high, the others low
        assert len(isoline_segments(grid, 0.5)) == 2


class TestRefusals:
    def test_a_grid_too_small_is_refused(self):
        with pytest.raises(Invalid):
            isoline_segments([[1.0]], 0.5)
