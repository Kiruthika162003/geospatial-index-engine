from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.hilbert import decode, encode


class TestBijection:
    def test_it_is_a_bijection_and_round_trips_on_full_grids(self):
        for order in range(1, 8):
            side = 1 << order
            codes = set()
            for x in range(side):
                for y in range(side):
                    d = encode(x, y, order)
                    codes.add(d)
                    assert decode(d, order) == (x, y)
            assert codes == set(range(side * side))

    def test_the_curve_starts_at_the_origin(self):
        assert decode(0, 4) == (0, 0)


class TestAdjacency:
    def test_consecutive_cells_are_always_one_step_apart(self):
        # this is the property Morton lacks: no seams, worst jump is exactly 1
        order = 5
        side = 1 << order
        pts = [decode(d, order) for d in range(side * side)]
        jumps = [
            abs(pts[i][0] - pts[i + 1][0]) + abs(pts[i][1] - pts[i + 1][1])
            for i in range(len(pts) - 1)
        ]
        assert max(jumps) == 1
        assert min(jumps) == 1

    def test_adjacency_holds_across_orders(self):
        for order in range(1, 7):
            side = 1 << order
            pts = [decode(d, order) for d in range(side * side)]
            worst = max(
                abs(pts[i][0] - pts[i + 1][0]) + abs(pts[i][1] - pts[i + 1][1])
                for i in range(len(pts) - 1)
            )
            assert worst == 1


class TestRefusals:
    def test_a_non_positive_order_is_refused(self):
        with pytest.raises(Invalid):
            encode(0, 0, 0)

    def test_a_cell_off_the_grid_is_refused(self):
        with pytest.raises(Invalid):
            encode(16, 0, 4)  # grid is 16x16, valid 0..15

    def test_a_distance_past_the_end_is_refused(self):
        with pytest.raises(Invalid):
            decode(256, 4)  # length is 256, valid 0..255
