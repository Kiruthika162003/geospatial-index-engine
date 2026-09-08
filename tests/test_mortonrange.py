from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.morton import encode
from atlas.mortonrange import cell_count, decompose, naive_interval, span

ORDER = 8
SIDE = 1 << ORDER


class TestExactness:
    def test_the_decomposition_covers_exactly_the_box_and_nothing_else(self):
        rng = random.Random(111)
        for _ in range(300):
            x0, x1 = sorted((rng.randint(0, SIDE - 1), rng.randint(0, SIDE - 1)))
            y0, y1 = sorted((rng.randint(0, SIDE - 1), rng.randint(0, SIDE - 1)))
            intervals = decompose(x0, y0, x1, y1, ORDER)
            codes = set()
            for lo, hi in intervals:
                codes.update(range(lo, hi + 1))
            expected = {encode(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
            assert codes == expected
            assert span(intervals) == cell_count(x0, y0, x1, y1)

    def test_a_whole_quadrant_is_a_single_interval(self):
        half = SIDE // 2
        assert len(decompose(0, 0, half - 1, half - 1, ORDER)) == 1


class TestNaiveWaste:
    def test_the_naive_interval_spans_far_more_than_the_box(self):
        rng = random.Random(111)
        wastes = []
        for _ in range(300):
            x0, x1 = sorted((rng.randint(0, SIDE - 1), rng.randint(0, SIDE - 1)))
            y0, y1 = sorted((rng.randint(0, SIDE - 1), rng.randint(0, SIDE - 1)))
            lo, hi = naive_interval(x0, y0, x1, y1)
            wastes.append((hi - lo + 1) / cell_count(x0, y0, x1, y1))
        assert sum(wastes) / len(wastes) > 5  # measured mean about 10x
        assert max(wastes) > 100  # measured max about 457x

    def test_a_seam_straddling_box_wastes_two_thousand_fold(self):
        s = SIDE // 2
        x0, y0, x1, y1 = s - 2, s - 2, s + 1, s + 1
        lo, hi = naive_interval(x0, y0, x1, y1)
        assert cell_count(x0, y0, x1, y1) == 16
        assert hi - lo + 1 == 32776  # 2048 times the 16 cells
        intervals = decompose(x0, y0, x1, y1, ORDER)
        assert len(intervals) == 4
        assert span(intervals) == 16


class TestRefusals:
    def test_an_inverted_box_is_refused(self):
        with pytest.raises(Invalid):
            decompose(5, 5, 2, 2, ORDER)

    def test_a_box_outside_the_grid_is_refused(self):
        with pytest.raises(Invalid):
            decompose(0, 0, SIDE, SIDE, ORDER)

    def test_a_negative_order_is_refused(self):
        with pytest.raises(Invalid):
            decompose(0, 0, 1, 1, -1)
