from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.unionarea import raster_area, summed_area, union_area


class TestKnownLayouts:
    def test_two_identical_stacked_boxes_count_once(self):
        two = [BBox(0, 0, 10, 10), BBox(0, 0, 10, 10)]
        assert union_area(two) == pytest.approx(100.0)
        assert summed_area(two) == pytest.approx(200.0)

    def test_disjoint_boxes_sum_exactly(self):
        boxes = [BBox(0, 0, 1, 1), BBox(5, 5, 7, 8), BBox(20, 0, 21, 3)]
        assert union_area(boxes) == pytest.approx(summed_area(boxes))

    def test_the_shifted_square_pair(self):
        assert union_area([BBox(0, 0, 10, 10), BBox(5, 5, 15, 15)]) == pytest.approx(175.0)

    def test_empty_is_zero(self):
        assert union_area([]) == 0.0


class TestBounds:
    def test_union_never_exceeds_the_sum_and_matches_a_raster(self):
        rng = random.Random(103)
        gaps = []
        for _ in range(60):  # the raster check is the slow part; 60 layouts suffice
            boxes = []
            for _ in range(rng.randint(1, 8)):
                x, y = rng.uniform(0, 20), rng.uniform(0, 20)
                boxes.append(BBox(x, y, x + rng.uniform(0.5, 6), y + rng.uniform(0.5, 6)))
            u = union_area(boxes)
            assert u <= summed_area(boxes) + 1e-9
            gaps.append(abs(u - raster_area(boxes, 0.05)) / u)
        assert sum(gaps) / len(gaps) < 0.02  # measured 0.42 percent

    def test_disjoint_random_boxes_equal_their_sum_exactly(self):
        rng = random.Random(104)
        for _ in range(300):
            boxes = [
                BBox(i * 10, 0, i * 10 + rng.uniform(1, 9), rng.uniform(1, 9))
                for i in range(rng.randint(1, 6))
            ]
            assert union_area(boxes) == pytest.approx(summed_area(boxes))


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            union_area(None)

    def test_a_non_positive_raster_cell_is_refused(self):
        with pytest.raises(Invalid):
            raster_area([BBox(0, 0, 1, 1)], 0)
