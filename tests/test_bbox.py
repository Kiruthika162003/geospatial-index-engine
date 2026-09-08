from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid


class TestBasics:
    def test_area_margin_and_center(self):
        b = BBox(0, 0, 2, 4)
        assert b.area() == 8
        assert b.margin() == 6
        assert b.center() == (1.0, 2.0)

    def test_containment(self):
        a = BBox(0, 0, 2, 2)
        assert a.contains_point(1, 1)
        assert not a.contains_point(3, 1)
        assert a.contains_box(BBox(0.5, 0.5, 1.5, 1.5))
        assert not a.contains_box(BBox(1, 1, 3, 3))

    def test_intersection(self):
        a = BBox(0, 0, 2, 2)
        b = BBox(1, 1, 3, 3)
        assert a.intersects(b)
        assert a.intersection_area(b) == 1
        assert not a.intersects(BBox(5, 5, 6, 6))
        assert a.intersection_area(BBox(5, 5, 6, 6)) == 0.0

    def test_touching_boxes_intersect_with_zero_area(self):
        a = BBox(0, 0, 1, 1)
        b = BBox(1, 0, 2, 1)  # shares an edge
        assert a.intersects(b)
        assert a.intersection_area(b) == 0.0

    def test_union_and_enlargement(self):
        a = BBox(0, 0, 2, 2)
        b = BBox(1, 1, 3, 3)
        assert a.union(b) == BBox(0, 0, 3, 3)
        assert a.enlargement(b) == 5  # union area 9 minus a's area 4

    def test_from_points(self):
        assert BBox.from_points([(1, 5), (3, 2), (-1, 4)]) == BBox(-1, 2, 3, 5)


class TestSuperadditivity:
    def test_enlargement_and_dead_space_are_never_negative(self):
        rng = random.Random(5)

        def rbox():
            x1, x2 = sorted((rng.uniform(0, 10), rng.uniform(0, 10)))
            y1, y2 = sorted((rng.uniform(0, 10), rng.uniform(0, 10)))
            return BBox(x1, y1, x2, y2)

        for _ in range(100000):
            p, q = rbox(), rbox()
            assert p.enlargement(q) >= -1e-9
            assert p.dead_space(q) >= -1e-9


class TestRefusals:
    def test_an_inverted_box_is_refused(self):
        with pytest.raises(Invalid):
            BBox(2, 0, 0, 2)

    def test_bounding_no_points_is_refused(self):
        with pytest.raises(Invalid):
            BBox.from_points([])
