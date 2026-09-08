from __future__ import annotations

import math
import random

import pytest

from atlas.boundingcircle import _in_circle, smallest_enclosing_circle
from atlas.errors import Invalid


class TestKnownShapes:
    def test_a_square_is_pinned_by_its_corners(self):
        (cx, cy), r = smallest_enclosing_circle([(0, 0), (2, 0), (2, 2), (0, 2)])
        assert (cx, cy) == pytest.approx((1.0, 1.0))
        assert r == pytest.approx(math.sqrt(2))

    def test_two_points_give_the_diameter_circle(self):
        (cx, cy), r = smallest_enclosing_circle([(0, 0), (6, 8)])
        assert (cx, cy) == pytest.approx((3.0, 4.0))
        assert r == pytest.approx(5.0)

    def test_a_single_point_is_a_zero_circle(self):
        (cx, cy), r = smallest_enclosing_circle([(7, 7)])
        assert (cx, cy) == (7, 7)
        assert r == 0.0


class TestContainmentAndMinimality:
    def test_it_contains_every_point_and_cannot_shrink(self):
        rng = random.Random(69)
        for _ in range(3000):
            n = rng.randint(1, 40)
            pts = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(n)]
            center, r = smallest_enclosing_circle(pts)
            assert all(_in_circle((center, r), p) for p in pts)
            if n >= 2:
                # shrinking by a tenth of a percent leaves some point outside
                assert not all(_in_circle((center, r * 0.999), p) for p in pts)

    def test_at_most_three_points_lie_on_the_boundary(self):
        rng = random.Random(70)
        for _ in range(2000):
            pts = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(30)]
            (cx, cy), r = smallest_enclosing_circle(pts)
            on = sum(1 for p in pts if abs(math.hypot(p[0] - cx, p[1] - cy) - r) < 1e-7)
            assert 2 <= on <= 3

    def test_the_answer_does_not_depend_on_the_shuffle(self):
        rng = random.Random(71)
        pts = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(50)]
        radii = {round(smallest_enclosing_circle(pts, seed=s)[1], 9) for s in range(10)}
        assert len(radii) == 1


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            smallest_enclosing_circle(None)

    def test_empty_is_refused(self):
        with pytest.raises(Invalid):
            smallest_enclosing_circle([])
