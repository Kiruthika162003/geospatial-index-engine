from __future__ import annotations

import random

import pytest

from atlas.arcintersect import arc_intersection, great_circle_crossings
from atlas.bearing import destination
from atlas.crosstrack import cross_track_km
from atlas.errors import Degenerate, Invalid
from atlas.haversine import haversine


class TestPlannedCrossings:
    def test_arcs_built_through_a_point_return_that_point(self):
        rng = random.Random(171)
        for _ in range(500):
            plat, plon = rng.uniform(-60, 60), rng.uniform(-150, 150)
            b1 = rng.uniform(0, 360)
            b2 = (b1 + rng.uniform(30, 150)) % 360
            a = destination(plat, plon, (b1 + 180) % 360, rng.uniform(200, 1500))
            b = destination(plat, plon, b1, rng.uniform(200, 1500))
            c = destination(plat, plon, (b2 + 180) % 360, rng.uniform(200, 1500))
            d = destination(plat, plon, b2, rng.uniform(200, 1500))
            x = arc_intersection(a, b, c, d)
            assert x is not None
            assert haversine(plat, plon, *x) < 1e-6
            assert abs(cross_track_km(*a, *b, *x)) < 1e-9
            assert abs(cross_track_km(*c, *d, *x)) < 1e-9

    def test_the_equator_meets_a_meridian_where_expected(self):
        x = arc_intersection((0, 0), (0, 40), (-10, 20), (10, 20))
        assert x == pytest.approx((0.0, 20.0), abs=1e-9)


class TestRoutesVersusCircles:
    def test_non_overlapping_arcs_do_not_meet_though_their_circles_do(self):
        assert arc_intersection((0, 0), (0, 10), (5, 50), (-5, 60)) is None
        first, second = great_circle_crossings((0, 0), (0, 10), (5, 50), (-5, 60))
        assert first == pytest.approx((0.0, -125.0), abs=1e-6)
        assert second == pytest.approx((0.0, 55.0), abs=1e-6)


class TestRefusals:
    def test_same_circle_arcs_are_refused(self):
        with pytest.raises(Degenerate):
            great_circle_crossings((0, 0), (0, 10), (0, 20), (0, 30))

    def test_a_degenerate_arc_is_refused(self):
        with pytest.raises(Invalid):
            arc_intersection((0, 0), (0, 0), (1, 1), (2, 2))
