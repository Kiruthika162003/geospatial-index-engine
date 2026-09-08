from __future__ import annotations

import random

import pytest

from atlas.bearing import destination
from atlas.crosstrack import along_track_km, cross_track_km
from atlas.errors import Outside


class TestOnRoute:
    def test_a_point_on_the_route_has_zero_cross_track(self):
        assert cross_track_km(0, 0, 0, 10, 0, 4) == pytest.approx(0.0, abs=1e-6)

    def test_along_track_is_the_distance_from_the_start(self):
        assert along_track_km(0, 0, 0, 10, 0, 4) == pytest.approx(444.78, abs=0.1)


class TestSign:
    def test_left_of_an_eastward_route_is_negative_right_is_positive(self):
        assert cross_track_km(0, 0, 0, 10, 1, 4) == pytest.approx(-111.2, abs=0.1)
        assert cross_track_km(0, 0, 0, 10, -1, 4) == pytest.approx(111.2, abs=0.1)

    def test_mirroring_across_the_route_flips_the_sign_exactly(self):
        rng = random.Random(73)
        for _ in range(3000):
            la1, lo1 = rng.uniform(-60, 60), rng.uniform(-150, 150)
            brg = rng.uniform(0, 360)
            la2, lo2 = destination(la1, lo1, brg, rng.uniform(100, 2000))
            along, off = rng.uniform(10, 900), rng.uniform(1, 300)
            lf, lof = destination(la1, lo1, brg, along)
            left = destination(lf, lof, (brg - 90) % 360, off)
            right = destination(lf, lof, (brg + 90) % 360, off)
            xl = cross_track_km(la1, lo1, la2, lo2, *left)
            xr = cross_track_km(la1, lo1, la2, lo2, *right)
            assert xl + xr == pytest.approx(0.0, abs=1e-6)
            assert cross_track_km(la1, lo1, la2, lo2, lf, lof) == pytest.approx(0.0, abs=1e-6)


class TestRefusals:
    def test_a_degenerate_route_is_refused(self):
        with pytest.raises(Outside):
            cross_track_km(5, 5, 5, 5, 6, 6)

    def test_along_track_refuses_a_degenerate_route_too(self):
        with pytest.raises(Outside):
            along_track_km(5, 5, 5, 5, 6, 6)
