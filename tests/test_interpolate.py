from __future__ import annotations

import random

import pytest

from atlas.crosstrack import cross_track_km
from atlas.errors import Invalid, Outside
from atlas.haversine import haversine
from atlas.interpolate import intermediate, naive_midpoint, sample


class TestOnTheRoute:
    def test_slerp_points_have_zero_cross_track_and_honor_the_fraction(self):
        rng = random.Random(93)
        for _ in range(3000):
            la1, lo1 = rng.uniform(-60, 60), rng.uniform(-120, 120)
            la2, lo2 = rng.uniform(-60, 60), rng.uniform(-120, 120)
            total = haversine(la1, lo1, la2, lo2)
            if total < 1:
                continue
            f = rng.uniform(0, 1)
            p = intermediate(la1, lo1, la2, lo2, f)
            assert abs(cross_track_km(la1, lo1, la2, lo2, *p)) < 1e-6
            assert haversine(la1, lo1, *p) == pytest.approx(f * total, abs=1e-6)

    def test_the_endpoints_are_honored(self):
        assert intermediate(10, 20, 30, 40, 0) == pytest.approx((10, 20))
        assert intermediate(10, 20, 30, 40, 1) == pytest.approx((30, 40))

    def test_samples_are_evenly_spaced_along_the_arc(self):
        pts = sample(0, 0, 0, 90, 10)
        gaps = [haversine(*pts[i], *pts[i + 1]) for i in range(9)]
        assert max(gaps) - min(gaps) < 1e-6
        assert gaps[0] == pytest.approx(1111.951, abs=0.01)


class TestTheNaiveAverageDrifts:
    def test_the_true_midpoint_sits_poleward_of_the_naive_one(self):
        mid = intermediate(60, -45, 60, 45, 0.5)
        naive = naive_midpoint(60, -45, 60, 45)
        assert mid[0] == pytest.approx(67.79, abs=0.05)
        assert naive[0] == pytest.approx(60.0)
        assert abs(cross_track_km(60, -45, 60, 45, *naive)) == pytest.approx(866.5, abs=1)

    def test_the_naive_midpoint_is_far_off_route_on_average(self):
        rng = random.Random(93)
        offs = []
        for _ in range(3000):
            la1, lo1 = rng.uniform(-60, 60), rng.uniform(-120, 120)
            la2, lo2 = rng.uniform(-60, 60), rng.uniform(-120, 120)
            if haversine(la1, lo1, la2, lo2) < 1:
                continue
            naive = naive_midpoint(la1, lo1, la2, lo2)
            offs.append(abs(cross_track_km(la1, lo1, la2, lo2, *naive)))
        assert sum(offs) / len(offs) > 500  # measured about 1329 km


class TestRefusals:
    def test_a_fraction_outside_zero_one_is_refused(self):
        with pytest.raises(Invalid):
            intermediate(0, 0, 10, 10, 1.5)

    def test_antipodal_endpoints_are_refused(self):
        with pytest.raises(Invalid):
            intermediate(0, 0, 0, 180, 0.5)

    def test_a_latitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            intermediate(91, 0, 0, 0, 0.5)

    def test_fewer_than_two_samples_is_refused(self):
        with pytest.raises(Invalid):
            sample(0, 0, 1, 1, 1)
