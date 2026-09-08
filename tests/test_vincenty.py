from __future__ import annotations

import pytest

from atlas.errors import Invalid, Outside
from atlas.haversine import haversine
from atlas.vincenty import distance_m


class TestKnownDistances:
    def test_london_to_paris(self):
        d = distance_m(51.5074, -0.1278, 48.8566, 2.3522) / 1000
        assert d == pytest.approx(343.9, abs=0.5)

    def test_the_same_point_is_zero(self):
        assert distance_m(10, 20, 10, 20) == 0.0

    def test_it_is_symmetric(self):
        a = distance_m(12, 34, -56, 78)
        b = distance_m(-56, 78, 12, 34)
        assert a == pytest.approx(b)


class TestGapWithHaversine:
    def test_the_sign_of_the_gap_flips_between_polar_and_equatorial(self):
        # north-south: the polar squash makes Vincenty shorter than the mean sphere
        v_ns = distance_m(0, 0, 80, 0) / 1000
        h_ns = haversine(0, 0, 80, 0)
        assert v_ns < h_ns
        assert (h_ns - v_ns) == pytest.approx(10.47, abs=0.5)

        # equator: the equatorial bulge makes Vincenty longer than the mean sphere
        v_ew = distance_m(0, 0, 0, 80) / 1000
        h_ew = haversine(0, 0, 0, 80)
        assert v_ew > h_ew
        assert (v_ew - h_ew) == pytest.approx(9.95, abs=0.5)

    def test_the_relative_gap_is_a_few_tenths_of_a_percent(self):
        v = distance_m(51.5074, -0.1278, 48.8566, 2.3522) / 1000
        h = haversine(51.5074, -0.1278, 48.8566, 2.3522)
        assert abs(v - h) / v < 0.005


class TestRefusals:
    def test_a_latitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            distance_m(91, 0, 0, 0)

    def test_near_antipodal_points_fail_to_converge(self):
        with pytest.raises(Invalid):
            distance_m(0, 0, 0.5, 179.7)
