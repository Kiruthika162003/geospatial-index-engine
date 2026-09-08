from __future__ import annotations

import math

import pytest

from atlas.errors import Outside
from atlas.haversine import EARTH_RADIUS_KM, equirectangular, haversine


class TestKnownDistances:
    def test_london_to_paris(self):
        d = haversine(51.5074, -0.1278, 48.8566, 2.3522)
        assert d == pytest.approx(343.6, abs=0.5)

    def test_a_quarter_of_the_equator(self):
        quarter = 2 * math.pi * EARTH_RADIUS_KM / 4
        assert haversine(0, 0, 0, 90) == pytest.approx(quarter)

    def test_a_meridian_quarter_to_the_pole(self):
        quarter = 2 * math.pi * EARTH_RADIUS_KM / 4
        assert haversine(0, 0, 90, 0) == pytest.approx(quarter)

    def test_the_same_point_is_zero(self):
        assert haversine(10, 20, 10, 20) == 0.0

    def test_it_is_symmetric(self):
        a = haversine(12, 34, -56, 78)
        b = haversine(-56, 78, 12, 34)
        assert a == pytest.approx(b)


class TestFlatApproximation:
    def test_the_flat_approximation_is_exact_along_a_meridian(self):
        # no longitude difference means no meridian convergence to get wrong
        for lat in (0, 30, 60, 89):
            h = haversine(0, 5, lat, 5)
            e = equirectangular(0, 5, lat, 5)
            assert e == pytest.approx(h, rel=1e-9)

    def test_the_flat_error_grows_with_latitude_under_longitude_spread(self):
        def reldiff(lat):
            h = haversine(lat, -20, lat, 20)
            e = equirectangular(lat, -20, lat, 20)
            return abs(h - e) / h

        errors = [reldiff(lat) for lat in (0, 30, 45, 60, 80)]
        assert errors[0] == pytest.approx(0.0, abs=1e-9)
        assert errors == sorted(errors)  # monotonically worse toward the pole
        assert reldiff(80) == pytest.approx(0.020, abs=0.002)

    def test_a_long_diagonal_hop_drifts_several_percent(self):
        h = haversine(10, 10, 60, 80)
        e = equirectangular(10, 10, 60, 80)
        assert abs(h - e) / h == pytest.approx(0.065, abs=0.005)


class TestRefusals:
    def test_a_latitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            haversine(91, 0, 0, 0)

    def test_a_longitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            haversine(0, 181, 0, 0)
