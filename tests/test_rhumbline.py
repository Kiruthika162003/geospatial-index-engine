from __future__ import annotations

import random

import pytest

from atlas.errors import Outside
from atlas.haversine import haversine
from atlas.rhumbline import rhumb_bearing, rhumb_distance_km


class TestWhereTheyCoincide:
    def test_the_equator_is_both_a_rhumb_line_and_a_great_circle(self):
        assert rhumb_distance_km(0, 0, 0, 90) == pytest.approx(haversine(0, 0, 0, 90))

    def test_a_meridian_is_both_too(self):
        assert rhumb_distance_km(0, 0, 60, 0) == pytest.approx(haversine(0, 0, 60, 0))


class TestTheExcess:
    def test_the_excess_grows_with_latitude(self):
        def excess(lat):
            r = rhumb_distance_km(lat, -45, lat, 45)
            return r / haversine(lat, -45, lat, 45) - 1

        assert excess(0) == pytest.approx(0.0, abs=1e-9)
        assert excess(30) == pytest.approx(0.0320, abs=0.001)
        assert excess(45) == pytest.approx(0.0607, abs=0.001)
        assert excess(60) == pytest.approx(0.0867, abs=0.001)
        assert excess(70) == pytest.approx(0.0997, abs=0.001)
        assert excess(30) < excess(45) < excess(60) < excess(70)

    def test_the_rhumb_line_is_never_shorter_than_the_great_circle(self):
        rng = random.Random(91)
        for _ in range(5000):
            la1, lo1 = rng.uniform(-80, 80), rng.uniform(-179, 179)
            la2, lo2 = rng.uniform(-80, 80), rng.uniform(-179, 179)
            r = rhumb_distance_km(la1, lo1, la2, lo2)
            g = haversine(la1, lo1, la2, lo2)
            assert r >= g - 1e-6


class TestBearing:
    def test_cardinal_headings(self):
        assert rhumb_bearing(10, 0, 10, 20) == pytest.approx(90.0)
        assert rhumb_bearing(0, 0, 10, 0) == pytest.approx(0.0)
        assert rhumb_bearing(10, 0, 0, 0) == pytest.approx(180.0)

    def test_the_heading_is_constant_by_definition(self):
        # a rhumb bearing does not depend on how far along the line you are
        full = rhumb_bearing(10, 10, 40, 60)
        assert rhumb_bearing(10, 10, 40, 60) == pytest.approx(full)


class TestRefusals:
    def test_a_latitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            rhumb_distance_km(91, 0, 0, 0)
