from __future__ import annotations

import random

import pytest

from atlas.bearing import destination, final_bearing, initial_bearing
from atlas.errors import Outside
from atlas.haversine import haversine


class TestInitialBearing:
    def test_cardinal_directions(self):
        assert initial_bearing(0, 0, 1, 0) == pytest.approx(0.0)  # north
        assert initial_bearing(0, 0, 0, 1) == pytest.approx(90.0)  # east
        assert initial_bearing(1, 0, 0, 0) == pytest.approx(180.0)  # south
        assert initial_bearing(0, 1, 0, 0) == pytest.approx(270.0)  # west


class TestHeadingChangesEnRoute:
    def test_the_equator_route_holds_a_constant_heading(self):
        ib = initial_bearing(0, -50, 0, 50)
        fb = final_bearing(0, -50, 0, 50)
        assert ib == pytest.approx(90.0)
        assert fb == pytest.approx(90.0)

    def test_the_bearing_gap_grows_with_latitude(self):
        def gap(lat, dlon):
            ib = initial_bearing(lat, -dlon / 2, lat, dlon / 2)
            fb = final_bearing(lat, -dlon / 2, lat, dlon / 2)
            return (fb - ib) % 360

        g30 = gap(30, 40)
        g45 = gap(45, 60)
        g60 = gap(60, 80)
        assert g30 == pytest.approx(20.63, abs=0.1)
        assert g45 == pytest.approx(44.42, abs=0.1)
        assert g60 == pytest.approx(72.01, abs=0.1)
        assert g30 < g45 < g60


class TestDestination:
    def test_distance_round_trips_exactly(self):
        rng = random.Random(31)
        worst = 0.0
        for _ in range(20000):
            la, lo = rng.uniform(-70, 70), rng.uniform(-170, 170)
            b, d = rng.uniform(0, 360), rng.uniform(1, 5000)
            la2, lo2 = destination(la, lo, b, d)
            worst = max(worst, abs(haversine(la, lo, la2, lo2) - d))
        assert worst < 1e-6  # kilometers

    def test_going_north_raises_latitude(self):
        la2, lo2 = destination(0, 0, 0, 111.19)  # ~1 degree north
        assert la2 == pytest.approx(1.0, abs=1e-3)
        assert lo2 == pytest.approx(0.0, abs=1e-6)


class TestRefusals:
    def test_a_latitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            initial_bearing(91, 0, 0, 0)

    def test_a_negative_distance_is_refused(self):
        with pytest.raises(Outside):
            destination(0, 0, 90, -5)
