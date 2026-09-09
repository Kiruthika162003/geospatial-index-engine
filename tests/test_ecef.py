from __future__ import annotations

import math
import random

import pytest

from atlas.ecef import (
    WGS84_A,
    WGS84_B,
    chord_m,
    forward,
    geocentric_latitude,
    inverse,
    prime_vertical_radius,
    surface_radius,
)
from atlas.errors import Invalid, Outside


class TestTheEllipsoid:
    def test_equator_and_pole_land_on_the_axes(self):
        assert forward(0, 0) == (WGS84_A, 0.0, 0.0)
        x, _, z = forward(90, 0)
        assert abs(x) < 1e-6
        assert z == pytest.approx(WGS84_B)
        assert pytest.approx(21384.686, abs=1e-3) == WGS84_A - WGS84_B

    def test_the_prime_vertical_radius_runs_from_a_to_a_squared_over_b(self):
        assert prime_vertical_radius(0) == WGS84_A
        assert prime_vertical_radius(90) == pytest.approx(WGS84_A**2 / WGS84_B)


class TestRoundTrip:
    def test_forward_then_inverse_returns_within_nanometers(self):
        rng = random.Random(187)
        worst_rounds = 0
        for _ in range(2000):
            lat, lon = rng.uniform(-90, 90), rng.uniform(-180, 180)
            h = rng.choice([0.0, rng.uniform(-500, 9000), rng.uniform(100e3, 20200e3)])
            x, y, z = forward(lat, lon, h)
            la, lo, hh, rounds = inverse(x, y, z)
            assert chord_m(forward(la, lo, hh), (x, y, z)) < 1e-6  # measured worst 1.2e-8 m
            worst_rounds = max(worst_rounds, rounds)
        assert worst_rounds <= 7

    def test_on_the_surface_the_first_guess_is_already_exact(self):
        rng = random.Random(189)
        for _ in range(300):
            x, y, z = forward(rng.uniform(-89, 89), rng.uniform(-180, 180), 0.0)
            assert inverse(x, y, z)[3] == 1

    def test_aloft_the_iteration_takes_a_few_rounds(self):
        rng = random.Random(190)
        rounds = set()
        for _ in range(300):
            x, y, z = forward(rng.uniform(-89, 89), rng.uniform(-180, 180), 20200e3)
            rounds.add(inverse(x, y, z)[3])
        assert min(rounds) >= 2
        assert max(rounds) <= 7

    def test_the_poles_are_handled_without_iterating(self):
        assert inverse(0, 0, WGS84_B) == (90.0, 0.0, 0.0, 0)
        assert inverse(0, 0, -WGS84_B - 1000) == (-90.0, 0.0, 1000.0, 0)


class TestTheSphericalGap:
    def test_geocentric_latitude_falls_short_by_a_fifth_of_a_degree(self):
        gaps = [(lat / 10 - geocentric_latitude(lat / 10), lat / 10) for lat in range(901)]
        worst, at = max(gaps)
        assert worst == pytest.approx(0.19242, abs=1e-5)
        assert at == 45.1
        assert math.radians(worst) * 6371.0088 == pytest.approx(21.40, abs=0.01)

    def test_the_surface_radius_drops_21_km_from_equator_to_pole(self):
        assert surface_radius(0) == pytest.approx(6378137.0)
        assert surface_radius(45) == pytest.approx(6367489.54, abs=0.01)
        assert surface_radius(90) == pytest.approx(6356752.31, abs=0.01)


class TestRefusals:
    def test_the_center_and_off_globe_inputs_are_refused(self):
        with pytest.raises(Invalid):
            inverse(0, 0, 0)
        with pytest.raises(Outside):
            forward(91, 0)
        with pytest.raises(Outside):
            forward(0, 181)
