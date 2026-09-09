from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.transversemercator import (
    RADIUS_M,
    convergence,
    forward,
    inverse,
    measured_convergence,
    measured_scale,
    point_scale,
    unit_scale_offset_km,
)

LON0 = 9.0


class TestAgainstClosedForms:
    @pytest.mark.parametrize("lat", [0, 45, 60])
    @pytest.mark.parametrize("offset", [0.5, 1.0, 2.0, 3.0])
    def test_scale_and_convergence_match_by_finite_difference(self, lat, offset):
        lon = LON0 + offset
        assert abs(point_scale(lat, lon, LON0) - measured_scale(lat, lon, LON0)) < 1e-8
        assert abs(convergence(lat, lon, LON0) - measured_convergence(lat, lon, LON0)) < 1e-6

    def test_the_central_meridian_reads_the_central_scale(self):
        assert point_scale(45, LON0, LON0) == 0.9996
        assert measured_scale(45, LON0, LON0) == pytest.approx(0.9996, abs=1e-9)
        assert convergence(45, LON0, LON0) == 0.0


class TestTheZone:
    def test_the_scale_crosses_one_at_180_km_at_every_latitude(self):
        for lat in (0, 45, 60):
            assert unit_scale_offset_km(lat, LON0) == pytest.approx(180.2, abs=0.1)

    def test_edge_scale_and_convergence(self):
        assert point_scale(0, LON0 + 3, LON0) == pytest.approx(1.000972, abs=1e-6)
        assert point_scale(45, LON0 + 3, LON0) == pytest.approx(1.000285, abs=1e-6)
        assert convergence(45, LON0 + 3, LON0) == pytest.approx(2.122, abs=1e-3)
        assert convergence(60, LON0 + 3, LON0) == pytest.approx(2.599, abs=1e-3)
        assert convergence(80, LON0 + 3, LON0) == pytest.approx(2.955, abs=1e-3)
        assert convergence(0, LON0 + 3, LON0) == 0.0
        assert forward(45, LON0 + 3, LON0)[0] == pytest.approx(235786.0, abs=1.0)

    def test_a_single_cylinder_cannot_serve_a_continent(self):
        assert point_scale(45, LON0 + 6, LON0) == pytest.approx(1.0023, abs=1e-4)
        assert point_scale(45, LON0 + 30, LON0) == pytest.approx(1.0686, abs=1e-4)
        assert point_scale(0, LON0 + 30, LON0) == pytest.approx(1.1543, abs=1e-4)


class TestRoundTrip:
    def test_inverse_returns_within_nanometers(self):
        rng = random.Random(247)
        for _ in range(500):
            lat, lon = rng.uniform(-80, 80), LON0 + rng.uniform(-3, 3)
            back_lat, back_lon = inverse(*forward(lat, lon, LON0), LON0)
            meters_per_degree = math.radians(1) * RADIUS_M
            north = abs(back_lat - lat) * meters_per_degree
            east = abs(back_lon - lon) * meters_per_degree * math.cos(math.radians(lat))
            assert max(north, east) < 1e-6


class TestRefusals:
    def test_the_poles_and_the_far_meridian_are_refused(self):
        with pytest.raises(Outside):
            forward(90, 0, 0)
        with pytest.raises(Invalid):
            forward(0, 90, 0)
