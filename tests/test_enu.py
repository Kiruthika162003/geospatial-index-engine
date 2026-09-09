from __future__ import annotations

import random

import pytest

from atlas.bearing import destination
from atlas.ecef import chord_m
from atlas.enu import (
    cross,
    curvature_drop_m,
    enu_to_geodetic,
    frame_axes,
    from_enu,
    geodetic_to_enu,
    to_enu,
)

REF = (40.0, -74.0)


class TestTheFrame:
    def test_the_reference_is_the_origin(self):
        assert geodetic_to_enu(40, -74, 0, *REF) == (0.0, 0.0, 0.0)

    def test_the_frame_is_right_handed(self):
        e, n, u = frame_axes(*REF)
        assert cross(e, n) == pytest.approx(u, abs=1e-12)
        assert cross(n, u) == pytest.approx(e, abs=1e-12)


class TestTheCurvatureDrop:
    @pytest.mark.parametrize("km", [0.1, 1, 10, 100])
    def test_surface_points_drop_below_the_plane_as_the_square_of_range(self, km):
        east = geodetic_to_enu(*destination(*REF, 90.0, km), 0.0, *REF)
        north = geodetic_to_enu(*destination(*REF, 0.0, km), 0.0, *REF)
        law = curvature_drop_m(km * 1000)
        assert -east[2] / law == pytest.approx(1.0036, abs=2e-4)
        assert -north[2] / law == pytest.approx(0.9997, abs=2e-4)

    def test_the_law_itself_slips_at_a_thousand_kilometers(self):
        east = geodetic_to_enu(*destination(*REF, 90.0, 1000), 0.0, *REF)
        assert -east[2] / curvature_drop_m(1e6) == pytest.approx(1.0015, abs=2e-4)

    def test_the_named_drops(self):
        assert curvature_drop_m(1000) == pytest.approx(0.0784, abs=1e-4)
        assert curvature_drop_m(10000) == pytest.approx(7.839, abs=1e-3)
        assert curvature_drop_m(100000) == pytest.approx(783.93, abs=0.01)

    def test_a_kilometer_east_on_the_plane_sits_eight_centimeters_up(self):
        lat, lon, h = enu_to_geodetic((1000.0, 0.0, 0.0), *REF, 0.0)
        assert h == pytest.approx(0.0783, abs=1e-4)
        assert lat == pytest.approx(40.0, abs=1e-5)
        assert lon == pytest.approx(-73.98829, abs=1e-5)


class TestRoundTrip:
    def test_enu_to_ecef_and_back_is_a_rigid_motion(self):
        rng = random.Random(188)
        for _ in range(1000):
            rl, ro, rh = rng.uniform(-89, 89), rng.uniform(-180, 180), rng.uniform(0, 3000)
            enu = (rng.uniform(-5e4, 5e4), rng.uniform(-5e4, 5e4), rng.uniform(-1000, 1000))
            assert chord_m(enu, to_enu(from_enu(enu, rl, ro, rh), rl, ro, rh)) < 1e-6
            lat, lon, h = enu_to_geodetic(enu, rl, ro, rh)
            assert chord_m(enu, geodetic_to_enu(lat, lon, h, rl, ro, rh)) < 1e-5
