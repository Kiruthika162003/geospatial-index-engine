from __future__ import annotations

import random

import pytest

from atlas.bearing import destination
from atlas.errors import Invalid, Outside
from atlas.gnomonic import (
    angular_distance_deg,
    forward,
    geodesic_deviation_km,
    inverse,
    map_line_deviation_km,
    radial_scale,
)
from atlas.haversine import haversine

NEW_YORK = (40.0, -74.0)
LONDON = (51.5, -0.1)


class TestStraightness:
    def test_a_long_geodesic_projects_to_a_straight_line(self):
        assert geodesic_deviation_km(NEW_YORK, LONDON, 50, -40) < 1e-9
        assert geodesic_deviation_km((0, 0), (10, 10), 5, 5) < 1e-9

    def test_the_map_line_is_not_straight_because_it_is_not_a_geodesic(self):
        assert map_line_deviation_km(NEW_YORK, LONDON, 50, -40) == pytest.approx(755.0, abs=0.1)
        assert map_line_deviation_km((0, 0), (10, 10), 5, 5) == pytest.approx(4.621, abs=1e-3)

    def test_the_equator_is_straight_both_ways(self):
        assert geodesic_deviation_km((0, 0), (0, 10), 0, 5) == 0.0
        assert map_line_deviation_km((0, 0), (0, 10), 0, 5) == 0.0

    def test_random_geodesics_across_the_visible_disc_stay_straight(self):
        rng = random.Random(177)
        for _ in range(200):
            lat0, lon0 = rng.uniform(-60, 60), rng.uniform(-150, 150)
            a = destination(lat0, lon0, rng.uniform(0, 360), rng.uniform(100, 6000))
            b = destination(lat0, lon0, rng.uniform(0, 360), rng.uniform(100, 6000))
            if haversine(*a, *b) < 100:
                continue
            assert geodesic_deviation_km(a, b, lat0, lon0) < 1e-8


class TestRoundTrip:
    def test_forward_then_inverse_returns_home_across_the_hemisphere(self):
        rng = random.Random(178)
        for _ in range(1000):
            lat0, lon0 = rng.uniform(-80, 80), rng.uniform(-180, 180)
            lat, lon = rng.uniform(-90, 90), rng.uniform(-180, 180)
            if angular_distance_deg(lat, lon, lat0, lon0) >= 85:
                continue
            back = inverse(*forward(lat, lon, lat0, lon0), lat0, lon0)
            assert haversine(lat, lon, *back) < 1e-8  # measured worst 2.9e-10 km

    def test_the_tangent_point_is_the_origin(self):
        assert forward(45, 10, 45, 10) == (0.0, 0.0)
        assert inverse(0, 0, 30, 40) == (30, 40)

    def test_longitude_wraps_on_the_way_back(self):
        assert inverse(*forward(10, -175, 0, 178), 0, 178) == pytest.approx((10.0, -175.0))


class TestScale:
    def test_the_radial_scale_follows_one_over_cos_squared(self):
        assert [radial_scale(a) for a in (0, 45, 60)] == pytest.approx([1.0, 2.0, 4.0])
        assert radial_scale(80) == pytest.approx(33.163, abs=1e-3)
        assert radial_scale(89) > 3000

    def test_the_measured_step_matches_the_formula(self):
        for angle in (0, 45, 60, 80):
            x1, _ = forward(0, angle, 0, 0)
            x2, _ = forward(0, angle + 0.001, 0, 0)
            measured = (x2 - x1) / (0.001 * 3.141592653589793 / 180)
            assert measured == pytest.approx(radial_scale(angle), rel=1e-3)


class TestRefusals:
    def test_the_horizon_and_the_far_hemisphere_are_refused(self):
        for p in ((0, 90), (0, 91), (-45, 180)):
            with pytest.raises(Outside):
                forward(*p, 0, 0)

    def test_an_angle_past_the_horizon_has_no_scale(self):
        with pytest.raises(Invalid):
            radial_scale(90)
