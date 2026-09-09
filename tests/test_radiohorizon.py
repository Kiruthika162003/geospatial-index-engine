from __future__ import annotations

import math

import pytest

from atlas.errors import Invalid
from atlas.radiohorizon import (
    clearance_m,
    earth_bulge_m,
    exact_geometric_horizon_km,
    flat_profile,
    fresnel_radius_m,
    horizon_km,
    link_range_km,
    max_range_over_flat,
    ray_height_m,
    ridge_profile,
    rule_of_thumb_km,
)

K = 4 / 3


class TestHorizons:
    @pytest.mark.parametrize(
        ("height", "chord", "bent"),
        [
            (1.5, 4.372, 5.048),
            (10, 11.288, 13.034),
            (100, 35.696, 41.218),
            (1000, 112.885, 130.347),
        ],
    )
    def test_the_rule_of_thumb_and_the_four_thirds_earth(self, height, chord, bent):
        assert horizon_km(height) == pytest.approx(chord, abs=1e-3)
        assert horizon_km(height, K) == pytest.approx(bent, abs=1e-3)
        gain = horizon_km(height, K) / horizon_km(height)
        assert gain == pytest.approx(math.sqrt(K), abs=1e-4)
        assert rule_of_thumb_km(height) == pytest.approx(chord, abs=5e-3)
        assert abs(exact_geometric_horizon_km(height) - chord) < 0.02

    def test_the_coefficients(self):
        assert rule_of_thumb_km(1.0) == pytest.approx(3.5696, abs=1e-4)
        assert rule_of_thumb_km(1.0, K) == pytest.approx(4.1218, abs=1e-4)

    @pytest.mark.parametrize(
        ("tx", "rx", "straight", "bent"), [(10, 10, 22.576, 26.069), (100, 1.5, 40.068, 46.266)]
    )
    def test_link_ranges_and_the_flat_search(self, tx, rx, straight, bent):
        assert link_range_km(tx, rx) == pytest.approx(straight, abs=1e-3)
        assert link_range_km(tx, rx, K) == pytest.approx(bent, abs=1e-3)
        assert abs(max_range_over_flat(tx, rx) - straight) <= 0.1
        assert abs(max_range_over_flat(tx, rx, K) - bent) <= 0.1


class TestClearance:
    def test_the_bulge(self):
        assert earth_bulge_m(10, 10) == pytest.approx(7.848, abs=1e-3)
        assert earth_bulge_m(10, 10, K) == pytest.approx(5.886, abs=1e-3)
        assert earth_bulge_m(25, 25) == pytest.approx(49.05, abs=1e-2)
        assert ray_height_m(30, 30, 20, 10) == pytest.approx(22.152, abs=1e-3)
        assert ray_height_m(30, 30, 20, 10, K) == pytest.approx(24.114, abs=1e-3)

    @pytest.mark.parametrize(
        ("ridge", "at", "straight", "bent", "fresnel"),
        [
            (20.0, 10.0, 2.152, 4.114, 24.999),
            (30.0, 10.0, -7.848, -5.886, 24.999),
            (20.0, 5.0, 4.114, 5.585, 21.65),
        ],
    )
    def test_a_clear_ray_sits_inside_the_fresnel_zone(self, ridge, at, straight, bent, fresnel):
        profile = ridge_profile(20.0, at, ridge, 201)
        assert clearance_m(30, 30, 20.0, profile) == pytest.approx(straight, abs=1e-3)
        assert clearance_m(30, 30, 20.0, profile, K) == pytest.approx(bent, abs=1e-3)
        assert fresnel_radius_m(2.4, at, 20.0 - at) == pytest.approx(fresnel, abs=1e-3)
        assert fresnel_radius_m(0.9, at, 20.0 - at) > fresnel
        assert bent < fresnel


class TestRefusals:
    def test_bad_heights_factors_paths_and_frequencies(self):
        with pytest.raises(Invalid):
            horizon_km(-1)
        with pytest.raises(Invalid):
            horizon_km(1, 0)
        with pytest.raises(Invalid):
            earth_bulge_m(-1, 1)
        with pytest.raises(Invalid):
            ray_height_m(1, 1, 10, 11)
        with pytest.raises(Invalid):
            clearance_m(1, 1, 10, [])
        with pytest.raises(Invalid):
            fresnel_radius_m(0, 1, 1)
        with pytest.raises(Invalid):
            flat_profile(10, 1)
