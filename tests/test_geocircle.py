from __future__ import annotations

import math

import pytest

from atlas.errors import Invalid, Outside
from atlas.geocircle import (
    cap_area_km2,
    flat_area_km2,
    polygon_area_km2,
    rim,
    rim_is_exact,
)
from atlas.haversine import EARTH_RADIUS_KM


class TestRim:
    def test_every_rim_point_sits_at_exactly_the_radius(self):
        assert rim_is_exact(40, 10, 1000) < 1e-9
        assert len(rim(40, 10, 1000, 72)) == 72


class TestCapVersusFlat:
    @pytest.mark.parametrize(
        ("radius", "ratio"),
        [
            (1, 1.0),
            (10, 0.99999979),
            (100, 0.99997947),
            (1000, 0.99794862),
            (3000, 0.98165846),
            (5000, 0.94971567),
        ],
    )
    def test_the_cap_falls_below_the_flat_disk_as_the_radius_grows(self, radius, ratio):
        assert cap_area_km2(radius) / flat_area_km2(radius) == pytest.approx(ratio, abs=1e-6)

    def test_a_hemisphere_is_eight_over_pi_squared_of_the_flat_disk(self):
        # the refuted guess said sixty-one percent; it is 81.06 percent exactly
        quarter = math.pi * EARTH_RADIUS_KM / 2
        assert cap_area_km2(quarter) / flat_area_km2(quarter) == pytest.approx(8 / math.pi**2)
        assert cap_area_km2(quarter) == pytest.approx(2 * math.pi * EARTH_RADIUS_KM**2)


class TestPolygonApproachesTheCap:
    def test_from_below_as_the_rim_resolution_rises(self):
        cap = cap_area_km2(1000)
        ratios = [polygon_area_km2(40, 10, 1000, pts) / cap for pts in (6, 12, 72, 360)]
        assert ratios == pytest.approx([0.82689, 0.95493, 0.99873, 0.99995], abs=2e-4)
        assert ratios == sorted(ratios)
        assert all(r <= 1.0 for r in ratios)


class TestRefusals:
    def test_a_non_positive_radius_is_refused(self):
        with pytest.raises(Invalid):
            rim(0, 0, 0)
        with pytest.raises(Invalid):
            cap_area_km2(-1)

    def test_a_radius_past_half_the_circumference_is_refused(self):
        with pytest.raises(Invalid):
            cap_area_km2(math.pi * EARTH_RADIUS_KM + 1)

    def test_too_few_rim_points_is_refused(self):
        with pytest.raises(Invalid):
            rim(0, 0, 10, points=2)

    def test_a_center_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            rim(91, 0, 10)
