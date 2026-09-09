from __future__ import annotations

import math
import random

import pytest

from atlas.azimuthalequidistant import (
    bearing_error_deg,
    chord_ratio,
    forward,
    inverse,
    radial_error_km,
    transverse_scale,
)
from atlas.bearing import destination
from atlas.errors import Invalid, Outside
from atlas.haversine import haversine

CENTER = (30.0, 20.0)
QUARTER_GLOBE_KM = 10007.557


class TestWhatIsKept:
    def test_distance_and_bearing_from_the_center_are_true_over_the_whole_globe(self):
        rng = random.Random(191)
        far = 0
        for _ in range(2000):
            lat0, lon0 = rng.uniform(-80, 80), rng.uniform(-180, 180)
            lat, lon = rng.uniform(-90, 90), rng.uniform(-180, 180)
            d = haversine(lat0, lon0, lat, lon)
            if d < 1 or d > 20000:
                continue
            far += d > 10007.5
            assert abs(radial_error_km(lat, lon, lat0, lon0)) < 1e-6
            assert abs(bearing_error_deg(lat, lon, lat0, lon0)) < 1e-9
            back = inverse(*forward(lat, lon, lat0, lon0), lat0, lon0)
            assert haversine(lat, lon, *back) < 1e-6
        assert far > 800  # half the points lie in the far hemisphere

    def test_the_center_is_the_origin_both_ways(self):
        assert forward(*CENTER, *CENTER) == (0.0, 0.0)
        assert inverse(0, 0, *CENTER) == CENTER


class TestWhatIsLost:
    def test_opposite_points_are_true_by_luck(self):
        a = destination(*CENTER, 0.0, QUARTER_GLOBE_KM)
        b = destination(*CENTER, 180.0, QUARTER_GLOBE_KM)
        assert chord_ratio(a, b, *CENTER) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        ("separation", "ratio"),
        [(30, 1.5529), (60, 1.5), (90, 1.4142), (120, 1.299), (150, 1.1591)],
    )
    def test_points_ninety_degrees_out_stretch_by_their_separation(self, separation, ratio):
        a = destination(*CENTER, 0.0, QUARTER_GLOBE_KM)
        b = destination(*CENTER, separation, QUARTER_GLOBE_KM)
        assert chord_ratio(a, b, *CENTER) == pytest.approx(ratio, abs=1e-3)

    @pytest.mark.parametrize(
        ("out", "ratio"),
        [(10, 1.0026), (30, 1.0246), (60, 1.1235), (90, 1.4142), (120, 2.2471), (150, 5.1228)],
    )
    def test_a_quarter_turn_stretches_more_the_farther_out(self, out, ratio):
        a = destination(*CENTER, 0.0, out * 111.195)
        b = destination(*CENTER, 90.0, out * 111.195)
        assert chord_ratio(a, b, *CENTER) == pytest.approx(ratio, abs=1e-3)

    def test_transverse_scale_is_c_over_sin_c(self):
        assert transverse_scale(0) == 1.0
        assert transverse_scale(90) == pytest.approx(math.pi / 2)
        assert transverse_scale(150) == pytest.approx(5.236, abs=1e-3)
        assert transverse_scale(179) == pytest.approx(179.009, abs=1e-2)

    def test_a_small_circle_round_the_antipode_becomes_the_rim(self):
        antipode = (-30.0, -160.0)
        ring = [destination(*antipode, brg, 100.0) for brg in range(0, 360, 10)]
        xs = [forward(*p, *CENTER) for p in ring]
        rim = sum(math.dist(xs[i], xs[i - 1]) for i in range(36))
        assert rim * 6371.0088 == pytest.approx(124971.6, abs=1.0)
        assert rim * 6371.0088 / (2 * math.pi * 100) == pytest.approx(198.9, abs=0.1)


class TestRefusals:
    def test_the_antipode_and_the_beyond_rim_are_refused(self):
        with pytest.raises(Invalid):
            forward(-30.0, -160.0, *CENTER)
        with pytest.raises(Outside):
            inverse(3.2, 0, *CENTER)
        with pytest.raises(Outside):
            forward(91, 0, *CENTER)
        with pytest.raises(Invalid):
            transverse_scale(180)
