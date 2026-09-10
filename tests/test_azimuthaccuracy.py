from __future__ import annotations

import math
import random

import pytest

from atlas.azimuthaccuracy import (
    bearing,
    bearing_error_law,
    dilution,
    fix_error,
    fix_error_law,
    intersect,
    measured_bearing_error,
    stations_at_angle,
)
from atlas.errors import Invalid


class TestBearings:
    @pytest.mark.parametrize(
        ("distance", "measured", "law"),
        [(100.0, 2.8549, 2.8648), (1000.0, 0.2852, 0.2865), (10000.0, 0.0285, 0.0286)],
    )
    def test_the_bearing_error_is_sigma_over_d(self, distance, measured, law):
        target = (distance / math.sqrt(2), distance / math.sqrt(2))
        assert measured_bearing_error(
            (0.0, 0.0), target, 5.0, random.Random(880)
        ) == pytest.approx(measured, abs=1e-4)
        assert bearing_error_law(5.0, distance) == pytest.approx(law, abs=1e-4)
        assert bearing((0.0, 0.0), (0.0, 1.0)) == 0.0
        assert bearing((0.0, 0.0), (1.0, 0.0)) == 90.0


class TestFixes:
    @pytest.mark.parametrize(
        ("angle", "miss", "law"),
        [
            (10.0, 157.59, 142.14),
            (30.0, 49.87, 49.37),
            (90.0, 24.64, 24.68),
            (150.0, 49.19, 49.37),
            (170.0, 144.85, 142.14),
        ],
    )
    def test_the_fix_follows_the_sine_law(self, angle, miss, law):
        stations = stations_at_angle((0.0, 0.0), 1000.0, angle)
        assert fix_error((0.0, 0.0), stations, 1.0, random.Random(881)) == pytest.approx(
            miss, abs=0.01
        )
        assert fix_error_law(1000.0, 1.0, angle) == pytest.approx(law, abs=0.01)
        assert dilution(angle) == pytest.approx(1 / math.sin(math.radians(angle)))
        if 30 <= angle <= 150:
            assert abs(miss / law - 1) < 0.011

    def test_the_miss_grows_with_the_sigma(self):
        stations = stations_at_angle((0.0, 0.0), 1000.0, 90.0)
        reads = [
            fix_error((0.0, 0.0), stations, s, random.Random(882)) for s in (0.1, 0.5, 1.0, 2.0)
        ]
        assert [round(r, 2) for r in reads] == [2.47, 12.34, 24.69, 49.44]
        assert intersect((0.0, 0.0), 45.0, (10.0, 0.0), 315.0) == pytest.approx((5.0, 5.0))

    def test_refusals(self):
        with pytest.raises(Invalid):
            bearing_error_law(1.0, 0.0)
        with pytest.raises(Invalid):
            intersect((0.0, 0.0), 45.0, (10.0, 0.0), 45.0)
        with pytest.raises(Invalid):
            fix_error_law(1000.0, 1.0, 180.0)
