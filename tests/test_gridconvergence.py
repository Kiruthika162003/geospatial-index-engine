from __future__ import annotations

import pytest

from atlas.errors import Invalid, Outside
from atlas.gridconvergence import (
    convergence_law,
    convergence_measured,
    forward,
    offset_at_distance,
    scale_law,
    scale_measured,
    small_angle_law,
    zone_extremes,
)


class TestConvergence:
    @pytest.mark.parametrize(
        ("lat", "readings"),
        [
            (0.0, (0.0, 0.0, 0.0)),
            (30.0, (0.25, 0.50004, 1.50103)),
            (45.0, (0.35356, 0.70714, 2.12229)),
            (60.0, (0.43302, 0.86605, 2.59867)),
            (80.0, (0.4924, 0.98481, 2.9545)),
        ],
    )
    def test_the_measured_bearing_meets_the_law(self, lat, readings):
        for dlon, reading in zip((0.5, 1.0, 3.0), readings, strict=True):
            assert convergence_measured(lat, dlon, 0.0) == pytest.approx(reading, abs=1e-5)
            assert convergence_law(lat, dlon, 0.0) == pytest.approx(reading, abs=1e-5)
            assert small_angle_law(lat, dlon, 0.0) == pytest.approx(reading, rel=1e-3)
        assert convergence_measured(lat, -3.0, 0.0) == pytest.approx(-readings[2], abs=1e-5)

    def test_the_small_angle_rule_at_ten_degrees(self):
        assert convergence_law(45.0, 10.0, 0.0) == pytest.approx(7.1071, abs=1e-4)
        assert small_angle_law(45.0, 10.0, 0.0) == pytest.approx(7.0711, abs=1e-4)
        assert convergence_measured(45.0, 10.0, 0.0) == pytest.approx(7.1071, abs=1e-4)

    @pytest.mark.parametrize(
        ("lat", "miss"), [(0.0, 0.0), (30.0, 261.95), (45.0, 370.32), (80.0, 515.43)]
    )
    def test_walking_grid_north(self, lat, miss):
        assert offset_at_distance(convergence_law(lat, 3.0, 0.0), 10000.0) == pytest.approx(
            miss, abs=0.01
        )


class TestScale:
    @pytest.mark.parametrize(
        ("lat", "scale"),
        [(0.0, 1.000972), (45.0, 1.000285), (60.0, 0.999942), (80.0, 0.999641)],
    )
    def test_the_zone_edge_scale(self, lat, scale):
        assert scale_measured(lat, 3.0, 0.0) == pytest.approx(scale, abs=1e-6)
        assert scale_law(lat, 3.0, 0.0) == pytest.approx(scale, abs=1e-6)
        assert scale_law(lat, 0.0, 0.0) == pytest.approx(0.9996)
        assert zone_extremes(lat)[1] == pytest.approx(scale, abs=1e-6)

    def test_refusals(self):
        with pytest.raises(Outside):
            forward(91.0, 0.0, 0.0)
        with pytest.raises(Invalid):
            forward(0.0, 90.0, 0.0)
        with pytest.raises(Invalid):
            zone_extremes(45.0, 0.0)
