from __future__ import annotations

import pytest

from atlas.errors import Invalid, Missing
from atlas.zonalstats import centroid_value, masked_values, mean_error, raster_from, summarize

TRIANGLE = [(0.3, 0.2), (9.1, 1.4), (4.7, 8.6)]
SQUARE = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)]
TILTED = [(2.0, 1.0), (8.0, 3.0), (6.0, 8.0), (1.0, 6.0)]


def plane(x: float, y: float) -> float:
    return 3.0 + 0.5 * x - 0.2 * y


def steep(x: float, _y: float) -> float:
    return 100.0 * x


def constant(_x: float, _y: float) -> float:
    return 7.0


class TestTheCentroidIdentity:
    def test_the_center_rule_converges_on_the_centroid_value(self):
        assert centroid_value(plane, TRIANGLE) == pytest.approx(4.67, abs=1e-3)
        errors = [mean_error(plane, TRIANGLE, cells, 10.0) for cells in (10, 20, 40, 80, 160)]
        assert errors[0] == pytest.approx(0.0376, abs=1e-3)
        assert errors[-1] == pytest.approx(0.00009, abs=1e-4)
        assert abs(errors[-1]) < abs(errors[0]) / 100

    def test_a_constant_raster_is_exact(self):
        raster = raster_from(constant, 20, 20, 0.5)
        stats = summarize(raster, [TRIANGLE], 0.5)
        assert stats["mean"] == 7.0
        assert stats["count"] == 137
        assert stats["std"] == 0.0


class TestTheMaskRule:
    def test_a_tilted_zone_brackets_the_centroid_by_a_quarter_percent(self):
        raster = raster_from(steep, 40, 40, 0.25)
        means = {rule: summarize(raster, [TILTED], 0.25, rule)["mean"] for rule in RULES}
        assert centroid_value(steep, TILTED) == pytest.approx(430.055, abs=1e-2)
        assert means["touch"] == pytest.approx(431.138, abs=1e-2)
        assert means["inside"] == pytest.approx(429.049, abs=1e-2)
        assert means["inside"] < centroid_value(steep, TILTED) < means["touch"]

    def test_an_axis_aligned_square_shows_the_grid_phase_not_the_rule(self):
        readings = ((8, 437.5, 375.0), (16, 406.25, 406.25), (32, 390.625, 406.25))
        for cells, center, both in readings:
            size = 10.0 / cells
            raster = raster_from(steep, cells, cells, size)
            assert summarize(raster, [SQUARE], size)["mean"] == pytest.approx(center)
            assert summarize(raster, [SQUARE], size, "touch")["mean"] == pytest.approx(both)
            assert summarize(raster, [SQUARE], size, "inside")["mean"] == pytest.approx(both)
        assert centroid_value(steep, SQUARE) == 400.0


class TestTheSummary:
    def test_min_max_count_and_std_agree_with_a_scan(self):
        raster = raster_from(plane, 40, 40, 0.25)
        values = masked_values(raster, [TRIANGLE], 0.25)
        stats = summarize(raster, [TRIANGLE], 0.25)
        assert stats["min"] == min(values)
        assert stats["max"] == max(values)
        assert stats["count"] == len(values) == 549
        assert stats["std"] == pytest.approx(0.9277, abs=1e-3)
        assert stats["sum"] == pytest.approx(sum(values))

    def test_refusals(self):
        raster = raster_from(plane, 40, 40, 0.25)
        with pytest.raises(Missing):
            summarize(raster, [[(20, 20), (21, 20), (21, 21)]], 0.25)
        with pytest.raises(Invalid):
            summarize(raster, [TRIANGLE], 0.25, "corner")
        with pytest.raises(Invalid):
            summarize([], [TRIANGLE], 0.25)


RULES = ("center", "touch", "inside")
