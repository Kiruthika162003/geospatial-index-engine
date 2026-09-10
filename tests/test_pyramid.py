from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.pyramid import (
    build,
    cells,
    cone,
    level_for_pixels,
    mean_at_levels,
    overhead,
    overhead_law,
    peak_at_levels,
    read_cost,
    reduce_level,
    spike,
)


class TestMemory:
    @pytest.mark.parametrize(
        ("size", "levels", "count"), [(64, 7, 5461), (256, 9, 87381), (1024, 11, 1398101)]
    )
    def test_a_third_more_at_every_size(self, size, levels, count):
        pyramid = build(cone(size, 100.0, size / 3))
        assert len(pyramid) == levels
        assert cells(pyramid) == count
        assert overhead(pyramid) == pytest.approx(overhead_law(), abs=1e-4)

    @pytest.mark.parametrize(
        ("pixels", "level", "read"), [(512, 1, 262144), (100, 4, 4096), (10, 7, 64)]
    )
    def test_reading_the_matching_level(self, pixels, level, read):
        assert level_for_pixels(1024, pixels) == level
        assert read_cost(1024, pixels) == (1048576, read)


class TestPeaks:
    def test_a_spike_under_the_three_rules(self):
        grid = spike(64, 100.0)
        means = peak_at_levels(build(grid, "mean"))
        assert means == pytest.approx(
            [100.0, 25.0, 6.25, 1.5625, 0.390625, 0.09765625, 0.0244140625]
        )
        assert all(
            m == pytest.approx(0.0244140625) for m in mean_at_levels(build(grid, "mean"))
        )
        assert peak_at_levels(build(grid, "max")) == [100.0] * 7
        assert mean_at_levels(build(grid, "max"))[-1] == 100.0
        nearest = peak_at_levels(build(grid, "nearest"))
        assert nearest[:6] == [100.0] * 6 and nearest[6] == 0.0

    def test_a_cone(self):
        grid = cone(256, 100.0, 40.0)
        means = [round(v, 2) for v in peak_at_levels(build(grid, "mean"))]
        assert means == [98.23, 96.26, 92.39, 84.72, 69.4, 38.91, 10.23, 2.56, 2.56]
        assert peak_at_levels(build(grid, "max")) == pytest.approx([98.23] * 9, abs=0.01)
        assert mean_at_levels(build(grid, "max"))[4] == pytest.approx(4.8843, abs=1e-4)


class TestRefusals:
    def test_bad_grids_rules_and_pixels(self):
        with pytest.raises(Invalid):
            reduce_level([], "mean")
        with pytest.raises(Invalid):
            reduce_level([[1.0, 2.0], [3.0, 4.0]], "median")
        with pytest.raises(Invalid):
            reduce_level([[1.0]], "mean")
        with pytest.raises(Invalid):
            level_for_pixels(0, 10)
