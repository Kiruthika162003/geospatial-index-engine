from __future__ import annotations

import pytest

from atlas.cutfill import (
    balance_height,
    by_cells,
    by_prisms,
    cone,
    cone_volume,
    flat,
    mean_height,
    paraboloid,
    paraboloid_volume,
    sample_centres,
    sample_corners,
    tilted,
)
from atlas.errors import Invalid

SIZE = 120.0
HILL = cone(60.0, 60.0, 50.0, 30.0)
BOWL = paraboloid(60.0, 60.0, 50.0, 30.0)


def _errors(field, truth, cell):
    base_c, base_k = sample_centres(flat(), SIZE, cell), sample_corners(flat(), SIZE, cell)
    centres = by_cells(base_c, sample_centres(field, SIZE, cell), cell)
    corners = by_prisms(base_k, sample_corners(field, SIZE, cell), cell)
    assert centres[0] == 0.0 and corners[0] == 0.0
    return 100 * (centres[1] / truth - 1), 100 * (corners[1] / truth - 1)


class TestVolumes:
    @pytest.mark.parametrize(
        ("cell", "cells", "prisms"),
        [
            (1.0, 0.001, -0.001),
            (4.0, -0.016, 0.044),
            (8.0, 0.016, 0.187),
            (12.0, 0.452, -0.315),
        ],
    )
    def test_the_cone_by_cells_and_by_prisms(self, cell, cells, prisms):
        assert cone_volume(50.0, 30.0) == pytest.approx(78539.816, abs=1e-3)
        read = _errors(HILL, cone_volume(50.0, 30.0), cell)
        assert read == pytest.approx((cells, prisms), abs=1e-3)
        if 2.0 <= cell <= 8.0:
            assert abs(read[0]) <= abs(read[1])

    @pytest.mark.parametrize(
        ("cell", "cells", "prisms"),
        [(2.0, 0.006, -0.011), (8.0, -0.095, 0.278), (12.0, 0.703, -0.817)],
    )
    def test_the_paraboloid(self, cell, cells, prisms):
        assert paraboloid_volume(50.0, 30.0) == pytest.approx(117809.725, abs=1e-3)
        read = _errors(BOWL, paraboloid_volume(50.0, 30.0), cell)
        assert read == pytest.approx((cells, prisms), abs=1e-3)

    def test_the_grids_phase(self):
        readings = []
        for cx in (60.0, 64.0, 62.0):
            field = cone(cx, 60.0, 50.0, 30.0)
            readings.append(round(_errors(field, cone_volume(50.0, 30.0), 8.0)[0], 3))
        assert readings == [0.016, -0.014, -0.075]


class TestBalance:
    @pytest.mark.parametrize(
        ("slope", "height", "moved"), [(0.1, 6.0, 21600.0), (0.5, 30.0, 108000.0)]
    )
    def test_the_balancing_platform_sits_at_the_mean(self, slope, height, moved):
        plane = sample_centres(tilted(slope), SIZE, 2.0)
        found = balance_height(plane, 2.0, 0.0, SIZE * slope)
        assert found == pytest.approx(height, abs=1e-6)
        assert mean_height(plane) == pytest.approx(height, abs=1e-9)
        level = [[found] * len(plane[0]) for _ in plane]
        cut, fill = by_cells(plane, level, 2.0)
        assert cut == pytest.approx(moved, abs=1e-3)
        assert fill == pytest.approx(moved, abs=1e-3)

    def test_the_cone(self):
        grid = sample_centres(HILL, SIZE, 4.0)
        found = balance_height(grid, 4.0, 0.0, 30.0)
        assert found == pytest.approx(mean_height(grid), abs=1e-6)
        assert mean_height(grid) == pytest.approx(5.453298, abs=1e-6)


class TestRefusals:
    def test_bad_grids_cells_and_searches(self):
        with pytest.raises(Invalid):
            by_cells([], [], 1.0)
        with pytest.raises(Invalid):
            by_cells([[1.0]], [[1.0, 2.0]], 1.0)
        with pytest.raises(Invalid):
            by_cells([[1.0]], [[1.0]], 0.0)
        with pytest.raises(Invalid):
            by_prisms([[1.0]], [[1.0]], 1.0)
        with pytest.raises(Invalid):
            sample_centres(flat(), 1.0, 5.0)
        with pytest.raises(Invalid):
            balance_height([[1.0]], 1.0, 2.0, 1.0)
