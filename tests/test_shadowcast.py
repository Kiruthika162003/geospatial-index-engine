from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.shadowcast import (
    cone,
    count,
    flat_with_tower,
    footprint,
    ridge,
    shadow_length,
    shadowed,
)


class TestTheTower:
    @pytest.mark.parametrize(
        ("elevation", "cells", "geometric"), [(30, 13, 13.86), (45, 7, 8.0), (60, 4, 4.62)]
    )
    def test_the_shadow_column_matches_the_geometry(self, elevation, cells, geometric):
        mask = shadowed(flat_with_tower(41, 8.0), 180.0, elevation)
        column = [r for r in range(41) if mask[r][20]]
        assert len(column) == cells == count(mask)
        assert max(column) == 19
        assert shadow_length(8.0, elevation) == pytest.approx(geometric, abs=0.01)
        assert abs(cells - geometric) <= 1.0

    def test_a_low_sun_runs_the_shadow_off_the_grid(self):
        mask = shadowed(flat_with_tower(41, 8.0), 180.0, 15)
        assert count(mask) == 20
        assert shadow_length(8.0, 15) == pytest.approx(29.86, abs=0.01)


class TestTheCone:
    @pytest.mark.parametrize(
        ("elevation", "fraction"), [(10, 0.322), (15, 0.223), (19, 0.096), (21, 0.0), (45, 0.0)]
    )
    def test_self_shadow_falls_to_zero_at_the_slope_angle(self, elevation, fraction):
        grid = cone(41, 6.552, 18.0)
        cells = footprint(grid)
        mask = shadowed(grid, 315.0, elevation)
        dark = sum(1 for r, c in cells if mask[r][c])
        assert dark / len(cells) == pytest.approx(fraction, abs=2e-3)
        if elevation > 20:
            assert count(mask) == 0


class TestTheRidgeAndTheStep:
    def test_the_ridge_shadows_a_strip_to_its_north_only(self):
        mask = shadowed(ridge(41, 5.0, 20), 180.0, 30.0)
        north = [r for r in range(20) if mask[r][20]]
        assert (min(north), max(north), len(north)) == (12, 19, 8)
        assert shadow_length(5.0, 30.0) == pytest.approx(8.66, abs=0.01)
        assert not any(mask[r][20] for r in range(21, 41))

    def test_a_coarse_step_loses_a_sixth_of_a_walls_shadow(self):
        wall = ridge(41, 5.0, 20)
        steps = (1.0, 0.5, 0.25, 0.1)
        counts = {step: count(shadowed(wall, 217.0, 30.0, step)) for step in steps}
        assert counts[0.5] == counts[0.25] == counts[0.1] == 230
        assert counts[1.0] == 192
        assert 1 - counts[1.0] / counts[0.25] == pytest.approx(0.165, abs=0.01)


class TestRefusals:
    def test_bad_suns_steps_and_grids_are_refused(self):
        grid = ridge(5, 1.0, 2)
        with pytest.raises(Invalid):
            shadowed(grid, 180, 0)
        with pytest.raises(Invalid):
            shadowed(grid, 180, 30, 0)
        with pytest.raises(Invalid):
            shadowed([], 180, 30)
        with pytest.raises(Invalid):
            shadow_length(1, 91)
