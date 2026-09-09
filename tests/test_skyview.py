from __future__ import annotations

import math

import pytest

from atlas.errors import Invalid
from atlas.skyview import (
    canyon_guess,
    horizon_angle,
    mean_interior,
    pit,
    pit_law,
    plane,
    sky_view,
    sky_view_grid,
    trench,
    v_valley,
    valley_floor_law,
)

SIZE = 61
MID = 30


class TestValleys:
    def test_a_plane_sees_everything(self):
        assert sky_view(plane(SIZE), MID, MID) == 1.0
        assert horizon_angle(plane(SIZE), MID, MID, 0.0) == 0.0
        assert sky_view(plane(SIZE, 0.5), MID, MID) == pytest.approx(0.8472, abs=1e-4)

    @pytest.mark.parametrize(
        ("slope", "readings", "law"),
        [
            (0.25, (0.8523, 0.8378, 0.8297, 0.8274), 0.844),
            (0.5, (0.7215, 0.6944, 0.6791, 0.675), 0.7048),
            (1.0, (0.5345, 0.4884, 0.4626, 0.4567), 0.5),
            (2.0, (0.3681, 0.2938, 0.2571, 0.2493), 0.2952),
        ],
    )
    def test_the_fan_settles_under_the_law_and_far_under_the_cosine(self, slope, readings, law):
        dem = v_valley(SIZE, slope)
        read = tuple(sky_view(dem, MID, MID, d) for d in (8, 16, 32, 64))
        assert read == pytest.approx(readings, abs=1e-4)
        assert valley_floor_law(slope) == pytest.approx(law, abs=1e-4)
        assert read[-1] < law < canyon_guess(math.degrees(math.atan(slope)))

    def test_the_first_cell_is_the_horizon_and_the_wall_opens_up(self):
        dem = v_valley(SIZE, 1.0)
        for steps in (4, 8, 16, 32):
            assert sky_view(dem, MID, MID, 32, 1.0, steps) == pytest.approx(0.4626, abs=1e-4)
        climb = [sky_view(dem, MID, c, 32) for c in (25, 20, 10, 5)]
        assert climb == pytest.approx([0.5619, 0.6252, 0.6961, 0.7177], abs=1e-4)


class TestTrenchesAndPits:
    @pytest.mark.parametrize(
        ("depth", "half", "centre", "foot"),
        [(5.0, 2, 0.3315, 0.3035), (10.0, 2, 0.1785, 0.1838), (10.0, 10, 0.5523, 0.4303)],
    )
    def test_trenches(self, depth, half, centre, foot):
        dem = trench(SIZE, depth, half)
        assert sky_view(dem, MID, MID, 32) == pytest.approx(centre, abs=1e-4)
        assert sky_view(dem, MID, MID + half, 32) == pytest.approx(foot, abs=1e-4)
        assert centre < canyon_guess(math.degrees(math.atan2(depth, half + 1)))

    @pytest.mark.parametrize(
        ("depth", "radius", "reading"),
        [(5.0, 5, 0.3293), (10.0, 5, 0.1252), (10.0, 20, 0.5631)],
    )
    def test_a_pit_reads_between_the_laws_for_its_two_rims(self, depth, radius, reading):
        dem = pit(SIZE, depth, radius)
        read = sky_view(dem, MID, MID, 32)
        assert read == pytest.approx(reading, abs=1e-4)
        assert pit_law(depth, radius) < read < pit_law(depth, radius + 1)

    def test_the_grid_and_its_interior_mean(self):
        grid = sky_view_grid(v_valley(31, 1.0), 16, 1.0, 32)
        assert mean_interior(grid, 5) == pytest.approx(0.6327, abs=1e-4)
        assert grid[15][15] == pytest.approx(0.4884, abs=1e-4)


class TestRefusals:
    def test_bad_grids_cells_and_fans(self):
        with pytest.raises(Invalid):
            sky_view([], 0, 0)
        with pytest.raises(Invalid):
            horizon_angle(plane(5), 9, 0, 0.0)
        with pytest.raises(Invalid):
            horizon_angle(plane(5), 2, 2, 0.0, 0.0)
        with pytest.raises(Invalid):
            sky_view(plane(5), 2, 2, 0)
        with pytest.raises(Invalid):
            mean_interior(plane(5), 3)
        with pytest.raises(Invalid):
            v_valley(5, -1.0)
        with pytest.raises(Invalid):
            trench(5, -1.0, 1)
        with pytest.raises(Invalid):
            plane(2)
