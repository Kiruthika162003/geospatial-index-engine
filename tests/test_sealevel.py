from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.sealevel import (
    area,
    bathtub,
    coast_with_basin,
    connected,
    overstatement,
    with_diagonal_gap,
)

GRID, SEA = coast_with_basin(30, 0.5, 12.0, 8.0)


class TestTheBasin:
    @pytest.mark.parametrize(
        ("level", "tub", "joined"),
        [
            (2, 120, 120),
            (5, 570, 300),
            (9.9, 870, 600),
            (12.0, 870, 600),
            (12.1, 900, 900),
            (15, 900, 900),
        ],
    )
    def test_the_bathtub_overstates_by_exactly_the_basin_until_the_sea_tops_the_ridge(
        self, level, tub, joined
    ):
        assert area(bathtub(GRID, level)) == tub
        assert area(connected(GRID, level, SEA)) == joined
        basin_below = sum(1 for r in range(21, 30) for c in range(30) if GRID[r][c] < level)
        assert overstatement(GRID, level, SEA) == (basin_below if level <= 12.0 else 0)

    def test_the_identities_hold_over_200_levels(self):
        previous_tub = previous_joined = -1
        for tenth in range(200):
            level = tenth / 10
            tub, joined = area(bathtub(GRID, level)), area(connected(GRID, level, SEA))
            assert joined <= tub
            assert tub >= previous_tub
            assert joined >= previous_joined
            previous_tub, previous_joined = tub, joined
        flooded = connected(GRID, 5, SEA)
        wet = [(r, c) for r, row in enumerate(flooded) for c, v in enumerate(row) if v]
        assert all(GRID[r][c] < 5 for r, c in wet)


class TestConnectivity:
    @pytest.mark.parametrize(
        ("level", "four", "eight"),
        [(5.0, 300, 300), (7.0, 420, 420), (11.0, 601, 870), (12.5, 900, 900)],
    )
    def test_a_diagonal_gap_floods_the_basin_under_eight_and_not_four(self, level, four, eight):
        gapped = with_diagonal_gap(GRID, 6.0)
        assert area(connected(gapped, level, SEA)) == four
        assert area(connected(gapped, level, SEA, eight=True)) == eight


class TestRefusals:
    def test_empty_grids_and_sea_cells_off_the_grid_are_refused(self):
        with pytest.raises(Invalid):
            bathtub([], 1.0)
        with pytest.raises(Invalid):
            connected(GRID, 1.0, [(99, 0)])
