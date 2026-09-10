from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.mapsheets import (
    SheetGrid,
    expected_sheets,
    hierarchical,
    one_sheet_law,
    sheets_for_route,
    straddle_histogram,
)

GRID = SheetGrid(50.0, 50.0)


class TestNames:
    def test_names_and_parsing_round_trip(self):
        assert GRID.name((0, 0)) == "A001"
        assert GRID.name((11, 27)) == "AB012"
        assert GRID.name((3, 702)) == "AAA004"
        for sheet in ((0, 0), (11, 27), (3, 702), (99, 25), (0, 26)):
            assert GRID.parse(GRID.name(sheet)) == sheet
        assert GRID.bounds((2, 3)) == (100.0, 150.0, 150.0, 200.0)
        assert GRID.sheet_of(125.0, 175.0) == (2, 3)
        assert GRID.sheets_covering(40, 40, 60, 60) == [(0, 0), (1, 0), (0, 1), (1, 1)]


class TestStraddling:
    @pytest.mark.parametrize(
        ("side", "one", "two", "four", "mean"),
        [
            (5.0, 0.8057, 0.183, 0.0112, 1.2167),
            (10.0, 0.6375, 0.3194, 0.043, 1.4486),
            (20.0, 0.3574, 0.4797, 0.1629, 1.9684),
            (30.0, 0.1598, 0.4753, 0.365, 2.5703),
        ],
    )
    def test_the_one_sheet_law(self, side, one, two, four, mean):
        hist = straddle_histogram(GRID, side, side, random.Random(630), 20000)
        assert (hist[1], hist[2], hist[4]) == pytest.approx((one, two, four), abs=1e-4)
        assert hist[1] == pytest.approx(one_sheet_law(side, side, 50, 50), abs=0.006)
        read_mean = sum(k * v for k, v in hist.items())
        assert read_mean == pytest.approx(mean, abs=1e-4)
        assert read_mean == pytest.approx(expected_sheets(side, side, 50, 50), abs=0.012)

    def test_a_box_the_size_of_a_sheet_always_touches_four(self):
        hist = straddle_histogram(GRID, 50.0, 50.0, random.Random(630), 2000)
        assert hist == {4: 1.0}
        fine = hierarchical(GRID, 5)
        assert (fine.width, fine.height) == (10.0, 10.0)
        assert straddle_histogram(fine, 10.0, 10.0, random.Random(631), 2000) == {4: 1.0}
        assert one_sheet_law(10, 10, 10, 10) == 0.0

    def test_a_route(self):
        route = [(10.0, 10.0), (120.0, 40.0), (130.0, 160.0), (40.0, 140.0)]
        sheets = sheets_for_route(GRID, route)
        assert len(sheets) == 9
        assert sheets[0] == (0, 0) and sheets[-1] == (0, 2)


class TestRefusals:
    def test_bad_grids_names_boxes_and_routes(self):
        with pytest.raises(Invalid):
            SheetGrid(0, 10)
        with pytest.raises(Invalid):
            GRID.parse("a001")
        with pytest.raises(Invalid):
            GRID.parse("12")
        with pytest.raises(Invalid):
            GRID.sheets_covering(10, 10, 0, 0)
        with pytest.raises(Invalid):
            straddle_histogram(GRID, 1, 1, random.Random(1), 0)
        with pytest.raises(Invalid):
            hierarchical(GRID, 1)
        with pytest.raises(Invalid):
            sheets_for_route(GRID, [(0, 0)])
