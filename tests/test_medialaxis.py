from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.medialaxis import (
    components,
    disc,
    endpoints,
    expected_rectangle_cells,
    inside_distance,
    junctions,
    l_shape,
    notched,
    prune,
    reach,
    rectangle,
    ridge,
)


def _summary(mask):
    axis = ridge(mask)
    dist = inside_distance(mask)
    counts = (len(axis), components(axis), len(junctions(axis)), len(endpoints(axis)))
    return (*counts, reach(axis, dist))


class TestRectangles:
    @pytest.mark.parametrize(
        ("width", "height", "cells", "joins", "ends", "radius"),
        [
            (100, 41, 140, 2, 4, 21.0),
            (100, 40, 200, 124, 4, 20.0),
            (61, 61, 121, 1, 4, 31.0),
            (60, 60, 120, 4, 4, 30.0),
            (100, 10, 200, 184, 4, 5.0),
            (100, 1, 100, 0, 2, 1.0),
        ],
    )
    def test_spine_arms_and_the_even_doubling(self, width, height, cells, joins, ends, radius):
        assert _summary(rectangle(width, height)) == (cells, 1, joins, ends, radius)

    def test_the_odd_count_is_the_spine_plus_four_arms(self):
        assert expected_rectangle_cells(100, 41) == 139
        assert len(ridge(rectangle(100, 41))) == 139 + 1
        assert expected_rectangle_cells(61, 61) == 120


class TestDiscs:
    @pytest.mark.parametrize(
        ("radius", "cells", "reach_read", "one_below", "two_below"),
        [
            (5, 45, 5.099, 9, 13),
            (10, 197, 10.05, 5, 21),
            (20, 841, 20.025, 5, 13),
            (40, 3713, 40.012, 5, 13),
        ],
    )
    def test_the_staircase_makes_most_of_the_disc_maximal(
        self, radius, cells, reach_read, one_below, two_below
    ):
        mask = disc(radius)
        axis = ridge(mask)
        dist = inside_distance(mask)
        assert len(axis) == cells
        assert components(axis) == 1
        assert reach(axis, dist) == pytest.approx(reach_read, abs=1e-3)
        assert len(prune(axis, dist, radius - 1)) == one_below
        assert len(prune(axis, dist, radius - 2)) == two_below
        if radius == 40:
            filled = sum(sum(1 for v in row if v) for row in mask)
            assert filled == 5025
            assert cells / filled == pytest.approx(0.739, abs=1e-3)


class TestNotches:
    def test_one_cell_off_the_edge_sprouts_169_that_pruning_keeps(self):
        base = rectangle(100, 41)
        base_axis = ridge(base)
        base_dist = inside_distance(base)
        for column, extra in ((50, 169), (30, 169), (51, 169), (15, 96)):
            mask = notched(base, 2, column)
            axis = ridge(mask)
            assert len(axis) - len(base_axis) == extra
            assert components(axis) == 1
        mask = notched(base, 2, 50)
        axis = ridge(mask)
        dist = inside_distance(mask)
        kept = []
        for t in (2, 3, 5, 10):
            kept.append(len(prune(axis, dist, t)) - len(prune(base_axis, base_dist, t)))
        assert kept == [167, 161, 153, 115]
        assert len(ridge(notched(base, 22, 50))) - len(base_axis) == 539


class TestLShapes:
    def test_odd_and_even_thickness(self):
        assert _summary(l_shape(60, 11)) == (144, 1, 34, 5, 7.0)
        assert _summary(l_shape(60, 10))[0] == 233
        assert _summary(l_shape(60, 21))[4] == pytest.approx(12.728, abs=1e-3)


class TestRefusals:
    def test_bad_masks_shapes_and_prunes(self):
        with pytest.raises(Invalid):
            inside_distance([])
        with pytest.raises(Invalid):
            inside_distance([[False, False]])
        with pytest.raises(Invalid):
            inside_distance([[True, True]])
        with pytest.raises(Invalid):
            rectangle(0, 5)
        with pytest.raises(Invalid):
            disc(0)
        with pytest.raises(Invalid):
            notched(rectangle(5, 5), 0, 0)
        with pytest.raises(Invalid):
            l_shape(5, 5)
        with pytest.raises(Invalid):
            prune([(0, 0)], [[1.0]], -1)
        with pytest.raises(Invalid):
            reach([], [[1.0]])
