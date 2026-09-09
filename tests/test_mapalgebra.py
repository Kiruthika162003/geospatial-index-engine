from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.mapalgebra import (
    curvature_bias,
    focal,
    local,
    mean,
    reclassify,
    sampled,
    scale,
    value_range,
)


def plane(x: float, y: float) -> float:
    return 2.0 + 0.5 * x - 0.3 * y


def bowl(x: float, y: float) -> float:
    return x * x + y * y


PLANE = sampled(plane, 21, 21)
BOWL = sampled(bowl, 21, 21)


class TestFocalMean:
    @pytest.mark.parametrize(("radius", "bias"), [(1, 4 / 3), (2, 4.0), (3, 8.0)])
    def test_exact_on_a_plane_and_biased_by_curvature_on_a_bowl(self, radius, bias):
        smooth_plane = focal(PLANE, radius, mean)
        smooth_bowl = focal(BOWL, radius, mean)
        interior = range(radius, 21 - radius)
        cells = [(r, c) for r in interior for c in interior]
        assert max(abs(smooth_plane[r][c] - PLANE[r][c]) for r, c in cells) < 1e-12
        biases = {round(smooth_bowl[r][c] - BOWL[r][c], 9) for r in interior for c in interior}
        assert biases == {round(bias, 9)}
        assert curvature_bias(2.0, radius) == pytest.approx(bias)

    def test_both_edge_rules_bias_the_same_way_padding_by_two_thirds(self):
        skip = focal(PLANE, 1, mean)
        pad = focal(PLANE, 1, mean, pad=True)
        assert PLANE[10][0] == -1.0
        assert skip[10][0] == pytest.approx(-0.75)
        assert pad[10][0] == pytest.approx(-0.8333, abs=1e-4)
        assert (skip[0][10], pad[0][10]) == pytest.approx((6.85, 6.9))
        assert (skip[0][0], pad[0][0]) == pytest.approx((2.1, 2.0667), abs=1e-4)


class TestOtherFocals:
    def test_focal_range_is_2r_times_the_sum_of_the_slope_components(self):
        ranges = focal(PLANE, 2, value_range)
        assert {round(ranges[r][c], 9) for r in range(2, 19) for c in range(2, 19)} == {3.2}

    def test_focal_max_is_the_far_corner(self):
        maxima = focal(PLANE, 1, max)
        for r in range(1, 20):
            for c in range(1, 20):
                assert maxima[r][c] == pytest.approx(plane(c + 1, r - 1))


class TestLocalOperations:
    def test_missing_cells_propagate_through_arithmetic_and_skip_reclassification(self):
        a = [[1.0, None], [3.0, 4.0]]
        b = [[10.0, 20.0], [None, 40.0]]
        assert local(lambda x, y: x + y, a, b) == [[11.0, None], [None, 44.0]]
        assert scale(a, 2.0, 1.0) == [[3.0, None], [7.0, 9.0]]
        classes = reclassify([[0.5, 1.5, None, 9.0]], [1.0, 5.0], [10, 20, 30])
        assert classes == [[10, 20, None, 30]]

    def test_refusals(self):
        a = [[1.0, None], [3.0, 4.0]]
        with pytest.raises(Invalid):
            local(lambda x, _y: x, a, [[1.0]])
        with pytest.raises(Invalid):
            focal(a, 0, mean)
        with pytest.raises(Invalid):
            reclassify(a, [1.0], [1])
        with pytest.raises(Invalid):
            reclassify(a, [5.0, 1.0], [1, 2, 3])
        with pytest.raises(Invalid):
            focal([], 1, mean)
