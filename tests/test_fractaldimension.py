from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.fractaldimension import (
    box_count,
    densify,
    filled_square,
    fit_dimension,
    geometric_sizes,
    koch_curve,
    koch_dimension,
)


def _dense_koch(levels: int):
    seg = 3.0**-levels
    return densify(koch_curve(levels), seg / 4), seg


class TestTheKochCurve:
    def test_the_curve_has_four_to_the_levels_segments(self):
        assert len(koch_curve(0)) == 2
        assert len(koch_curve(3)) == 4**3 + 1
        assert len(koch_curve(5)) == 1025

    def test_box_counting_inside_the_features_lands_within_a_few_hundredths(self):
        dense, seg = _dense_koch(5)
        assert fit_dimension(dense, geometric_sizes(0.25, seg * 3, 8)) == pytest.approx(
            1.2628, abs=1e-3
        )
        assert fit_dimension(dense, geometric_sizes(0.5, seg, 8)) == pytest.approx(
            1.2487, abs=1e-3
        )
        assert koch_dimension() == pytest.approx(math.log(4) / math.log(3))

    def test_more_levels_do_not_remove_the_segment_scale_bias(self):
        readings = []
        for levels in (4, 5, 6):
            dense, seg = _dense_koch(levels)
            readings.append(fit_dimension(dense, geometric_sizes(0.5, seg, 8)))
        assert readings == pytest.approx([1.2491, 1.2487, 1.2493], abs=1e-3)

    def test_the_failure_modes_read_wrong_in_known_ways(self):
        dense, seg = _dense_koch(5)
        assert fit_dimension(dense, geometric_sizes(seg, seg / 64, 6)) == pytest.approx(
            0.3215, abs=1e-3
        )
        assert fit_dimension(dense, geometric_sizes(64.0, 2.0, 6)) == 0.0
        raw = koch_curve(5)
        assert fit_dimension(raw, geometric_sizes(seg, seg / 64, 6)) == pytest.approx(
            0.0208, abs=1e-3
        )
        assert fit_dimension(dense, geometric_sizes(0.25, 1 / 9, 8)) == pytest.approx(
            1.059, abs=1e-3
        )


class TestKnownDimensions:
    def test_a_line_reads_one_as_the_coarse_boxes_shrink(self):
        line = densify([(0.0, 0.0), (1.0, 0.0)], 1e-4)
        coarse = fit_dimension(line, geometric_sizes(0.25, 0.002, 8))
        fine = fit_dimension(line, geometric_sizes(0.02, 0.001, 8))
        assert coarse == pytest.approx(0.9726, 1e-3)
        assert fine == pytest.approx(0.9956, 1e-3)

    def test_a_square_reads_two_once_the_edge_effect_fades(self):
        square = filled_square(1.0, 1e-3)
        assert len(square) == 1001 * 1001
        coarsest = (0.25, 0.1, 0.05, 0.02)
        readings = [fit_dimension(square, geometric_sizes(lo, 0.004, 8)) for lo in coarsest]
        assert readings == pytest.approx([1.9087, 1.9577, 1.9731, 1.9861], abs=1e-3)
        assert readings == sorted(readings)

    def test_a_random_walk_reads_between_the_koch_curve_and_two(self):
        rng = random.Random(192)
        walk = [(0.0, 0.0)]
        for _ in range(20000):
            x, y = walk[-1]
            a = rng.uniform(0, 2 * math.pi)
            walk.append((x + 0.01 * math.cos(a), y + 0.01 * math.sin(a)))
        xs = [p[0] for p in walk]
        ys = [p[1] for p in walk]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        assert fit_dimension(walk, geometric_sizes(span / 4, 0.02, 8)) == pytest.approx(
            1.647, abs=1e-2
        )


class TestBoxes:
    def test_box_count_and_sizes(self):
        assert box_count([(0.1, 0.1), (0.2, 0.2), (0.9, 0.9)], 0.5) == 2
        sizes = geometric_sizes(1.0, 0.125, 4)
        assert sizes == pytest.approx([1.0, 0.5, 0.25, 0.125])

    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            koch_curve(-1)
        with pytest.raises(Invalid):
            box_count([(0, 0)], 0)
        with pytest.raises(Invalid):
            fit_dimension([(0, 0)], [1.0])
        with pytest.raises(Invalid):
            fit_dimension([(0, 0)], [1.0, 1.0])
        with pytest.raises(Invalid):
            geometric_sizes(1.0, 2.0, 4)
        with pytest.raises(Invalid):
            densify([(0, 0), (1, 1)], 0)
