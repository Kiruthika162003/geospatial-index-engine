from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.integralimage import (
    SummedArea,
    count_in_window,
    integer_grid,
    local_variance,
    naive_box_sum,
    naive_mean_field,
    random_grid,
    worst_difference,
)


def _worst(size, single, rng_seed=451):
    grid = random_grid(size, random.Random(450 + size))
    table = SummedArea(grid, single=single)
    q = random.Random(rng_seed)
    worst = 0.0
    for _ in range(300):
        r0, c0 = q.randrange(size), q.randrange(size)
        r1, c1 = min(size - 1, r0 + q.randrange(1, 4)), min(size - 1, c0 + q.randrange(1, 4))
        truth = naive_box_sum(grid, r0, c0, r1, c1)
        worst = max(worst, abs(table.box_sum(r0, c0, r1, c1) - truth))
    return worst


class TestPrecision:
    @pytest.mark.parametrize(
        ("size", "double", "single"), [(64, 1e-12, 1e-3), (256, 2e-11, 1e-2)]
    )
    def test_double_holds_and_single_drifts(self, size, double, single):
        assert _worst(size, False) < double
        narrow = _worst(size, True)
        assert 1e-5 < narrow < single
        assert narrow > 1e6 * _worst(size, False)

    def test_integer_grids_are_exact(self):
        ints = integer_grid(256, random.Random(452))
        table = SummedArea(ints)
        q = random.Random(453)
        for _ in range(200):
            r0, c0 = q.randrange(250), q.randrange(250)
            truth = naive_box_sum(ints, r0, c0, r0 + 5, c0 + 5)
            assert table.box_sum(r0, c0, r0 + 5, c0 + 5) == truth


class TestFields:
    @pytest.mark.parametrize("radius", [1, 3, 15])
    def test_the_mean_field_matches_the_naive_field(self, radius):
        grid = random_grid(128, random.Random(454))
        table = SummedArea(grid)
        gap = worst_difference(table.mean_field(radius), naive_mean_field(grid, radius))
        assert gap < 1e-12

    def test_variance_and_counts(self):
        grid = random_grid(128, random.Random(454))
        variance = local_variance(grid, 3)
        assert sum(sum(row) for row in variance) / 128**2 == pytest.approx(0.08108, abs=1e-4)
        mask = [[(r + c) % 3 == 0 for c in range(20)] for r in range(20)]
        assert count_in_window(mask, 10, 10, 2) == 8
        table = SummedArea([[1.0, 2.0], [3.0, 4.0]])
        assert table.box_sum(0, 0, 1, 1) == 10.0
        assert table.window_mean(0, 0, 0) == 1.0
        assert table.window_mean(1, 1, 5) == 2.5


class TestRefusals:
    def test_bad_grids_and_boxes(self):
        with pytest.raises(Invalid):
            SummedArea([])
        table = SummedArea([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(Invalid):
            table.box_sum(1, 0, 0, 1)
        with pytest.raises(Invalid):
            table.box_sum(0, 0, 2, 0)
