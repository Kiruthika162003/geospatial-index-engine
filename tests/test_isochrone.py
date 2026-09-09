from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.isochrone import (
    OCTILE_BALL,
    band,
    count,
    disc_area,
    effective_cost,
    frontier,
    multi_reach_count,
    nested_counts,
    octile_ball_area,
    radius_from_count,
    random_cost,
    reach,
    reach_mask,
    shadow,
    uniform_cost,
    union_reach_count,
    walled_cost,
)

SOURCE = (60, 60)


class TestTheOctagon:
    def test_counts_meet_the_octile_ball_and_nine_tenths_of_the_disc(self):
        grid = uniform_cost(121)
        counts = nested_counts(grid, SOURCE, [5, 10, 20, 30, 50])
        assert counts == [73, 285, 1137, 2549, 7069]
        budgets = (5, 10, 20, 30, 50)
        ratios = [n / octile_ball_area(b) for n, b in zip(counts, budgets, strict=True)]
        assert ratios == pytest.approx([1.0324, 1.0076, 1.005, 1.0013, 0.9997], abs=1e-4)
        assert counts[-1] / disc_area(50) == pytest.approx(0.9001, abs=1e-4)
        assert OCTILE_BALL / math.pi == pytest.approx(0.9003, abs=1e-4)
        assert radius_from_count(counts[-1]) == pytest.approx(50.0, abs=0.01)

    def test_frontier_and_bands(self):
        grid = uniform_cost(121)
        assert len(frontier(reach_mask(grid, SOURCE, 10))) == 56
        assert len(frontier(reach_mask(grid, SOURCE, 50))) == 280
        assert len(band(grid, SOURCE, 0, 10)) == 285
        assert len(band(grid, SOURCE, 10, 20)) == 1137 - 285
        assert len(reach(grid, SOURCE, 5)) == count(reach_mask(grid, SOURCE, 5)) == 73
        assert effective_cost(grid, SOURCE, 5) == pytest.approx(0.9842, abs=1e-4)
        assert effective_cost(grid, SOURCE, 50) == pytest.approx(1.0001, abs=1e-4)


class TestRandomCost:
    @pytest.mark.parametrize(
        ("low", "high", "reading"), [(1.0, 2.0, 1.3349), (1.0, 10.0, 3.8272)]
    )
    def test_the_effective_cost_sits_below_the_mean_and_the_log_mean(self, low, high, reading):
        readings = []
        for seed in range(5):
            grid = random_cost(121, low, high, random.Random(280 + seed))
            readings.append(effective_cost(grid, SOURCE, 40))
        mean = sum(readings) / 5
        assert mean == pytest.approx(reading, abs=1e-4)
        assert mean < (high - low) / math.log(high / low) < (low + high) / 2


class TestWallsAndSources:
    def test_a_gap_on_axis_keeps_the_far_side_and_a_corner_gap_keeps_none(self):
        open_grid = uniform_cost(121)
        assert shadow(open_grid, SOURCE, 45, 80) == (4463, 1193)
        assert shadow(walled_cost(121, 80, 60), SOURCE, 45, 80) == (4463, 859)
        assert shadow(walled_cost(121, 80, 30), SOURCE, 45, 80) == (4463, 58)
        assert shadow(walled_cost(121, 80, 0), SOURCE, 45, 80) == (4463, 0)
        assert shadow(walled_cost(121, 80, 60), SOURCE, 60, 80)[1] == 2224

    def test_multi_source_equals_the_union_and_overlaps_at_eight(self):
        grid = random_cost(81, 1, 3, random.Random(290))
        rng = random.Random(291)
        sources = [(rng.randrange(81), rng.randrange(81)) for _ in range(8)]
        for k, expected in ((2, 451), (8, 1470)):
            assert multi_reach_count(grid, sources[:k], 15) == expected
            assert union_reach_count(grid, sources[:k], 15) == expected
        singles = sum(count(reach_mask(grid, s, 15)) for s in sources)
        assert singles == 1551


class TestRefusals:
    def test_bad_budgets_bands_grids_and_sources(self):
        grid = uniform_cost(5)
        with pytest.raises(Invalid):
            reach_mask(grid, (0, 0), -1)
        with pytest.raises(Invalid):
            band(grid, (0, 0), 3, 2)
        with pytest.raises(Invalid):
            nested_counts(grid, (0, 0), [])
        with pytest.raises(Invalid):
            multi_reach_count(grid, [], 1)
        with pytest.raises(Invalid):
            uniform_cost(0)
        with pytest.raises(Invalid):
            random_cost(3, 0, 1, random.Random(1))
