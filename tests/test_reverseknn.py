from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.reverseknn import (
    count_distribution,
    counts,
    mutual_fraction,
    nearest_of_each,
    reverse_neighbours,
    ring_side,
    ring_with_center,
)


class TestRandomPoints:
    def test_counts_average_one_top_out_at_four_and_leave_28_percent_with_none(self):
        rng = random.Random(239)
        totals: dict[int, int] = {}
        largest = 0
        mutual = []
        for _ in range(40):
            pts = [(rng.random(), rng.random()) for _ in range(300)]
            dist = count_distribution(pts)
            for k, v in dist.items():
                totals[k] = totals.get(k, 0) + v
            largest = max(largest, *dist)
            mutual.append(mutual_fraction(pts))
            assert sum(counts(pts)) == 300
        n = 12000
        assert {k: round(v / n, 4) for k, v in totals.items()} == {
            0: 0.283,
            1: 0.4652,
            2: 0.2218,
            3: 0.0288,
            4: 0.0012,
        }
        assert largest == 4
        assert sum(k * v for k, v in totals.items()) / n == 1.0
        assert sum(mutual) / 40 == pytest.approx(0.6223, abs=1e-3)
        assert 0.58 < min(mutual) < max(mutual) < 0.68


class TestTheBound:
    @pytest.mark.parametrize(("k", "count"), [(3, 3), (4, 4), (5, 5), (7, 0), (8, 0)])
    def test_rings_up_to_five_count_the_center_and_past_six_prefer_each_other(self, k, count):
        ring = ring_with_center(k)
        assert counts(ring)[0] == count
        assert (ring_side(k) > 1.0) == (count > 0)

    def test_six_is_a_tie_since_a_hexagons_side_is_its_radius(self):
        assert ring_side(6) == pytest.approx(1.0)
        ring = ring_with_center(6)
        assert counts(ring)[0] == 1
        assert max(counts(ring)) == 2

    def test_reverse_lists_invert_the_nearest_map(self):
        rng = random.Random(242)
        pts = [(rng.random(), rng.random()) for _ in range(50)]
        nearest = nearest_of_each(pts)
        for j, members in enumerate(reverse_neighbours(pts)):
            assert all(nearest[i] == j for i in members)


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            nearest_of_each([(0, 0)])
        with pytest.raises(Invalid):
            ring_with_center(0)
        with pytest.raises(Invalid):
            ring_with_center(3, radius=0)
