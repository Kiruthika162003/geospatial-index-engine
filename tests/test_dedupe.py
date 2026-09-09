from __future__ import annotations

import random

import pytest

from atlas.dedupe import (
    double_snap_merge,
    merged_count,
    merged_distances,
    pairs_split,
    planted_pairs,
    radius_merge,
    snap_merge,
)
from atlas.errors import Invalid


class TestPlantedPairs:
    @pytest.mark.parametrize(
        ("separation", "snap_split", "double_split"),
        [(0.1, 241, 10), (0.2, 462, 43), (0.5, 1091, 251), (0.9, 1761, 1356)],
    )
    def test_snapping_splits_pairs_in_proportion_to_their_separation(
        self, separation, snap_split, double_split
    ):
        rng = random.Random(235)
        for earlier in (0.1, 0.2, 0.5, 0.9):
            if earlier == separation:
                break
            planted_pairs(2000, earlier, rng)
        points, pairs = planted_pairs(2000, separation, rng)
        assert pairs_split(snap_merge(points, 1.0), pairs) == snap_split
        assert pairs_split(double_snap_merge(points, 1.0), pairs) == double_split
        assert pairs_split(radius_merge(points, 1.0), pairs) == 0

    def test_the_split_fraction_is_about_1_2_times_the_separation(self):
        rng = random.Random(240)
        points, pairs = planted_pairs(4000, 0.1, rng)
        fraction = pairs_split(snap_merge(points, 1.0), pairs) / 4000
        assert 0.10 < fraction < 0.14


class TestOverreachAndChaining:
    def test_snapping_merges_past_a_cell_and_the_radius_method_never_does(self):
        rng = random.Random(237)
        points = [(rng.uniform(0, 300), rng.uniform(0, 300)) for _ in range(3000)]
        snap, radius = snap_merge(points, 1.0), radius_merge(points, 1.0)
        assert merged_count(snap, 3000) == 58
        assert merged_count(radius, 3000) == 154
        assert merged_count(double_snap_merge(points, 1.0), 3000) == 93
        snapped = merged_distances(points, snap)
        assert max(snapped) == pytest.approx(1.174, abs=1e-3)
        assert max(merged_distances(points, radius)) < 1.0
        assert sum(1 for d in snapped if d > 1) / len(snapped) == pytest.approx(0.018, abs=1e-3)

    def test_the_radius_method_chains_everything_past_the_percolation_density(self):
        rng = random.Random(237)
        for _ in range(3000):
            rng.uniform(0, 300)
            rng.uniform(0, 300)
        largest = {}
        for extent in (300, 100, 60, 40, 30):
            points = [(rng.uniform(0, extent), rng.uniform(0, extent)) for _ in range(3000)]
            density = round(3000 / extent**2, 3)
            largest[density] = (
                max(len(g) for g in radius_merge(points, 1.0)),
                max(len(g) for g in snap_merge(points, 1.0)),
            )
        assert largest[0.033] == (3, 2)
        assert largest[0.833] == (51, 6)
        assert largest[1.875] == (2936, 8)
        assert largest[3.333] == (2995, 10)


class TestExactDuplicates:
    def test_every_method_merges_all_of_them(self):
        rng = random.Random(238)
        base = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(500)]
        duplicated = base + base[:200]
        for groups in (
            snap_merge(duplicated, 0.001),
            double_snap_merge(duplicated, 0.001),
            radius_merge(duplicated, 0.001),
        ):
            assert merged_count(groups, len(duplicated)) == 200


class TestRefusals:
    def test_non_positive_sizes_are_refused(self):
        with pytest.raises(Invalid):
            snap_merge([(0, 0)], 0)
        with pytest.raises(Invalid):
            radius_merge([(0, 0)], -1)
