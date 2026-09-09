from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.kmeans import cluster as kmeans_cluster
from atlas.kmedoids import (
    blobs,
    build,
    cost,
    lloyd,
    matched_shift,
    medoid_gap,
    pam,
    swap,
    with_outliers,
)


@pytest.fixture(scope="module")
def clean():
    return blobs(4, 50, random.Random(330))


class TestClean:
    def test_medoids_land_within_a_data_points_reach(self, clean):
        points, truth = clean
        medoids, labels, total, rounds = pam(points, 4)
        assert total == pytest.approx(269.122, abs=1e-3)
        assert rounds == 1
        assert matched_shift(medoids, truth) == pytest.approx(0.5711, abs=1e-4)
        assert medoid_gap(points, labels, medoids) == pytest.approx(0.3749, abs=1e-4)
        assert all(m in points for m in medoids)
        assert sorted(set(labels)) == [0, 1, 2, 3]
        assert matched_shift(lloyd(points, 4, random.Random(331)), truth) == pytest.approx(
            0.4115, abs=1e-4
        )


class TestOutliers:
    @pytest.mark.parametrize(
        ("count", "pam_shift", "lloyd_shift", "seeded_shift"),
        [(1, 0.5711, 7.172, 3.1602), (2, 0.5711, 7.172, 7.0322), (5, 11.8229, 7.172, 7.172)],
    )
    def test_two_are_shrugged_off_and_five_are_followed(
        self, clean, count, pam_shift, lloyd_shift, seeded_shift
    ):
        points, truth = clean
        dirty = with_outliers(points, count, random.Random(332))
        medoids, _, _, _ = pam(dirty, 4)
        assert matched_shift(medoids, truth) == pytest.approx(pam_shift, abs=1e-4)
        centers = lloyd(dirty, 4, random.Random(333))
        assert matched_shift(centers, truth) == pytest.approx(lloyd_shift, abs=1e-3)
        seeds = [points[i] for i in (0, 50, 100, 150)]
        _, seeded, _ = kmeans_cluster(dirty, seeds)
        assert matched_shift(seeded, truth) == pytest.approx(seeded_shift, abs=1e-3)
        assert sum(1 for m in medoids if m[0] > 50) == (1 if count == 5 else 0)


class TestBuildAndSwap:
    @pytest.mark.parametrize(
        ("per", "built", "swapped", "gain"),
        [(25, 161.681, 133.836, 0.1722), (50, 311.87, 257.841, 0.1732)],
    )
    def test_the_swap_phase_gains_a_sixth(self, per, built, swapped, gain):
        points, _ = blobs(4, per, random.Random(334))
        first = build(points, 4)
        assert cost(points, [points[i] for i in first]) == pytest.approx(built, abs=1e-3)
        before = cost(points, [points[i] for i in first])
        final, rounds = swap(points, first)
        after = cost(points, [points[i] for i in final])
        assert after == pytest.approx(swapped, abs=1e-3)
        assert 1 - after / before == pytest.approx(gain, abs=1e-4)
        assert 1 <= rounds <= 2

    @pytest.mark.parametrize(
        ("spread", "gap", "shift"), [(0.5, 0.1598, 0.2357), (2.0, 0.6392, 0.9429)]
    )
    def test_the_medoid_gap_is_a_third_of_the_spread(self, spread, gap, shift):
        points, truth = blobs(4, 50, random.Random(335), spread=spread)
        medoids, labels, _, _ = pam(points, 4)
        assert medoid_gap(points, labels, medoids) == pytest.approx(gap, abs=1e-4)
        assert matched_shift(medoids, truth) == pytest.approx(shift, abs=1e-4)
        assert gap > spread / 50**0.5


class TestRefusals:
    def test_bad_k_and_empty_points(self):
        with pytest.raises(Invalid):
            build([(0, 0), (1, 1)], 3)
        with pytest.raises(Invalid):
            build([(0, 0)], 0)
        with pytest.raises(Invalid):
            pam([], 1)
        with pytest.raises(Invalid):
            lloyd([(0, 0)], 2, random.Random(1))
