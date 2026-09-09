from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.spatialhash import (
    SpatialHash,
    best_cell,
    brute_within,
    bucket_histogram,
    build,
    candidate_ratio,
    cheapest_cell,
    clustered,
    cost_curve,
    expected_cells,
    query_cost,
    uniform,
)


@pytest.fixture(scope="module")
def cloud():
    return uniform(20000, random.Random(390)), uniform(300, random.Random(391))


class TestTheLaw:
    @pytest.mark.parametrize(
        ("radius", "ratios", "best"),
        [
            (5.0, (1.57, 1.99, 2.83, 5.01, 11.56, 30.27), (0.25, 2.0, 2.0)),
            (10.0, (1.63, 2.01, 2.83, 4.96, 10.6, 28.38), (0.25, 1.0, 2.0)),
            (40.0, (1.59, 1.96, 2.81, 4.84, 10.43, 29.1), (0.25, 0.5, 0.5)),
        ],
    )
    def test_candidates_over_hits_and_the_cost_question(self, cloud, radius, ratios, best):
        points, queries = cloud
        cells = [radius * f for f in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
        curve = cost_curve(points, queries, radius, cells)
        read = tuple(round(curve[c][1] / curve[c][0], 2) for c in cells)
        assert read == ratios
        for c in cells[:4]:
            law = candidate_ratio(radius, c)
            assert curve[c][1] / curve[c][0] == pytest.approx(law, rel=0.06)
            assert curve[c][2] == expected_cells(radius, c)
        for cost, fraction in zip((0.0, 1.0, 5.0), best, strict=True):
            assert best_cell(curve, cost) / radius == fraction
            assert cheapest_cell(0.02, radius, cost, cells) / radius == fraction

    def test_the_ratio_at_the_radius_is_nine_over_pi(self):
        assert candidate_ratio(10.0, 10.0) == pytest.approx(9 / math.pi)
        assert candidate_ratio(10.0, 20.0) == pytest.approx(16 / math.pi)


class TestBuckets:
    def test_uniform_buckets_are_poisson(self, cloud):
        points, queries = cloud
        grid = build(points, 10.0)
        mean, worst = grid.occupancy()
        assert mean == pytest.approx(2.2967, abs=1e-4)
        assert worst == 9
        head = list(bucket_histogram(grid).items())[:8]
        counts = [2763, 2708, 1826, 917, 336, 122, 27, 7]
        assert head == list(enumerate(counts, start=1))
        for q in queries[:100]:
            assert sorted(grid.within(q, 10.0)[0]) == sorted(brute_within(points, q, 10.0))

    def test_clustered_buckets(self):
        points = clustered(20000, 20, random.Random(392))
        grid = build(points, 10.0)
        assert len(grid.buckets) == 1883
        mean, worst = grid.occupancy()
        assert mean == pytest.approx(10.6213, abs=1e-4)
        assert worst == 86
        queries = uniform(300, random.Random(393))
        curve = cost_curve(points, queries, 10.0, [2.5, 10.0, 40.0])
        assert [round(curve[c][1], 1) for c in (2.5, 10.0, 40.0)] == [9.2, 16.3, 63.0]
        assert curve[2.5][0] == pytest.approx(5.71, abs=0.01)


class TestRefusals:
    def test_bad_cells_radii_and_costs(self):
        with pytest.raises(Invalid):
            SpatialHash(0)
        with pytest.raises(Invalid):
            SpatialHash(1.0).occupancy()
        with pytest.raises(Invalid):
            SpatialHash(1.0).within((0, 0), -1)
        with pytest.raises(Invalid):
            query_cost(SpatialHash(1.0), [], 1.0)
        with pytest.raises(Invalid):
            best_cell({1.0: (1.0, 1.0, 1.0)}, -1)
        with pytest.raises(Invalid):
            cheapest_cell(0.02, 1.0, 0.0, [])
