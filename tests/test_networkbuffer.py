from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.networkbuffer import (
    coverage_ratio,
    detour_law,
    disc_area,
    grid,
    interior_sources,
    manhattan_law,
    mean_coverage,
    network_area_estimate,
    reachable,
    thinned_town,
    town,
    within_disc,
)


@pytest.fixture(scope="module")
def scene():
    points, graph = town(600, 690)
    return points, graph, interior_sources(points, 30.0, 20, random.Random(691))


class TestDelaunay:
    @pytest.mark.parametrize(
        ("distance", "share"), [(5.0, 0.978), (10.0, 0.9048), (20.0, 0.8713), (30.0, 0.8718)]
    )
    def test_the_share_settles_at_the_detour_law(self, scene, distance, share):
        points, graph, sources = scene
        assert mean_coverage(points, graph, distance, sources) == pytest.approx(share, abs=1e-4)
        assert detour_law(1.06) == pytest.approx(0.89, abs=1e-3)
        if distance >= 20:
            assert abs(share - detour_law(1.06)) < 0.02

    @pytest.mark.parametrize(
        ("keep", "shares"), [(0.8, (0.769, 0.7609, 0.7714)), (0.6, (0.5674, 0.5481, 0.5728))]
    )
    def test_thinning_costs_more_than_a_little(self, scene, keep, shares):
        _, _, sources = scene
        points, graph = thinned_town(600, 690, keep)
        read = tuple(mean_coverage(points, graph, d, sources) for d in (10.0, 20.0, 30.0))
        assert read == pytest.approx(shares, abs=1e-4)

    def test_the_area_estimate(self, scene):
        points, graph, sources = scene
        assert network_area_estimate(points, graph, sources[0], 10.0, 100.0) == pytest.approx(
            316.7, abs=0.1
        )
        assert network_area_estimate(points, graph, sources[0], 20.0, 100.0) == pytest.approx(
            1166.7, abs=0.1
        )
        assert disc_area(20.0) == pytest.approx(1256.6, abs=0.1)


class TestGrid:
    @pytest.mark.parametrize(
        ("distance", "reached", "inside", "share"),
        [(10.0, 13, 13, 1.0), (20.0, 41, 49, 0.8367), (45.0, 181, 253, 0.7154)],
    )
    def test_the_diamond_in_the_disc(self, distance, reached, inside, share):
        points, graph = grid(21, 5.0)
        centre = 10 * 21 + 10
        assert len(reachable(points, graph, centre, distance)) == reached
        assert len(within_disc(points, centre, distance)) == inside
        assert coverage_ratio(points, graph, centre, distance) == pytest.approx(share, abs=1e-4)
        assert share > manhattan_law() == pytest.approx(2 / math.pi)


class TestRefusals:
    def test_bad_distances_sources_and_discs(self, scene):
        points, graph, _ = scene
        with pytest.raises(Invalid):
            reachable(points, graph, 0, -1.0)
        with pytest.raises(Invalid):
            mean_coverage(points, graph, 10.0, [])
        with pytest.raises(Invalid):
            interior_sources(points, 49.9, 500, random.Random(1))
