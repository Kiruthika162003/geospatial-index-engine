from __future__ import annotations

import math
import random

import pytest

from atlas.edgebands import BandedPolygon
from atlas.errors import Invalid


def _fine_circle(rng, n=2000):
    return [
        (
            10 * math.cos(2 * math.pi * i / n) + rng.uniform(-0.1, 0.1),
            10 * math.sin(2 * math.pi * i / n) + rng.uniform(-0.1, 0.1),
        )
        for i in range(n)
    ]


class TestVerdict:
    def test_it_matches_plain_ray_casting_everywhere(self):
        rng = random.Random(113)
        banded = BandedPolygon(_fine_circle(rng), bands=64)
        for _ in range(5000):
            q = (rng.uniform(-12, 12), rng.uniform(-12, 12))
            assert banded.contains(q) == banded.plain(q)

    def test_a_query_above_the_polygon_short_circuits(self):
        rng = random.Random(1)
        banded = BandedPolygon(_fine_circle(rng), bands=64)
        assert banded.contains((0, 50)) is False
        assert banded.edges_tested == 0


class TestEdgesTested:
    def test_a_fine_polygon_tests_a_small_fraction_of_its_edges(self):
        rng = random.Random(113)
        banded = BandedPolygon(_fine_circle(rng), bands=64)
        tested = []
        for _ in range(2000):
            q = (rng.uniform(-10, 10), rng.uniform(-10, 10))
            banded.contains(q)
            tested.append(banded.edges_tested)
        mean = sum(tested) / len(tested)
        assert mean < 0.05 * banded.edge_count()  # measured 1.9 percent

    def test_more_bands_test_fewer_edges(self):
        rng = random.Random(113)
        poly = _fine_circle(rng)
        means = []
        for bands in (4, 16, 64, 256):
            banded = BandedPolygon(poly, bands=bands)
            total = 0
            for _ in range(300):
                banded.contains((rng.uniform(-10, 10), rng.uniform(-10, 10)))
                total += banded.edges_tested
            means.append(total / 300)
        assert means == sorted(means, reverse=True)  # measured 500, 133, 36, 14

    def test_a_tall_coarse_polygon_gains_little(self):
        tall = BandedPolygon([(0, 0), (10, 0), (10, 100), (0, 100)], bands=64)
        tall.contains((5, 50))
        assert tall.edges_tested == 2  # the two tall sides span every band


class TestRefusals:
    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            BandedPolygon([(0, 0), (1, 1)])

    def test_a_non_positive_band_count_is_refused(self):
        with pytest.raises(Invalid):
            BandedPolygon([(0, 0), (1, 0), (0, 1)], bands=0)
