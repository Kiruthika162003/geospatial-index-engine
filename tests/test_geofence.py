from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.geofence import Geofence


def _regular(cx, cy, r, k=8):
    angles = [2 * math.pi * i / k for i in range(k)]
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]


def _fence(rng, count, radius):
    g = Geofence()
    for i in range(count):
        g.add(f"r{i}", _regular(rng.uniform(0, 100), rng.uniform(0, 100), radius))
    return g


class TestVerdict:
    def test_the_prefilter_never_changes_the_answer(self):
        rng = random.Random(149)
        for count, radius in ((4, 40), (50, 8), (400, 2)):
            g = _fence(rng, count, radius)
            for _ in range(500):
                p = (rng.uniform(0, 100), rng.uniform(0, 100))
                assert sorted(g.containing(p)) == sorted(g.containing_exact(p))

    def test_nested_zones_are_both_reported(self):
        g = Geofence()
        g.add("city", _regular(50, 50, 40))
        g.add("downtown", _regular(50, 50, 10))
        assert g.containing((50, 50)) == ["city", "downtown"]
        assert g.containing((85, 50)) == ["city"]
        assert g.containing((99, 99)) == []
        assert len(g) == 2


class TestPrefilter:
    def test_smaller_regions_reach_the_exact_test_less_often(self):
        rng = random.Random(150)
        rates = []
        for count, radius in ((4, 40), (50, 8), (400, 2)):
            g = _fence(rng, count, radius)
            total = 0
            for _ in range(500):
                g.containing((rng.uniform(0, 100), rng.uniform(0, 100)))
                total += g.exact_tests
            rates.append(total / 500 / count)
        # measured about 35, 2.4, and 0.2 percent of the regions
        assert rates[0] > 0.2
        assert rates[1] < 0.05
        assert rates[2] < 0.01
        assert rates == sorted(rates, reverse=True)


class TestRefusals:
    def test_a_nameless_region_is_refused(self):
        with pytest.raises(Invalid):
            Geofence().add("", _regular(0, 0, 1))

    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            Geofence().add("tiny", [(0, 0), (1, 1)])
