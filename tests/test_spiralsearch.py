from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.spiralsearch import (
    GridIndex,
    brute_nearest,
    error_rate,
    expected_first_ring,
    uniform,
)


@pytest.fixture(scope="module")
def scene():
    return uniform(2000, random.Random(790)), uniform(1000, random.Random(791))


class TestStopRules:
    @pytest.mark.parametrize(
        ("cell", "wrong", "naive_rings", "safe_rings", "law"),
        [
            (5.0, 0.046, 1.949, 2.721, 1.982),
            (20.0, 0.147, 0.428, 1.082, 0.45),
            (40.0, 0.311, 0.034, 1.0, 0.041),
            (80.0, 0.181, 0.0, 1.0, 0.0),
        ],
    )
    def test_the_naive_rule_errs_and_the_safe_rule_never_does(
        self, scene, cell, wrong, naive_rings, safe_rings, law
    ):
        points, queries = scene
        read = error_rate(points, cell, queries)
        assert read["naive_wrong"] == pytest.approx(wrong, abs=1e-4)
        assert read["safe_wrong"] == 0.0
        assert read["naive_rings"] == pytest.approx(naive_rings, abs=1e-3)
        assert read["safe_rings"] == pytest.approx(safe_rings, abs=1e-3)
        assert expected_first_ring(0.002, cell) == pytest.approx(law, abs=1e-3)
        assert read["safe_rings"] >= 1.0

    def test_the_safe_rule_matches_brute_force(self, scene):
        points, queries = scene
        index = GridIndex(points, 40.0)
        for q in queries[:200]:
            assert index.nearest_safe_stop(q)[0] == brute_nearest(points, q)


class TestPieces:
    def test_rings_empties_and_refusals(self):
        index = GridIndex([(1.0, 1.0)], 20.0)
        assert [len(index.ring((5, 5), k)) for k in range(5)] == [1, 8, 16, 24, 32]
        assert len(set(index.ring((0, 0), 3))) == 24
        empty = GridIndex([], 10.0)
        assert empty.nearest_naive_stop((0.0, 0.0), 3) == (None, 3)
        assert empty.nearest_safe_stop((0.0, 0.0), 3) == (None, 3)
        with pytest.raises(Invalid):
            GridIndex([], 0.0)
        with pytest.raises(Invalid):
            error_rate([(0.0, 0.0)], 1.0, [])
