from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid, Missing
from atlas.octree import Octree, brute_nearest, brute_within


def _cube(n: int, rng) -> list[tuple[float, float, float]]:
    return [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]


def _shell(n: int, rng) -> list[tuple[float, float, float]]:
    out = []
    for _ in range(n):
        z = rng.uniform(-1, 1)
        t = rng.uniform(0, 2 * math.pi)
        r = math.sqrt(1 - z * z)
        out.append((r * math.cos(t), r * math.sin(t), z))
    return out


def _tree(points, capacity: int = 8) -> Octree:
    tree = Octree((0.0, 0.0, 0.0), 1.0, capacity=capacity)
    for p in points:
        tree.insert(p)
    return tree


def _mean_visits(tree: Octree, points, queries) -> float:
    total = 0
    for q in queries:
        _, d = tree.nearest(q)
        total += tree.visits
        assert d == pytest.approx(brute_nearest(points, q)[1], abs=1e-12)
    return total / len(queries)


class TestTheCube:
    def test_depth_is_log_eight_of_the_count(self):
        rng = random.Random(217)
        points = _cube(512, rng)
        tree = _tree(points)
        assert tree.depth() == 3 == round(math.log(512, 8))
        assert tree.node_count() == 313
        assert _mean_visits(tree, points, _cube(100, rng)) == pytest.approx(8.0, abs=0.1)
        points = _cube(4096, rng)
        tree = _tree(points)
        assert tree.depth() == 4 == round(math.log(4096, 8))
        assert 2000 < tree.node_count() < 2500
        assert 9.0 < _mean_visits(tree, points, _cube(100, rng)) < 12.0


class TestTheShell:
    def test_the_shell_runs_deeper_and_visits_depend_on_where_the_query_stands(self):
        rng = random.Random(217)
        points = _shell(4096, rng)
        tree = _tree(points)
        assert tree.depth() == 5
        on_shell = _mean_visits(tree, points, _shell(100, rng))
        interior = _mean_visits(tree, points, _cube(100, rng))
        assert on_shell < 12
        assert interior > 40
        assert interior / on_shell > 4


class TestRangeAndCapacity:
    def test_the_range_query_matches_brute_force(self):
        rng = random.Random(218)
        points = _cube(4096, rng)
        tree = _tree(points)
        for _ in range(30):
            q = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            assert sorted(tree.within(q, 0.2)) == sorted(brute_within(points, q, 0.2))

    def test_a_larger_leaf_cuts_nodes_fourfold_and_visits_by_a_third(self):
        rng = random.Random(219)
        points = _cube(4096, rng)
        small, large = _tree(points, 8), _tree(points, 32)
        assert small.depth() == 4
        assert large.depth() == 3
        assert small.node_count() / large.node_count() > 3.5
        queries = _cube(100, rng)
        assert _mean_visits(large, points, queries) < _mean_visits(small, points, queries)


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            Octree((0, 0, 0), 1.0).insert((2, 0, 0))
        with pytest.raises(Missing):
            Octree((0, 0, 0), 1.0).nearest((0, 0, 0))
        with pytest.raises(Invalid):
            Octree((0, 0, 0), 0)
        with pytest.raises(Invalid):
            Octree((0, 0, 0), 1.0).within((0, 0, 0), -1)
        with pytest.raises(Missing):
            brute_nearest([], (0, 0, 0))
