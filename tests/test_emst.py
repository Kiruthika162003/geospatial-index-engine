from __future__ import annotations

import random

import pytest

from atlas.emst import all_pairs_tree, minimum_spanning_tree, total_length
from atlas.errors import Invalid
from atlas.gabriel import delaunay_edges, is_connected, relative_neighborhood_edges


def _random_sets(seed, count=150):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(3, 60)
        yield list({(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)})


class TestTheTheorem:
    def test_the_delaunay_restricted_tree_has_the_all_pairs_length(self):
        for pts in _random_sets(127):
            tree = minimum_spanning_tree(pts)
            assert total_length(tree) == pytest.approx(total_length(all_pairs_tree(pts)))

    def test_the_tree_has_n_minus_one_edges_and_connects_everything(self):
        for pts in _random_sets(127):
            tree = minimum_spanning_tree(pts)
            assert len(tree) == len(pts) - 1
            edges = {(a, b) if a <= b else (b, a) for a, b in tree}
            assert is_connected(pts, edges)

    def test_the_tree_sits_inside_the_relative_neighborhood_graph(self):
        for pts in _random_sets(127):
            edges = {(a, b) if a <= b else (b, a) for a, b in minimum_spanning_tree(pts)}
            assert edges <= relative_neighborhood_edges(pts)


class TestTheSaving:
    def test_delaunay_candidates_are_a_small_fraction_of_all_pairs(self):
        delaunay_total = pairs_total = 0
        for pts in _random_sets(127):
            n = len(pts)
            delaunay_total += len(delaunay_edges(pts))
            pairs_total += n * (n - 1) // 2
        assert delaunay_total / pairs_total == pytest.approx(0.129, abs=0.02)


class TestEdges:
    def test_two_points_make_one_edge(self):
        tree = minimum_spanning_tree([(0, 0), (3, 4)])
        assert tree == [((0, 0), (3, 4))]
        assert total_length(tree) == pytest.approx(5.0)


class TestRefusals:
    def test_fewer_than_two_distinct_points_is_refused(self):
        with pytest.raises(Invalid):
            minimum_spanning_tree([(1, 1), (1, 1)])
        with pytest.raises(Invalid):
            all_pairs_tree([(1, 1)])
