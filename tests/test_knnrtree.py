from __future__ import annotations

import math
import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.knnrtree import BestFirst, mindist
from atlas.rtree import RTree


def _count_nodes(node):
    return 1 if node.leaf else 1 + sum(_count_nodes(e.child) for e in node.entries)


class TestMindist:
    def test_inside_the_box_is_zero(self):
        assert mindist((5, 5), BBox(0, 0, 10, 10)) == 0.0

    def test_outside_is_the_distance_to_the_nearest_corner(self):
        assert mindist((13, 14), BBox(0, 0, 10, 10)) == pytest.approx(5.0)

    def test_beside_an_edge_is_the_edge_gap(self):
        assert mindist((15, 5), BBox(0, 0, 10, 10)) == pytest.approx(5.0)


class TestKNearest:
    def test_it_matches_brute_force_in_distance_order(self):
        rng = random.Random(75)
        for _ in range(1500):
            n = rng.randint(1, 60)
            pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
            tree = RTree(max_entries=6)
            for i, (x, y) in enumerate(pts):
                tree.insert(BBox(x, y, x, y), i)
            q = (rng.uniform(0, 100), rng.uniform(0, 100))
            k = min(5, n)
            got = [d for d, _ in BestFirst(tree).k_nearest(q, k)]
            brute = sorted(math.hypot(px - q[0], py - q[1]) for px, py in pts)[:k]
            assert got == pytest.approx(brute)
            assert got == sorted(got)

    def test_it_opens_a_small_fraction_of_the_nodes(self):
        rng = random.Random(1)
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(5000)]
        tree = RTree(max_entries=8)
        for i, (x, y) in enumerate(pts):
            tree.insert(BBox(x, y, x, y), i)
        total = _count_nodes(tree._root)
        opened = []
        for _ in range(300):
            bf = BestFirst(tree)
            bf.k_nearest((rng.uniform(0, 1000), rng.uniform(0, 1000)), 10)
            opened.append(bf.nodes_opened)
        # measured ~24 of ~1043 nodes for k=10
        assert sum(opened) / len(opened) < total * 0.1


class TestRefusals:
    def test_a_none_tree_is_refused(self):
        with pytest.raises(Invalid):
            BestFirst(None)

    def test_a_non_positive_k_is_refused(self):
        tree = RTree()
        tree.insert(BBox(0, 0, 1, 1), "a")
        with pytest.raises(Invalid):
            BestFirst(tree).k_nearest((0, 0), 0)
