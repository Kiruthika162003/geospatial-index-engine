from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.rangetree import RangeTree


class TestQueries:
    def test_a_concrete_count_and_report(self):
        pts = [(1, 1), (2, 5), (5, 2), (7, 7), (3, 3)]
        t = RangeTree(pts)
        box = BBox(0, 0, 4, 4)
        assert t.count(box) == 2
        assert sorted(t.report(box)) == [(1, 1), (3, 3)]

    def test_count_and_report_match_brute(self):
        rng = random.Random(79)
        for _ in range(2000):
            n = rng.randint(1, 80)
            pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
            t = RangeTree(pts)
            x0, y0 = rng.uniform(0, 50), rng.uniform(0, 50)
            x1, y1 = rng.uniform(50, 100), rng.uniform(50, 100)
            box = BBox(x0, y0, x1, y1)
            brute = [p for p in pts if box.contains_point(*p)]
            assert t.count(box) == len(brute)
            assert sorted(t.report(box)) == sorted(brute)

    def test_an_empty_tree_counts_zero(self):
        assert RangeTree([]).count(BBox(0, 0, 1, 1)) == 0
        assert len(RangeTree([])) == 0


class TestCanonicalDecomposition:
    def test_a_rectangle_touches_order_log_n_canonical_nodes(self):
        rng = random.Random(1)
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(8192)]
        t = RangeTree(pts)
        counts = []
        for _ in range(300):
            x0, y0 = rng.uniform(0, 400), rng.uniform(0, 400)
            x1, y1 = rng.uniform(600, 1000), rng.uniform(600, 1000)
            t.count(BBox(x0, y0, x1, y1))
            counts.append(t.canonical)
        # measured mean 12.6, max 19; the bound is about 2*log2(8192) = 26
        assert max(counts) <= 26
        assert sum(counts) / len(counts) < 20


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            RangeTree(None)
