from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.rtree import RTree
from atlas.rtreebulk import pack_leaves, sibling_overlap


def _cover(boxes):
    b = boxes[0]
    for other in boxes[1:]:
        b = b.union(other)
    return b


def _insertion_leaf_boxes(node):
    if node.leaf:
        return [_cover([e.box for e in node.entries])]
    out = []
    for e in node.entries:
        out += _insertion_leaf_boxes(e.child)
    return out


class TestPacking:
    def test_the_leaf_count_is_near_optimal(self):
        boxes = [BBox(i, 0, i + 1, 1) for i in range(4000)]
        leaves = pack_leaves(boxes, capacity=8)
        # 4000 entries at capacity 8 need exactly 500 full leaves
        assert len(leaves) == 500

    def test_every_point_lands_in_some_leaf(self):
        rng = random.Random(51)
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(500)]
        boxes = [BBox(x, y, x + 1, y + 1) for x, y in pts]
        leaves = pack_leaves(boxes, capacity=8)
        assert all(any(le.contains_point(x, y) for le in leaves) for x, y in pts)


class TestOverlapVsInsertion:
    def test_str_overlap_is_a_tiny_fraction_of_insertion_overlap(self):
        rng = random.Random(51)
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(4000)]
        boxes = [BBox(x, y, x + 1, y + 1) for x, y in pts]

        str_overlap = sibling_overlap(pack_leaves(boxes, capacity=8))

        tree = RTree(max_entries=8)
        for i, b in enumerate(boxes):
            tree.insert(b, i)
        ins_overlap = sibling_overlap(_insertion_leaf_boxes(tree._root))

        # measured ~946x less; require at least 20x
        assert str_overlap < ins_overlap / 20


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            pack_leaves(None)

    def test_a_non_positive_capacity_is_refused(self):
        with pytest.raises(Invalid):
            pack_leaves([BBox(0, 0, 1, 1)], capacity=0)

    def test_empty_packs_to_nothing(self):
        assert pack_leaves([]) == []
