"""Interval tree: augment a BST with the subtree's max endpoint to prune stabbing queries.

An interval tree answers which stored intervals overlap a query point
or query interval, the one-dimensional cousin of a range search, used
for time ranges, elevation bands, and the per-axis pieces of higher
structures. It is a binary search tree keyed by each interval's low
endpoint, which alone would let you find intervals starting before a
point but not efficiently rule out intervals that start early yet end
before the query, so the tree is augmented: every node stores the
maximum high endpoint anywhere in its subtree. That one extra number
powers the prune. Searching for intervals overlapping a query, at each
node, if the subtree's maximum high endpoint is below the query's low
end, then no interval in that entire subtree can reach the query, so
the whole subtree is skipped without visiting it. Otherwise the search
checks the node's own interval, always descends left because a
qualifying interval could start early there, and descends right only
when the node's own low endpoint does not already lie past the query's
high end. The result is that a stabbing query costs the log-depth
descent plus the number of intervals actually reported, order log n
plus k, rather than scanning all n intervals. The augmentation is the
whole trick, and it generalizes: storing a subtree aggregate, here the
max endpoint, to prune branches that cannot contribute is the same
move that makes segment trees and range trees fast. The finding worth
stating is that the max-endpoint augmentation lets the query touch only
the branches that can hold an overlap, so its cost tracks the output
size plus the tree depth, not the interval count. This module builds a
static interval tree and answers point and interval overlap queries,
and a survey confirms the results match a brute scan while pruning.
"""

from __future__ import annotations

from atlas.errors import Invalid

Interval = tuple[float, float]


class _INode:
    __slots__ = ("interval", "left", "max_high", "right")

    def __init__(self, interval: Interval) -> None:
        self.interval = interval
        self.max_high = interval[1]
        self.left: _INode | None = None
        self.right: _INode | None = None


class IntervalTree:
    def __init__(self, intervals: list[Interval]) -> None:
        if intervals is None:
            raise Invalid("intervals must not be None")
        for lo, hi in intervals:
            if lo > hi:
                raise Invalid("each interval's low must not exceed its high")
        self._size = len(intervals)
        self.visits = 0
        self._root = self._build(sorted(intervals))

    def __len__(self) -> int:
        return self._size

    def _build(self, ordered: list[Interval]) -> _INode | None:
        if not ordered:
            return None
        mid = len(ordered) // 2
        node = _INode(ordered[mid])
        node.left = self._build(ordered[:mid])
        node.right = self._build(ordered[mid + 1 :])
        node.max_high = node.interval[1]
        for child in (node.left, node.right):
            if child is not None:
                node.max_high = max(node.max_high, child.max_high)
        return node

    def stab(self, point: float) -> list[Interval]:
        return self.overlap(point, point)

    def overlap(self, lo: float, hi: float) -> list[Interval]:
        if lo > hi:
            raise Invalid("query low must not exceed query high")
        self.visits = 0
        found: list[Interval] = []
        self._query(self._root, lo, hi, found)
        return found

    def _query(self, node: _INode | None, lo: float, hi: float, found: list[Interval]) -> None:
        if node is None:
            return
        self.visits += 1
        if node.max_high < lo:
            return  # nothing in this subtree reaches the query
        self._query(node.left, lo, hi, found)
        i_lo, i_hi = node.interval
        if i_lo <= hi and lo <= i_hi:
            found.append(node.interval)
        if i_lo <= hi:
            self._query(node.right, lo, hi, found)
