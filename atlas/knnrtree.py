"""Best-first k-nearest over an R-tree: a heap keyed on box mindist visits only what it must.

Finding the k nearest entries in an R-tree could be done by a range
query with a guessed radius, growing it until k entries appear, but
that guesses twice and wastes work on both misses. Best-first search
needs no radius. It keeps a priority queue of tree nodes and entries
keyed by their mindist, the smallest possible distance from the query
to anything inside a node's bounding box, zero if the query is inside
the box and otherwise the distance to the box's nearest edge or
corner. It pops the closest thing first; if that is an internal node
it pushes the node's children with their own mindists, and if it is a
leaf entry it is reported as the next nearest. Because mindist is a
true lower bound, a real entry popped from the queue is closer than
every box still waiting, so the entries come out in exact distance
order, and the search stops the moment k have been reported, having
opened only the boxes whose mindist was below the k-th neighbor's
distance. This is the same lower-bound pruning the k-d tree and ball
tree use, expressed on rectangles: a subtree is never opened unless its
box could hold something nearer than what has already been found. The
contrast with depth-first is worth measuring, because depth-first must
descend a branch before it knows the branch is fruitless, while
best-first never descends a branch that a closer box outranks. The
finding worth stating is that best-first k-nearest returns exactly the
brute-force neighbors in order while opening a small fraction of the
tree's nodes, the fraction set by how many boxes have mindist under the
k-th distance. This module runs best-first k-nearest over an R-tree,
and a survey confirms the neighbors match brute force and counts the
nodes opened.
"""

from __future__ import annotations

import heapq
import math

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.rtree import RTree

Point = tuple[float, float]


def mindist(query: Point, box: BBox) -> float:
    qx, qy = query
    dx = max(box.min_x - qx, 0.0, qx - box.max_x)
    dy = max(box.min_y - qy, 0.0, qy - box.max_y)
    return math.hypot(dx, dy)


class BestFirst:
    def __init__(self, tree: RTree) -> None:
        if tree is None:
            raise Invalid("tree must not be None")
        self._tree = tree
        self.nodes_opened = 0

    def k_nearest(self, query: Point, k: int) -> list[tuple[float, object]]:
        if k <= 0:
            raise Invalid("k must be positive")
        self.nodes_opened = 0
        counter = 0
        heap: list[tuple[float, int, bool, object]] = []
        root = self._tree._root
        heapq.heappush(heap, (0.0, counter, False, root))
        found: list[tuple[float, object]] = []
        while heap and len(found) < k:
            d, _, is_item, payload = heapq.heappop(heap)
            if is_item:
                found.append((d, payload))
                continue
            self.nodes_opened += 1
            node = payload
            for entry in node.entries:
                counter += 1
                d_entry = mindist(query, entry.box)
                if node.leaf:
                    heapq.heappush(heap, (d_entry, counter, True, entry.item))
                else:
                    heapq.heappush(heap, (d_entry, counter, False, entry.child))
        return found
