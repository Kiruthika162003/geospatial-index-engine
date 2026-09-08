"""Sort-Tile-Recursive: bulk-load a static R-tree with far less overlap than inserting.

Building an R-tree by inserting entries one at a time, choosing a leaf
and splitting on overflow, gives a usable tree but a loose one: the
incremental choices leave sibling boxes overlapping more than
necessary, and overlap is what slows queries. When all the data is
known up front and the tree will not change, a bulk-loading method does
much better, and Sort-Tile-Recursive is the standard one. It packs the
leaves by a tidy geometric recipe. Decide how many leaves are needed,
the entry count divided by the leaf capacity, and let the number of
vertical slices be the square root of that. Sort all entries by the x
of their center and cut them into that many vertical slices of equal
count. Within each slice, sort by the y of the center and cut into
leaves of the capacity. The effect is a grid of tiles, each tile a
leaf holding entries that are close in both x and y, so the leaf boxes
are compact and, being laid out on a grid, overlap far less than the
insertion tree's boxes. The same recipe then packs those leaf boxes
into the next level up, and so on to the root, giving a fully balanced
tree in one sorting pass per level. The tradeoff is stated plainly: STR
builds a better tree but a static one, since it assumes all entries are
present, so it suits read-mostly data, a loaded map that is queried but
not updated, while incremental insertion suits data that changes. The
finding worth stating is that STR's grid packing yields dramatically
less sibling overlap than one-at-a-time insertion on the same points,
which is the measurable reason bulk-loaded trees query faster. This
module bulk-loads the leaf level by STR and reports its sibling
overlap, and a survey compares that overlap against an insertion-built
R-tree on identical points.
"""

from __future__ import annotations

import math

from atlas.bbox import BBox
from atlas.errors import Invalid


def pack_leaves(boxes: list[BBox], capacity: int = 8) -> list[BBox]:
    if boxes is None:
        raise Invalid("boxes must not be None")
    if capacity <= 0:
        raise Invalid("capacity must be positive")
    if not boxes:
        return []
    leaf_count = math.ceil(len(boxes) / capacity)
    slices = max(1, math.ceil(math.sqrt(leaf_count)))
    per_slice = slices * capacity
    by_x = sorted(boxes, key=lambda b: b.center()[0])
    leaves: list[BBox] = []
    for s in range(0, len(by_x), per_slice):
        slice_boxes = by_x[s : s + per_slice]
        slice_boxes.sort(key=lambda b: b.center()[1])
        for t in range(0, len(slice_boxes), capacity):
            group = slice_boxes[t : t + capacity]
            leaves.append(_cover(group))
    return leaves


def sibling_overlap(leaves: list[BBox]) -> float:
    total = 0.0
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            total += leaves[i].intersection_area(leaves[j])
    return total


def _cover(boxes: list[BBox]) -> BBox:
    box = boxes[0]
    for other in boxes[1:]:
        box = box.union(other)
    return box
