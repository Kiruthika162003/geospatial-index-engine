"""R-tree: a balanced tree of bounding boxes where sibling overlap is the enemy.

The R-tree indexes rectangles, or points seen as tiny rectangles, in a
height-balanced tree that groups nearby entries under a shared bounding
box, so a query descends only into boxes it intersects. Unlike the
k-d tree and quadtree, which partition space so a point falls in
exactly one region, the R-tree's sibling boxes are allowed to overlap,
and that overlap is the whole difficulty of the structure. When two
sibling boxes overlap, a point query landing in the overlap must
descend into both, so overlap directly multiplies the work a query
does; a perfect R-tree would have zero sibling overlap and a query
would follow one root-to-leaf path, while a bad one degenerates toward
scanning everything. Insertion is a running fight against overlap.
Choosing where to put a new entry follows least enlargement, the child
whose box must grow the least to include it, which tends to keep boxes
tight. When a node overflows it splits, and the split quality is what
matters: Guttman's quadratic split first picks the two entries that
would waste the most space if placed together, the seeds, sends them
to opposite groups, then repeatedly assigns the remaining entry with
the strongest preference for one group, keeping the two resulting boxes
as small and separate as it can. The finding worth stating is that the
R-tree trades the clean space partition of the other trees for the
ability to bound arbitrary rectangles, and pays for it in sibling
overlap, which a good split minimizes but never fully removes, so query
cost is governed by how much the boxes overlap rather than by depth
alone. This module builds an R-tree with least-enlargement insertion
and quadratic split, answers range and point queries, and a survey
measures the sibling overlap and confirms queries match a brute scan.
"""

from __future__ import annotations

from atlas.bbox import BBox
from atlas.errors import Invalid

Point = tuple[float, float]


class _Entry:
    __slots__ = ("box", "child", "item")

    def __init__(self, box: BBox, child: _RNode | None = None, item: object = None) -> None:
        self.box = box
        self.child = child
        self.item = item


class _RNode:
    __slots__ = ("entries", "leaf")

    def __init__(self, leaf: bool) -> None:
        self.entries: list[_Entry] = []
        self.leaf = leaf


class RTree:
    def __init__(self, max_entries: int = 8) -> None:
        if max_entries < 4:
            raise Invalid("max_entries must be at least 4")
        self._max = max_entries
        self._min = max(2, max_entries // 2)
        self._root = _RNode(leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, box: BBox, item: object) -> None:
        if box is None:
            raise Invalid("box must not be None")
        entry = _Entry(box, item=item)
        split = self._insert(self._root, entry, self._height())
        if split is not None:
            left, right = split
            new_root = _RNode(leaf=False)
            new_root.entries = [
                _Entry(_cover(left.entries), child=left),
                _Entry(_cover(right.entries), child=right),
            ]
            self._root = new_root
        self._size += 1

    def _height(self) -> int:
        node = self._root
        h = 0
        while not node.leaf:
            node = node.entries[0].child
            h += 1
        return h

    def _insert(self, node: _RNode, entry: _Entry, depth: int) -> tuple[_RNode, _RNode] | None:
        if node.leaf:
            node.entries.append(entry)
        else:
            chosen = self._choose(node, entry.box)
            split = self._insert(chosen.child, entry, depth - 1)
            chosen.box = _cover(chosen.child.entries)
            if split is not None:
                left, right = split
                node.entries = [e for e in node.entries if e is not chosen]
                node.entries.append(_Entry(_cover(left.entries), child=left))
                node.entries.append(_Entry(_cover(right.entries), child=right))
        if len(node.entries) > self._max:
            return self._split(node)
        return None

    def _choose(self, node: _RNode, box: BBox) -> _Entry:
        best = None
        best_key = None
        for entry in node.entries:
            enlargement = entry.box.enlargement(box)
            key = (enlargement, entry.box.area())
            if best_key is None or key < best_key:
                best_key = key
                best = entry
        return best

    def _split(self, node: _RNode) -> tuple[_RNode, _RNode]:
        entries = node.entries
        i, j = self._pick_seeds(entries)
        left = _RNode(leaf=node.leaf)
        right = _RNode(leaf=node.leaf)
        left.entries.append(entries[i])
        right.entries.append(entries[j])
        remaining = [e for k, e in enumerate(entries) if k not in (i, j)]
        left_box = _cover(left.entries)
        right_box = _cover(right.entries)
        while remaining:
            # if one group must take all the rest to meet the minimum, give them
            if len(left.entries) + len(remaining) == self._min:
                left.entries.extend(remaining)
                break
            if len(right.entries) + len(remaining) == self._min:
                right.entries.extend(remaining)
                break
            entry = remaining.pop()
            d_left = left_box.enlargement(entry.box)
            d_right = right_box.enlargement(entry.box)
            if d_left < d_right or (d_left == d_right and left_box.area() <= right_box.area()):
                left.entries.append(entry)
                left_box = left_box.union(entry.box)
            else:
                right.entries.append(entry)
                right_box = right_box.union(entry.box)
        return left, right

    def _pick_seeds(self, entries: list[_Entry]) -> tuple[int, int]:
        worst = None
        pair = (0, 1)
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                combined = entries[a].box.union(entries[b].box)
                waste = combined.area() - entries[a].box.area() - entries[b].box.area()
                if worst is None or waste > worst:
                    worst = waste
                    pair = (a, b)
        return pair

    def query(self, box: BBox) -> list[object]:
        found: list[object] = []
        self._query(self._root, box, found)
        return found

    def _query(self, node: _RNode, box: BBox, found: list[object]) -> None:
        for entry in node.entries:
            if entry.box.intersects(box):
                if node.leaf:
                    found.append(entry.item)
                else:
                    self._query(entry.child, box, found)

    def sibling_overlap(self) -> float:
        return self._overlap(self._root)

    def _overlap(self, node: _RNode) -> float:
        total = 0.0
        if not node.leaf:
            for a in range(len(node.entries)):
                for b in range(a + 1, len(node.entries)):
                    total += node.entries[a].box.intersection_area(node.entries[b].box)
            for entry in node.entries:
                total += self._overlap(entry.child)
        return total


def _cover(entries: list[_Entry]) -> BBox:
    box = entries[0].box
    for entry in entries[1:]:
        box = box.union(entry.box)
    return box
