"""Zero sibling overlap is not a fast tree: x-sorted strips touch 11.3 leaves a query.

Bulk loading an R-tree packs the leaves from a sorted order of the
entries, and the usual argument for Sort-Tile-Recursive is that its
tiles overlap less than a space-filling-curve order. The measurement
here on 2000 uniform points, 250 leaves of 8, and 300 square queries
of side 50 on a side of 1000 says something sharper. STR leaves have
overlap 0 and touch 2.71 leaves a query. Leaves cut from a single sort
by x also have overlap 0, since vertical strips do not meet, and touch
11.32 leaves a query, because each strip runs the full height and its
margin is 7.3 times STR's, 192,569 against 26,308. Overlap alone does
not predict the query cost; the margin does. The guess that the tile
model predicts STR's touches was high by 18 percent: ((50 + 63.2) /
63.2) squared gives 3.21 and the tree reads 2.71, since a leaf box is
the tight cover of its points and not the whole tile.

Hilbert packing reads overlap 104,791 and 3.04 touches, 12 percent
above STR; Morton packing reads overlap 922,822, nine times Hilbert's,
and 4.44 touches, 64 percent above STR, since the Z curve jumps across
the square at every quadrant boundary. Give every entry a box of side
10 and STR loses its zero: overlap 80,728, touches 3.11, against
Hilbert 270,187 and 3.55 and Morton 1,458,818 and 5.15.

The guess that STR always wins is refuted on non-uniform data. On
2000 points in 10 Gaussian clusters Hilbert touches 1.97 leaves to
STR's 2.06 with a smaller total area, 345,872 against 354,404, despite
an overlap of 43,361 against STR's 0. On points strung along 8 streets
Hilbert touches 2.13 to STR's 2.60, 18 percent fewer, and even Morton's
2.37 beats STR, because STR's vertical slices cut every horizontal
street into pieces. On a single diagonal all four orders give the same
250 leaves, touches 0.923.

Capacity 4, 8, 16 and 32 on a second uniform set reads STR 3.18, 2.63,
2.12 and 1.75 touches, Hilbert 3.42, 2.95, 2.54 and 2.24, Morton 4.83,
4.59, 4.08 and 3.54, and x-sorted 17.81, 11.41, 6.82 and 4.08. The
Hilbert curve order matters up to 8 bits: order 4 reads overlap 464,067
and 3.67 touches, order 6 123,661 and 3.02, order 8 104,204 and 2.95,
order 10 108,713 and 2.95, since a 16 by 16 grid puts eight points in
one cell and breaks the tie by input order.
"""

from __future__ import annotations

import math
import random

from atlas import hilbert, morton
from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.rtreebulk import pack_leaves as str_leaves

Stats = dict[str, float]
Extent = tuple[float, float, float, float]


def _extent(boxes: list[BBox]) -> Extent:
    xs = [b.center()[0] for b in boxes]
    ys = [b.center()[1] for b in boxes]
    return min(xs), min(ys), max(xs), max(ys)


def _quantize(box: BBox, extent: Extent, order: int) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = extent
    side = (1 << order) - 1
    cx, cy = box.center()
    width = max_x - min_x or 1.0
    height = max_y - min_y or 1.0
    qx = round((cx - min_x) / width * side)
    qy = round((cy - min_y) / height * side)
    return min(max(qx, 0), side), min(max(qy, 0), side)


def hilbert_key(box: BBox, extent: Extent, order: int = 10) -> int:
    qx, qy = _quantize(box, extent, order)
    return hilbert.encode(qx, qy, order)


def morton_key(box: BBox, extent: Extent, order: int = 10) -> int:
    qx, qy = _quantize(box, extent, order)
    return morton.encode(qx, qy)


def pack_by_key(boxes: list[BBox], keys: list[int], capacity: int) -> list[BBox]:
    if boxes is None or keys is None:
        raise Invalid("boxes and keys must not be None")
    if len(boxes) != len(keys):
        raise Invalid("one key per box")
    if capacity <= 0:
        raise Invalid("capacity must be positive")
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: keys[i])
    leaves = []
    for start in range(0, len(order), capacity):
        group = [boxes[i] for i in order[start : start + capacity]]
        cover = group[0]
        for other in group[1:]:
            cover = cover.union(other)
        leaves.append(cover)
    return leaves


def hilbert_leaves(boxes: list[BBox], capacity: int = 8, order: int = 10) -> list[BBox]:
    if not boxes:
        return []
    extent = _extent(boxes)
    return pack_by_key(boxes, [hilbert_key(b, extent, order) for b in boxes], capacity)


def morton_leaves(boxes: list[BBox], capacity: int = 8, order: int = 10) -> list[BBox]:
    if not boxes:
        return []
    extent = _extent(boxes)
    return pack_by_key(boxes, [morton_key(b, extent, order) for b in boxes], capacity)


def x_sorted_leaves(boxes: list[BBox], capacity: int = 8) -> list[BBox]:
    if not boxes:
        return []
    keys = sorted(range(len(boxes)), key=lambda i: boxes[i].center()[0])
    rank = [0] * len(boxes)
    for position, index in enumerate(keys):
        rank[index] = position
    return pack_by_key(boxes, rank, capacity)


def overlap(leaves: list[BBox]) -> float:
    total = 0.0
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            total += leaves[i].intersection_area(leaves[j])
    return total


def total_area(leaves: list[BBox]) -> float:
    return sum(leaf.area() for leaf in leaves)


def total_margin(leaves: list[BBox]) -> float:
    return sum(leaf.margin() for leaf in leaves)


def touches(leaves: list[BBox], queries: list[BBox]) -> float:
    if not queries:
        raise Invalid("at least one query is needed")
    hits = 0
    for query in queries:
        hits += sum(1 for leaf in leaves if leaf.intersects(query))
    return hits / len(queries)


def stats(leaves: list[BBox], queries: list[BBox]) -> Stats:
    return {
        "leaves": float(len(leaves)),
        "overlap": overlap(leaves),
        "area": total_area(leaves),
        "margin": total_margin(leaves),
        "touches": touches(leaves, queries),
    }


METHODS = ("str", "hilbert", "morton", "x_sorted")


def leaves_by(method: str, boxes: list[BBox], capacity: int = 8) -> list[BBox]:
    if method == "str":
        return str_leaves(boxes, capacity)
    if method == "hilbert":
        return hilbert_leaves(boxes, capacity)
    if method == "morton":
        return morton_leaves(boxes, capacity)
    if method == "x_sorted":
        return x_sorted_leaves(boxes, capacity)
    raise Invalid(f"unknown packing method {method!r}")


def compare(boxes: list[BBox], queries: list[BBox], capacity: int = 8) -> dict[str, Stats]:
    if not boxes:
        raise Invalid("boxes must not be empty")
    return {method: stats(leaves_by(method, boxes, capacity), queries) for method in METHODS}


def uniform_boxes(
    n: int, rng: random.Random, size: float = 0.0, side: float = 1000.0
) -> list[BBox]:
    boxes = []
    for _ in range(n):
        x, y = rng.uniform(0, side), rng.uniform(0, side)
        boxes.append(BBox(x, y, x + size, y + size))
    return boxes


def clustered_boxes(
    n: int, clusters: int, rng: random.Random, spread: float = 30.0, side: float = 1000.0
) -> list[BBox]:
    centers = [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(clusters)]
    boxes = []
    for i in range(n):
        cx, cy = centers[i % clusters]
        x, y = rng.gauss(cx, spread), rng.gauss(cy, spread)
        boxes.append(BBox(x, y, x, y))
    return boxes


def diagonal_boxes(n: int, side: float = 1000.0) -> list[BBox]:
    boxes = []
    for i in range(n):
        t = side * i / max(n - 1, 1)
        boxes.append(BBox(t, t, t, t))
    return boxes


def street_boxes(n: int, streets: int, rng: random.Random, side: float = 1000.0) -> list[BBox]:
    boxes = []
    for i in range(n):
        street = i % streets
        along = rng.uniform(0, side)
        across = side * (street + 0.5) / streets + rng.uniform(-2, 2)
        if street % 2 == 0:
            boxes.append(BBox(along, across, along, across))
        else:
            boxes.append(BBox(across, along, across, along))
    return boxes


def square_queries(n: int, rng: random.Random, size: float, side: float = 1000.0) -> list[BBox]:
    queries = []
    for _ in range(n):
        x, y = rng.uniform(0, side - size), rng.uniform(0, side - size)
        queries.append(BBox(x, y, x + size, y + size))
    return queries


def ideal_touches(n: int, capacity: int, query_size: float, side: float = 1000.0) -> float:
    leaves = math.ceil(n / capacity)
    tile = side / math.sqrt(leaves)
    return ((query_size + tile) / tile) ** 2
