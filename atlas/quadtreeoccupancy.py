"""Quadtree leaves sit under half full at any capacity; 200 near-duplicates dig 24 levels down.

A point quadtree splits a leaf when it passes its capacity, so
the leaves are never full on average. On 4000 uniform points over
a 1000 square the mean fill reads 0.471, 0.474 and 0.44 of the
capacity at capacities 4, 8 and 16, with 2122, 1054 and 568 leaves,
13.8, 2.6 and 0.7 percent of them empty, and 2829, 1405 and 757
nodes; the guess that leaves run about three quarters full was
wrong at every capacity. The occupancy histogram at capacity 8 is a
broad hump, 27, 76, 169, 223, 207, 151, 103, 68 and 30 leaves
holding 0 to 8 points. The fill holds near 0.47 from 1000 to 16,000
points, the leaves number about the points over 3.8, the depth
reads 3, 5, 6 and 7 at 200, 1000, 4000 and 16,000 points, and with
64-byte nodes and 16-byte points the tree costs 38 to 42 bytes a
point, 2.4 times the points themselves.

Eight Gaussian clusters of spread 15 lift the empty share to 18.7,
9.2 and 8.5 percent at the three capacities and the depth to 10, 9
and 8, since a split in a crowded cell makes three near-empty
siblings; the fill falls to 0.42, 0.42 and 0.38. Two hundred points
within a thousandth of a unit of each other are the pathological
case: they dig the tree 24, 23 and 23 levels deep, leave 47.7, 54.2
and 72.7 percent of the leaves empty and a fill of 0.29, 0.21 and
0.14, because every level that fails to separate them splits again.
"""

from __future__ import annotations

import math
import random

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.quadtree import QuadTree

Point = tuple[float, float]


def build(points: list[Point], capacity: int, side: float = 1000.0) -> QuadTree:
    tree = QuadTree(BBox(0.0, 0.0, side, side), capacity)
    for p in points:
        tree.insert(p)
    return tree


def leaves(tree: QuadTree) -> list[tuple[int, int]]:
    # (depth, occupancy) for every leaf
    out = []
    stack = [(tree._root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.children is None:
            out.append((depth, len(node.points)))
        else:
            stack.extend((child, depth + 1) for child in node.children)
    return out


def node_count(tree: QuadTree) -> int:
    count = 0
    stack = [tree._root]
    while stack:
        node = stack.pop()
        count += 1
        if node.children is not None:
            stack.extend(node.children)
    return count


def occupancy(tree: QuadTree, capacity: int) -> dict[str, float]:
    found = leaves(tree)
    if not found:
        raise Invalid("the tree has no leaves")
    counts = [n for _, n in found]
    depths = [d for d, _ in found]
    return {
        "leaves": float(len(found)),
        "empty_fraction": sum(1 for n in counts if n == 0) / len(counts),
        "mean_fill": sum(counts) / (len(counts) * capacity),
        "max_depth": float(max(depths)),
        "mean_depth": sum(depths) / len(depths),
        "nodes": float(node_count(tree)),
    }


def histogram(tree: QuadTree) -> dict[int, int]:
    out: dict[int, int] = {}
    for _, n in leaves(tree):
        out[n] = out.get(n, 0) + 1
    return dict(sorted(out.items()))


def uniform(n: int, rng: random.Random, side: float = 1000.0) -> list[Point]:
    return [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]


def clustered(
    n: int, clusters: int, spread: float, rng: random.Random, side: float = 1000.0
) -> list[Point]:
    centres = [
        (rng.uniform(100, side - 100), rng.uniform(100, side - 100)) for _ in range(clusters)
    ]
    out = []
    for i in range(n):
        cx, cy = centres[i % clusters]
        x = min(max(rng.gauss(cx, spread), 0.0), side - 1e-9)
        y = min(max(rng.gauss(cy, spread), 0.0), side - 1e-9)
        out.append((x, y))
    return out


def near_duplicates(
    n: int, rng: random.Random, gap: float = 1e-3, side: float = 1000.0
) -> list[Point]:
    x, y = rng.uniform(100, side - 100), rng.uniform(100, side - 100)
    return [(x + rng.uniform(0, gap), y + rng.uniform(0, gap)) for _ in range(n)]


def balanced_leaves(n: int, capacity: int) -> int:
    # a perfect quadtree would need this many leaves at the fill it can reach
    return 4 ** math.ceil(math.log(n / capacity, 4)) if n > capacity else 1


def bytes_estimate(tree: QuadTree, node_bytes: int = 64, point_bytes: int = 16) -> int:
    return node_count(tree) * node_bytes + sum(n for _, n in leaves(tree)) * point_bytes
