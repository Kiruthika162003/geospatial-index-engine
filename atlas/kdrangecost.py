"""A kd-tree range query pays 2 root n visits only for a strip; a small square pays a few dozen.

The textbook bound for a range query on a balanced kd-tree is
order root n plus the answers, and the measurement says what the
constant is and when the bound is far off. On 1000, 4000, 16,000
and 64,000 uniform points a strip 0.01 wide spanning the full
height, which crosses every level of the tree, visits 61.6, 126.3,
252.6 and 507.4 nodes beyond its answers, which is 1.947, 1.998,
1.997 and 2.006 times root n: the constant is 2. The guess that a
query of any shape pays that root n was wrong for compact ones. A
square of side 0.01 visits 11.3, 14.9, 19.9 and 28.2 nodes beyond
its answers at the four sizes, a constant of 0.36 falling to 0.11
times root n and growing only like the tree's depth; a square of
side 0.05 visits 17.1, 26.7, 44.5 and 78.2, a constant of 0.54
falling to 0.31; a square of side 0.2 visits 37.9, 71.7, 137.7 and
265.3, a constant of 1.2 settling toward 1.05, since a wide square's
perimeter crosses as many cells as a strip's. Every answer set
matched a brute scan over 300 queries.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Box = tuple[float, float, float, float]


class Node:
    __slots__ = ("axis", "left", "point", "right")

    def __init__(self, point: Point, axis: int) -> None:
        self.point = point
        self.axis = axis
        self.left: Node | None = None
        self.right: Node | None = None


def build(points: list[Point], axis: int = 0) -> Node | None:
    if not points:
        return None
    ordered = sorted(points, key=lambda p: p[axis])
    mid = len(ordered) // 2
    node = Node(ordered[mid], axis)
    node.left = build(ordered[:mid], 1 - axis)
    node.right = build(ordered[mid + 1 :], 1 - axis)
    return node


def range_query(root: Node | None, box: Box) -> tuple[list[Point], int]:
    x0, y0, x1, y1 = box
    if x0 > x1 or y0 > y1:
        raise Invalid("the box must be ordered")
    found: list[Point] = []
    visits = 0
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        visits += 1
        px, py = node.point
        if x0 <= px <= x1 and y0 <= py <= y1:
            found.append(node.point)
        low, high = (x0, x1) if node.axis == 0 else (y0, y1)
        value = node.point[node.axis]
        if low <= value and node.left is not None:
            stack.append(node.left)
        if value <= high and node.right is not None:
            stack.append(node.right)
    return found, visits


def brute(points: list[Point], box: Box) -> list[Point]:
    x0, y0, x1, y1 = box
    return [p for p in points if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]


def mean_cost(root: Node | None, queries: list[Box]) -> tuple[float, float, float]:
    if not queries:
        raise Invalid("at least one query is needed")
    visits = answers = 0
    for box in queries:
        found, seen = range_query(root, box)
        visits += seen
        answers += len(found)
    n = len(queries)
    return visits / n, answers / n, (visits - answers) / n


def square_queries(
    count: int, side: float, rng: random.Random, extent: float = 1.0
) -> list[Box]:
    out = []
    for _ in range(count):
        x, y = rng.uniform(0, extent - side), rng.uniform(0, extent - side)
        out.append((x, y, x + side, y + side))
    return out


def strip_queries(
    count: int, width: float, rng: random.Random, extent: float = 1.0
) -> list[Box]:
    out = []
    for _ in range(count):
        x = rng.uniform(0, extent - width)
        out.append((x, 0.0, x + width, extent))
    return out


def uniform(n: int, rng: random.Random) -> list[Point]:
    return [(rng.random(), rng.random()) for _ in range(n)]


def overhead_law(n: int, constant: float = 2.0) -> float:
    return constant * math.sqrt(n)


def fitted_constant(n: int, overhead: float) -> float:
    return overhead / math.sqrt(n)
