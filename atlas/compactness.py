"""Compactness: scores that say how much a district's shape resembles a circle.

Whether a boundary is drawn to follow a community or to capture
one is partly a question of shape, and the shape measures used to
argue about it are simple ratios that a survey can calibrate on
known figures. Polsby-Popper is four pi times the area over the
perimeter squared, one for a circle and falling as the boundary
wanders. Schwartzberg is the ratio of the perimeter of a circle of
the same area to the shape's perimeter, the square root of
Polsby-Popper, so the two always agree on ordering. Reock is the
area over the area of the smallest enclosing circle, which ignores
boundary wiggle entirely and punishes elongation. The convex hull
ratio is the area over the area of the convex hull, which ignores
elongation entirely and punishes indentation. The survey calibrates
each on a circle, a square, a 10-to-1 rectangle, and a comb, a
square with deep teeth cut into one side, since the measures are
only useful if their disagreements are understood. The calibration
shows they measure different things: the square scores 0.785 on
Polsby-Popper and 0.637 on Reock but 1.0 on the hull ratio, since a
square is convex; the 10-to-1 rectangle scores 0.260 on
Polsby-Popper and 0.126 on Reock but again 1.0 on the hull ratio.
The comb was guessed wrong twice. With four teeth cut to depth 0.8
its area is 0.6 of the square's, not half, and its perimeter 10.4
against the square's 4, so Polsby-Popper reads 0.070 rather than
the guessed 0.05 and the hull ratio reads 0.6 rather than 0.5; and
the guess that Reock would barely notice was wrong in a precise
way, since the enclosing circle is unchanged and Reock therefore
falls by exactly the area fraction, 0.637 times 0.6 = 0.382. Eight
teeth at depth 0.9 push the three to 0.020, 0.350, and 0.55. A
circle approximated by a regular polygon scores 0.948 on
Polsby-Popper with 8 sides, 0.987 with 16, and 0.997 with 32, the
first count past 0.99 on both Polsby-Popper and Reock being 32.
The finding worth stating is that no single score captures
compactness, since elongation and indentation are independent
failures caught by different ratios, so a claim about a district's
shape must say which ratio it means. This module computes the four
scores, and a survey calibrates them on known figures.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.boundingcircle import smallest_enclosing_circle
from atlas.errors import Degenerate, Invalid
from atlas.jarvismarch import convex_hull
from atlas.shoelace import signed_area

Point = tuple[float, float]


def perimeter(ring: Sequence[Point]) -> float:
    total = 0.0
    for i, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(i + 1) % len(ring)]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _area(ring: Sequence[Point]) -> float:
    if len(ring) < 3:
        raise Invalid("a shape needs at least three vertices")
    area = abs(signed_area(ring))
    if area == 0.0:
        raise Degenerate("the shape has no area")
    return area


def polsby_popper(ring: Sequence[Point]) -> float:
    return 4.0 * math.pi * _area(ring) / perimeter(ring) ** 2


def schwartzberg(ring: Sequence[Point]) -> float:
    # perimeter of the equal-area circle over the shape's perimeter
    return 2.0 * math.sqrt(math.pi * _area(ring)) / perimeter(ring)


def reock(ring: Sequence[Point]) -> float:
    _, radius = smallest_enclosing_circle(list(ring))
    return _area(ring) / (math.pi * radius * radius)


def convex_hull_ratio(ring: Sequence[Point]) -> float:
    hull = convex_hull(list(ring))
    return _area(ring) / abs(signed_area(hull))


def scores(ring: Sequence[Point]) -> dict[str, float]:
    return {
        "polsby_popper": polsby_popper(ring),
        "schwartzberg": schwartzberg(ring),
        "reock": reock(ring),
        "convex_hull_ratio": convex_hull_ratio(ring),
    }


def regular_polygon(sides: int, radius: float = 1.0) -> list[Point]:
    if sides < 3:
        raise Invalid("a polygon needs at least three sides")
    step = 2.0 * math.pi / sides
    return [(radius * math.cos(i * step), radius * math.sin(i * step)) for i in range(sides)]


def rectangle(width: float, height: float) -> list[Point]:
    return [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]


def comb(size: float, teeth: int, depth: float) -> list[Point]:
    # a square with teeth cut into its top edge: teeth slots of full depth, alternating
    if teeth < 1 or not 0 < depth < size:
        raise Invalid("a comb needs at least one tooth and a depth inside the square")
    pitch = size / (2 * teeth)
    ring: list[Point] = [(0.0, 0.0), (size, 0.0), (size, size)]
    x = size
    for _ in range(teeth):
        x -= pitch
        ring.append((x, size))
        ring.append((x, size - depth))
        x -= pitch
        ring.append((x, size - depth))
        ring.append((x, size))
    ring.pop()  # the last point is (0, size), the square's corner, appended next
    ring.append((0.0, size))
    return ring
