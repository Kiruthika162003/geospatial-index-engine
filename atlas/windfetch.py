"""A 400-cell channel 11 cells wide has an effective fetch of 45, eleven percent of its length.

Fetch is the distance wind travels over water before reaching a
point, read here along a rounded ray over a water mask until land or
the grid edge. The Shore Protection Manual's effective fetch averages
the radials within 45 degrees of the wind, every 6 degrees, weighted
by the cosine of their offset, and it is that average that separates
a lake from a channel. From the west shore of a round lake of radius
90 the straight fetch east reads 180 and the effective fetch 159.11,
0.884 of it, against the chord law sum of 2R cos squared over sum of
cos of 159.75. The rose at that shore reads 0 landward and 115, 137,
155, 168, 177, 180 from 50 to 90 degrees, the chords 2R cos theta.

The guess that a long channel's fetch is its length held only for
the single radial: a channel 400 long and 5, 11, 21 and 41 wide
reads a straight fetch of 400 and effective fetches of 34.95, 44.68,
61.29 and 94.63, 8.7 to 23.7 percent of the length, within 1.2
percent of the law that clips each oblique radial at half the width over the
sine of its offset. A bay behind a wall with a 21-cell mouth reads a
straight fetch of 100 through the mouth and 73.5 effective on the
axis, and off the axis at row 30 the straight fetch drops to 59 while
the effective fetch reads 70.4, since the oblique radials still find
the mouth. The radial spacing matters on the channel and not on the
lake: a step of 15 degrees reads the lake at 158.3 against 159.1 and
the 11-wide channel at 76.96 against 44.68, 72 percent high, because
seven radials miss the near-axis reaches that thirty-one resolve. A
spread of 90 degrees reads the lake at 130.3 and 30 degrees at 170.1.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Mask = list[list[bool]]


def _check(water: Mask) -> tuple[int, int]:
    if not water or not water[0]:
        raise Invalid("the water mask is empty")
    return len(water), len(water[0])


def fetch(
    water: Mask, r: int, c: int, angle: float, cell: float = 1.0, max_steps: int = 10_000
) -> float:
    rows, cols = _check(water)
    if not (0 <= r < rows and 0 <= c < cols):
        raise Invalid("the point must lie on the grid")
    if cell <= 0:
        raise Invalid("the cell size must be positive")
    dr, dc = -math.cos(angle), math.sin(angle)
    last = 0.0
    for k in range(1, max_steps + 1):
        rr, cc = round(r + k * dr), round(c + k * dc)
        if not (0 <= rr < rows and 0 <= cc < cols) or not water[rr][cc]:
            break
        last = k * cell
    return last


def fetch_rose(
    water: Mask, r: int, c: int, directions: int = 36, cell: float = 1.0
) -> list[float]:
    if directions < 1:
        raise Invalid("at least one direction is needed")
    return [fetch(water, r, c, 2 * math.pi * k / directions, cell) for k in range(directions)]


def effective_fetch(
    water: Mask,
    r: int,
    c: int,
    angle: float,
    cell: float = 1.0,
    spread: float = 45.0,
    step: float = 6.0,
) -> float:
    if spread <= 0 or step <= 0 or step > spread:
        raise Invalid("spread and step must be positive with step at most the spread")
    count = round(spread / step)
    weighted = 0.0
    weights = 0.0
    for k in range(-count, count + 1):
        offset = math.radians(k * step)
        w = math.cos(offset)
        weighted += w * fetch(water, r, c, angle + offset, cell)
        weights += w
    return weighted / weights


def longest_fetch(
    water: Mask, r: int, c: int, directions: int = 36, cell: float = 1.0
) -> tuple[float, float]:
    rose = fetch_rose(water, r, c, directions, cell)
    best = max(range(directions), key=lambda k: rose[k])
    return rose[best], math.degrees(2 * math.pi * best / directions)


def lake(size: int, radius: float) -> Mask:
    if size < 3 or radius <= 0:
        raise Invalid("a lake needs three cells a side and a positive radius")
    centre = size / 2 - 0.5
    return [
        [math.hypot(r - centre, c - centre) <= radius for c in range(size)] for r in range(size)
    ]


def channel(rows: int, cols: int, width: int) -> Mask:
    if width < 1 or width > rows:
        raise Invalid("the channel must fit the grid")
    top = (rows - width) // 2
    return [[top <= r < top + width for c in range(cols)] for r in range(rows)]


def bay(size: int, mouth: int, wall_col: int) -> Mask:
    if not 0 < mouth <= size or not 0 <= wall_col < size:
        raise Invalid("the mouth must fit the shore and the wall the grid")
    start = (size - mouth) // 2
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            row.append(not (c == wall_col and not start <= r < start + mouth))
        out.append(row)
    return out


def shore_point(water: Mask, from_row: int, from_col: int, angle: float) -> tuple[int, int]:
    rows, cols = _check(water)
    dr, dc = -math.cos(angle), math.sin(angle)
    r, c = from_row, from_col
    if not water[r][c]:
        raise Invalid("start on water")
    for k in range(1, 10_000):
        rr, cc = round(from_row + k * dr), round(from_col + k * dc)
        if not (0 <= rr < rows and 0 <= cc < cols) or not water[rr][cc]:
            return r, c
        r, c = rr, cc
    raise Invalid("no shore found")


def circle_law(radius: float, spread: float = 45.0, step: float = 6.0) -> float:
    count = round(spread / step)
    weighted = weights = 0.0
    for k in range(-count, count + 1):
        offset = math.radians(k * step)
        weighted += math.cos(offset) * 2 * radius * math.cos(offset)
        weights += math.cos(offset)
    return weighted / weights


def channel_law(length: float, width: float, spread: float = 45.0, step: float = 6.0) -> float:
    count = round(spread / step)
    weighted = weights = 0.0
    for k in range(-count, count + 1):
        offset = math.radians(k * step)
        reach = length if offset == 0 else min(length, (width / 2) / abs(math.sin(offset)))
        weighted += math.cos(offset) * reach
        weights += math.cos(offset)
    return weighted / weights
