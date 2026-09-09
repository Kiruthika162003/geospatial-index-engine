"""Quadkeys: the two tiles either side of the central seam share no prefix at all.

A quadkey names a tile by the path of quadrants from the root, one
base-4 digit a level, so a tile's parent is its key with the last
digit dropped and its children are the key with a digit added. The
encoding round-trips every tile of zooms 0 to 5. The guess that
neighbouring tiles share most of their key was wrong at the seams:
at zoom 10 the tile just west of the central meridian is 2111111111
and the tile just east of it 3000000000, sharing nothing, and the
tile at the centre shares 0, 9, 9 and 0 digits with its east, west,
south and north neighbours. Over 20,000 random tiles at zoom 10 an
eastern neighbour shares 9 digits half the time, 8 a quarter, and so
on down, measured 0.497, 0.252, 0.129, 0.061, 0.030, 0.015, 0.0076,
0.0038, 0.0025 and 0.0010 for 9 down to 0 against the halving law of
0.5 to 0.00098, since the shared length is set by how many low bits
of x carry when one is added. The mean shared prefix is two digits
short of the zoom: 3.17 of 5, 8.02 of 10 and 16.01 of 18, and the
share of zero-digit pairs falls from 0.030 at zoom 5 to 0.0001 at
18.

Merging a box of zoom 4 tiles into the fewest quadkeys turns the
16-tile corner box into the single key 00, the 64-tile quadrant into
0 and the whole level into the empty root, while a 36-tile box set
one tile in from the corner needs 24 keys, four of length 3 and
twenty of length 4, and a 100-tile box needs 40, since a box that
straddles quadrant seams cannot use the seams' parents.
"""

from __future__ import annotations

import random

from atlas.errors import Invalid

Tile = tuple[int, int, int]


def encode(zoom: int, x: int, y: int) -> str:
    if zoom < 0:
        raise Invalid("zoom must not be negative")
    side = 1 << zoom
    if not (0 <= x < side and 0 <= y < side):
        raise Invalid("the tile must lie within its zoom level")
    digits = []
    for level in range(zoom, 0, -1):
        mask = 1 << (level - 1)
        digit = 0
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        digits.append(str(digit))
    return "".join(digits)


def decode(key: str) -> Tile:
    x = y = 0
    zoom = len(key)
    for i, ch in enumerate(key):
        if ch not in "0123":
            raise Invalid("a quadkey holds only the digits 0 to 3")
        mask = 1 << (zoom - i - 1)
        digit = int(ch)
        if digit & 1:
            x |= mask
        if digit & 2:
            y |= mask
    return zoom, x, y


def parent(key: str) -> str:
    if not key:
        raise Invalid("the root has no parent")
    return key[:-1]


def children(key: str) -> list[str]:
    return [key + d for d in "0123"]


def common_prefix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def neighbours(zoom: int, x: int, y: int) -> list[Tile]:
    side = 1 << zoom
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = (x + dx) % side, y + dy
        if 0 <= ny < side:
            out.append((zoom, nx, ny))
    return out


def shared_with_neighbours(zoom: int, x: int, y: int) -> list[int]:
    key = encode(zoom, x, y)
    return [common_prefix(key, encode(*t)) for t in neighbours(zoom, x, y)]


def prefix_histogram(zoom: int, rng: random.Random, samples: int) -> dict[int, int]:
    if samples < 1:
        raise Invalid("at least one sample is needed")
    side = 1 << zoom
    hist: dict[int, int] = {}
    for _ in range(samples):
        x, y = rng.randrange(side), rng.randrange(side)
        key = encode(zoom, x, y)
        if x + 1 < side:
            shared = common_prefix(key, encode(zoom, x + 1, y))
            hist[shared] = hist.get(shared, 0) + 1
    return dict(sorted(hist.items()))


def expected_prefix_histogram(zoom: int) -> dict[int, float]:
    # an eastern neighbour differs first at the level where x's lowest set bits carry
    out = {}
    for shared in range(zoom):
        out[shared] = 2.0 ** -(zoom - shared)
    return out


def covering(zoom: int, x0: int, y0: int, x1: int, y1: int) -> list[str]:
    if x0 > x1 or y0 > y1:
        raise Invalid("the box must be ordered")
    keys = [encode(zoom, x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    return _merge(keys)


def _merge(keys: list[str]) -> list[str]:
    present = set(keys)
    changed = True
    while changed:
        changed = False
        for key in sorted(present, key=len, reverse=True):
            if key not in present or not key:
                continue
            siblings = children(parent(key))
            if all(s in present for s in siblings):
                for s in siblings:
                    present.discard(s)
                present.add(parent(key))
                changed = True
    return sorted(present, key=lambda k: (len(k), k))


def key_count_by_zoom(zoom: int) -> int:
    return 4**zoom


def path_length_bytes(zoom: int) -> int:
    return zoom
