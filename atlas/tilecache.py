"""An LRU tile cache smaller than one viewport gets no hits at all under a panning scan.

A map viewer requests every tile under its viewport on every frame,
35 tiles for a 6 by 4 viewport with its partial edges, and a cache
answers from memory or fetches. A panning session of 400 frames at
zoom 12 with a stride of half a tile makes 14,000 requests for 997
distinct tiles, so the compulsory floor, the best any cache can do,
is 0.9288. The guess that a small cache helps a little was wrong for
LRU: with room for 10 or 20 tiles it reads a hit rate of exactly 0.0,
since each frame's scan of 35 tiles evicts every tile before the next
frame asks for it again, while Belady's clairvoyant policy reads
0.2628 and 0.5439 in the same room. At 35 tiles LRU reads 0.6499 and
Belady 0.8903; at 50, 0.8922 and 0.9015; at 100, 200 and 400, 0.9014,
0.9094 and 0.9173 against 0.9131, 0.9236 and 0.9288.

A session that zooms between levels 8 and 12 over one spot touches
175 distinct tiles, and LRU reaches its floor of 0.9875 at 200 tiles,
reading 0.495 at 35 and 0.7975 at 100. A session that jumps between
50 hotspots with Zipf weights touches 2333 tiles, floor 0.8334, and
LRU reads 0.047, 0.134, 0.246, 0.409 and 0.597 at 35, 100, 200, 400
and 800 tiles against Belady's 0.227, 0.388, 0.508, 0.651 and 0.766.
The pan stride sets the floor: strides of a quarter, half, one, two
and four tiles touch 533, 1144, 2539, 5356 and 9963 distinct tiles,
floors 0.962, 0.918, 0.819, 0.617 and 0.288, and a 100-tile LRU reads
0.956, 0.904, 0.799, 0.604 and 0.279, within 0.02 of the floor at
every stride.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict

from atlas.errors import Invalid

Tile = tuple[int, int, int]


class LRUCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise Invalid("the cache needs room for one tile")
        self.capacity = capacity
        self.tiles: OrderedDict[Tile, None] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def fetch(self, tile: Tile) -> bool:
        if tile in self.tiles:
            self.tiles.move_to_end(tile)
            self.hits += 1
            return True
        self.misses += 1
        self.tiles[tile] = None
        if len(self.tiles) > self.capacity:
            self.tiles.popitem(last=False)
        return False

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            raise Invalid("no requests yet")
        return self.hits / total


def viewport_tiles(zoom: int, cx: float, cy: float, width: int, height: int) -> list[Tile]:
    if zoom < 0 or width < 1 or height < 1:
        raise Invalid("zoom must be non-negative and the viewport at least one tile")
    side = 1 << zoom
    left = math.floor(cx - width / 2)
    top = math.floor(cy - height / 2)
    tiles = []
    for y in range(top, top + height + 1):
        for x in range(left, left + width + 1):
            if 0 <= y < side:
                tiles.append((zoom, x % side, y))
    return tiles


def pan_session(
    zoom: int,
    steps: int,
    rng: random.Random,
    width: int = 6,
    height: int = 4,
    stride: float = 0.5,
) -> list[Tile]:
    side = 1 << zoom
    cx, cy = rng.uniform(0, side), rng.uniform(height, side - height)
    heading = rng.uniform(0, 2 * math.pi)
    requests: list[Tile] = []
    for _ in range(steps):
        heading += rng.gauss(0, 0.4)
        cx += stride * math.cos(heading)
        cy = min(max(cy + stride * math.sin(heading), height), side - height)
        requests.extend(viewport_tiles(zoom, cx, cy, width, height))
    return requests


def zoom_session(
    zoom_low: int,
    zoom_high: int,
    steps: int,
    rng: random.Random,
    width: int = 6,
    height: int = 4,
) -> list[Tile]:
    if zoom_low > zoom_high:
        raise Invalid("zoom_low must not exceed zoom_high")
    requests: list[Tile] = []
    fx, fy = rng.random(), rng.random()
    zoom = zoom_low
    for _ in range(steps):
        zoom = min(max(zoom + rng.choice((-1, 0, 1)), zoom_low), zoom_high)
        side = 1 << zoom
        requests.extend(viewport_tiles(zoom, fx * side, fy * side, width, height))
    return requests


def zipf_session(
    zoom: int, steps: int, rng: random.Random, hot: int = 50, skew: float = 1.0
) -> list[Tile]:
    side = 1 << zoom
    spots = [(rng.uniform(0, side), rng.uniform(3, side - 3)) for _ in range(hot)]
    weights = [1 / (k + 1) ** skew for k in range(hot)]
    requests: list[Tile] = []
    for _ in range(steps):
        cx, cy = rng.choices(spots, weights)[0]
        jx, jy = cx + rng.gauss(0, 0.5), cy + rng.gauss(0, 0.5)
        requests.extend(viewport_tiles(zoom, jx, jy, 6, 4))
    return requests


def replay(requests: list[Tile], capacity: int) -> float:
    cache = LRUCache(capacity)
    for tile in requests:
        cache.fetch(tile)
    return cache.hit_rate()


def distinct(requests: list[Tile]) -> int:
    return len(set(requests))


def compulsory_floor(requests: list[Tile]) -> float:
    if not requests:
        raise Invalid("no requests")
    return 1 - distinct(requests) / len(requests)


def belady(requests: list[Tile], capacity: int) -> float:
    if capacity < 1:
        raise Invalid("the cache needs room for one tile")
    future: dict[Tile, list[int]] = {}
    for i, tile in enumerate(requests):
        future.setdefault(tile, []).append(i)
    positions = {tile: 0 for tile in future}
    cache: set[Tile] = set()
    hits = 0
    for tile in requests:
        positions[tile] += 1
        if tile in cache:
            hits += 1
            continue
        if len(cache) >= capacity:
            victim = max(cache, key=lambda t: _next_use(future, positions, t))
            cache.remove(victim)
        cache.add(tile)
    return hits / len(requests)


def _next_use(future: dict[Tile, list[int]], positions: dict[Tile, int], tile: Tile) -> int:
    uses = future[tile]
    k = positions[tile]
    return uses[k] if k < len(uses) else 1 << 60
