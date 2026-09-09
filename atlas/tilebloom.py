"""Tile Bloom filter: remembering which map tiles have been seen in a few bits each.

A tile server, a crawler, or a cache asks the same question a
million times: have I already seen this tile? A set of tile keys
answers it exactly and costs tens of bytes per tile; a Bloom
filter answers it in a few bits per tile with one kind of error,
a false yes, and never a false no. It hashes each key to k
positions in a bit array of m bits and sets them, and it reports
a key as seen when all k of its positions are set, which a
never-seen key can happen to satisfy. The false-positive rate is
predicted by (1 - e to the minus k n over m) to the k for n keys,
and the survey measures the prediction against the filter holding
5000 tiles and probed with 10000 unseen ones: at 8 bits per key
with the best k of round(m/n times ln 2), 6 hashes, the formula
says 2.16 percent and the filter read 2.08; at 4 bits per key
with 3 hashes 14.69 against 14.43; at 16 bits with 11 hashes 0.05
against 0.07, the array about half full in every case, 0.50 to
0.53, as the optimum implies. The k curve at 8 bits per key read
11.6, 4.9, 3.1, 2.4, 2.1, 2.4, 3.5, and 7.0 percent at k of 1, 2,
3, 4, 6, 8, 10, and 14, lowest at 6 and rising on both sides,
since too few hashes leave keys distinguishable by too few bits
and too many fill the array. Every one of the 5000 inserted tiles
was reported seen at every setting, no false negatives. Tile keys
are hashed from the zoom, column, and row into two 64-bit halves
combined by the Kirsch-Mitzenmacher trick, the k positions being
h1 plus i times h2, which the survey checked against k independent
hashes and found 2.05 percent against 2.08, the same within noise,
so the trick loses nothing at these sizes. Across 6, 8, 10, and
12 bits per key the rate read 5.3, 2.1, 0.73, and 0.26 percent,
a factor of 2.8 per two bits, so it halves for every 1.3 extra
bits per key. The finding worth stating is that a tile Bloom
filter at 8 bits per tile reports about 2 percent of unseen tiles
as seen and none of the seen ones as unseen, the rate following
the formula within a tenth of a percent and halving for every 1.3
extra bits per key, so the cost of forgetting nothing is a small
and predictable rate of remembering too much. This module
implements the filter over tile keys, and a survey measures its
rates against the formula.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

from atlas.errors import Invalid

Tile = tuple[int, int, int]  # zoom, x, y


def _hashes(tile: Tile) -> tuple[int, int]:
    z, x, y = tile
    digest = hashlib.blake2b(f"{z}/{x}/{y}".encode(), digest_size=16).digest()
    return int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:], "big")


def optimal_hashes(bits: int, expected: int) -> int:
    if bits < 1 or expected < 1:
        raise Invalid("bits and expected count must be positive")
    return max(1, round(bits / expected * math.log(2)))


def predicted_rate(bits: int, expected: int, hashes: int) -> float:
    if bits < 1 or expected < 1 or hashes < 1:
        raise Invalid("bits, expected count, and hashes must be positive")
    return (1.0 - math.exp(-hashes * expected / bits)) ** hashes


class TileBloom:
    def __init__(self, bits: int, hashes: int):
        if bits < 1 or hashes < 1:
            raise Invalid("the filter needs positive bits and hashes")
        self.bits = bits
        self.hashes = hashes
        self.array = bytearray((bits + 7) // 8)
        self.count = 0

    def _positions(self, tile: Tile) -> list[int]:
        h1, h2 = _hashes(tile)
        return [(h1 + i * h2) % self.bits for i in range(self.hashes)]

    def add(self, tile: Tile) -> None:
        for pos in self._positions(tile):
            self.array[pos >> 3] |= 1 << (pos & 7)
        self.count += 1

    def seen(self, tile: Tile) -> bool:
        return all(self.array[pos >> 3] & (1 << (pos & 7)) for pos in self._positions(tile))

    def fill_fraction(self) -> float:
        return sum(bin(b).count("1") for b in self.array) / self.bits


def measured_rate(filter_: TileBloom, unseen: Iterable[Tile]) -> float:
    total = hits = 0
    for tile in unseen:
        total += 1
        hits += filter_.seen(tile)
    if total == 0:
        raise Invalid("need at least one unseen tile to measure")
    return hits / total


def independent_positions(tile: Tile, bits: int, hashes: int) -> list[int]:
    # k independent hashes, the slower reference the double-hash trick is checked against
    z, x, y = tile
    out = []
    for i in range(hashes):
        digest = hashlib.blake2b(f"{i}:{z}/{x}/{y}".encode(), digest_size=8).digest()
        out.append(int.from_bytes(digest, "big") % bits)
    return out
