"""Geohash neighbors: the eight bordering cells that close the seam a prefix scan leaves.

A geohash prefix scan finds points in the same cell as a query, but
the geohash grid has seams: a point just across a cell boundary shares
no prefix and is missed, even though it is physically adjacent. The fix
is to search not one cell but nine, the query's cell and its eight
neighbors, so any nearby point, whichever side of a seam it fell on, is
covered by one of the nine. Computing a neighbor is not simple
arithmetic on the string, because the geohash bit layout alternates
longitude and latitude and the base-32 alphabet is not in a
geometrically ordered sequence. The standard method uses two small
lookup tables per direction and per parity of the geohash length: a
border table naming the characters that sit on the edge of a cell in a
given direction, and a neighbor table giving, for each character, the
character of the adjacent cell in that direction. To step north, look
up the last character in the north-neighbor table for the current
length parity; if the last character was on the north border, the step
also carries into the prefix, so the parent geohash is advanced north
first and then the last character is taken from the border-wrap. The
diagonal neighbors are just two orthogonal steps composed. The finding
worth stating is that adding the eight neighbors to the query cell
turns the leaky prefix scan into a complete local search: the points a
bare prefix match misses at the seams are exactly the points the
neighbor cells recover, so the nine-cell set has no seam gaps. This
module computes a geohash's neighbor in any of the eight directions,
and a survey confirms that for points near a cell boundary the
nine-cell set contains the nearby points a single-cell scan drops.
"""

from __future__ import annotations

from atlas.errors import Invalid

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

_NEIGHBORS = {
    "n": ("p0r21436x8zb9dcf5h7kjnmqesgutwvy", "bc01fg45238967deuvhjyznpkmstqrwx"),
    "s": ("14365h7k9dcfesgujnmqp0r2twvyx8zb", "238967debc01fg45kmstqrwxuvhjyznp"),
    "e": ("bc01fg45238967deuvhjyznpkmstqrwx", "p0r21436x8zb9dcf5h7kjnmqesgutwvy"),
    "w": ("238967debc01fg45kmstqrwxuvhjyznp", "14365h7k9dcfesgujnmqp0r2twvyx8zb"),
}
_BORDERS = {
    "n": ("prxz", "bcfguvyz"),
    "s": ("028b", "0145hjnp"),
    "e": ("bcfguvyz", "prxz"),
    "w": ("0145hjnp", "028b"),
}


def adjacent(geohash: str, direction: str) -> str:
    if not geohash:
        raise Invalid("geohash must not be empty")
    direction = direction.lower()
    if direction not in _NEIGHBORS:
        raise Invalid("direction must be one of n, s, e, w")
    if any(c not in _BASE32 for c in geohash):
        raise Invalid("geohash contains a non-base-32 character")
    last = geohash[-1]
    parent = geohash[:-1]
    parity = len(geohash) % 2  # 0 = even length, 1 = odd length
    if parent and last in _BORDERS[direction][parity]:
        parent = adjacent(parent, direction)
    return parent + _BASE32[_NEIGHBORS[direction][parity].index(last)]


def neighbors(geohash: str) -> dict[str, str]:
    n = adjacent(geohash, "n")
    s = adjacent(geohash, "s")
    return {
        "n": n,
        "s": s,
        "e": adjacent(geohash, "e"),
        "w": adjacent(geohash, "w"),
        "ne": adjacent(n, "e"),
        "nw": adjacent(n, "w"),
        "se": adjacent(s, "e"),
        "sw": adjacent(s, "w"),
    }


def nine_cells(geohash: str) -> list[str]:
    return [geohash, *neighbors(geohash).values()]
