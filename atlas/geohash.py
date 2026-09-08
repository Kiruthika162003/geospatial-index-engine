"""Geohash: fold latitude and longitude into a short base-32 string with a prefix rule.

A geohash turns a latitude and longitude into a short text string by
repeatedly halving the world. Start with the full ranges of longitude
and latitude; each bit chosen says which half the point falls in,
alternating longitude then latitude, and the bits are grouped five at
a time and mapped to a base-32 alphabet, giving a string like a postal
code for a patch of the globe. Two properties make it useful. First,
the length controls precision: each extra character narrows the cell
by a factor of thirty-two split across the two axes, so a longer
geohash names a smaller box, and truncating a geohash zooms out to the
enclosing box exactly. Second, and the reason geohashes index well, is
the prefix rule: if two geohashes share a leading prefix, their points
lie in the same coarse cell that the prefix names, so a database can
find everything near a point by matching a prefix, turning a spatial
query into a string range scan. The rule has the same seam caveat as
every grid laid over a continuous space, and it is worth stating
plainly rather than hiding: two points can be physically adjacent yet
sit on opposite sides of a cell boundary and share no prefix at all,
so prefix matching finds most nearby points but misses the ones just
across a seam, which is why geohash neighbor search must also check the
eight bordering cells. The finding worth stating is that geohash buys
a text-sortable, prefix-searchable key whose shared-prefix length
tracks proximity well inside a cell and fails at the seams. This module
encodes a point to a geohash, decodes a geohash to its bounding box and
center, and a survey measures how shared-prefix length tracks distance
and where the seams break it.
"""

from __future__ import annotations

from atlas.errors import Invalid, Outside

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE = {c: i for i, c in enumerate(_BASE32)}


def encode(lat: float, lon: float, precision: int = 12) -> str:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    if precision <= 0:
        raise Invalid("precision must be positive")
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    chars: list[str] = []
    bit = 0
    value = 0
    even = True
    while len(chars) < precision:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                value = (value << 1) | 1
                lon_range[0] = mid
            else:
                value <<= 1
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                value = (value << 1) | 1
                lat_range[0] = mid
            else:
                value <<= 1
                lat_range[1] = mid
        even = not even
        bit += 1
        if bit == 5:
            chars.append(_BASE32[value])
            bit = 0
            value = 0
    return "".join(chars)


def bounds(geohash: str) -> tuple[float, float, float, float]:
    if not geohash:
        raise Invalid("geohash must not be empty")
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    even = True
    for ch in geohash:
        if ch not in _DECODE:
            raise Invalid(f"'{ch}' is not a geohash character")
        value = _DECODE[ch]
        for i in range(4, -1, -1):
            bit = (value >> i) & 1
            if even:
                mid = (lon_range[0] + lon_range[1]) / 2
                lon_range[0 if bit else 1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                lat_range[0 if bit else 1] = mid
            even = not even
    return (lat_range[0], lon_range[0], lat_range[1], lon_range[1])


def decode(geohash: str) -> tuple[float, float]:
    min_lat, min_lon, max_lat, max_lon = bounds(geohash)
    return ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
