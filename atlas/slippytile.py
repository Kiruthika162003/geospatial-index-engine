"""Slippy tiles: address the map as a pyramid of squares that doubles each zoom.

Online maps are served as square image tiles arranged on a pyramid.
At zoom level zero the whole world is one tile; each level up splits
every tile into four, so level z has two-to-the-z tiles across and
two-to-the-z down, four-to-the-z tiles in all. A tile is addressed by
its zoom and its integer column and row, counting from the top-left,
and the mapping from a longitude and latitude to the tile that contains
it uses the Web Mercator projection: longitude scales linearly to the
column, latitude through the log-tangent to the row. Two consequences
are worth stating as numbers rather than intuitions. First, the tile
count explodes: each zoom quadruples the tiles, so the whole world at
zoom twenty is over a trillion tiles, which is why maps are rendered on
demand, not stored. Second, the ground resolution, how many meters of
earth each pixel covers, halves with every zoom level, because the same
world is split into twice as many tiles per axis; at the equator zoom
zero is about a hundred and fifty-six kilometers per pixel for a
two-hundred-fifty-six-pixel tile, and each level down halves it, so
resolution and zoom are tied by a clean power of two. The same
quadtree address has a compact string form, the quadkey, which
interleaves the column and row bits into a base-four string whose
length is the zoom and whose prefixes name the containing tiles, giving
tiles the same prefix-locality a geohash gives points. The finding
worth stating is that tile count grows as four-to-the-zoom while
ground resolution shrinks as one-half-to-the-zoom, the two exponential
halves of the same doubling. This module converts between longitude and
latitude and tile coordinates, builds and parses quadkeys, and a survey
measures the tile count and resolution law across zooms.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside

_MAX_LAT = 85.05112877980659


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    if zoom < 0:
        raise Invalid("zoom must not be negative")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    if not -_MAX_LAT <= lat <= _MAX_LAT:
        raise Outside("latitude must lie within the Web Mercator limit")
    n = 1 << zoom
    x = int((lon + 180.0) / 360.0 * n)
    phi = math.radians(lat)
    y = int((1 - math.log(math.tan(phi) + 1 / math.cos(phi)) / math.pi) / 2 * n)
    x = min(x, n - 1)
    y = min(y, n - 1)
    return (x, y)


def tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 1 << zoom
    if not (0 <= x < n and 0 <= y < n):
        raise Invalid("tile coordinates fall outside the zoom level's grid")

    def lon_of(col: int) -> float:
        return col / n * 360.0 - 180.0

    def lat_of(row: int) -> float:
        t = math.pi * (1 - 2 * row / n)
        return math.degrees(math.atan(math.sinh(t)))

    north = lat_of(y)
    south = lat_of(y + 1)
    west = lon_of(x)
    east = lon_of(x + 1)
    return (south, west, north, east)


def tile_count(zoom: int) -> int:
    if zoom < 0:
        raise Invalid("zoom must not be negative")
    return 1 << (2 * zoom)


def ground_resolution(lat: float, zoom: int, tile_pixels: int = 256) -> float:
    # meters per pixel at a latitude and zoom, on a spherical earth
    if zoom < 0:
        raise Invalid("zoom must not be negative")
    earth_circumference = 2 * math.pi * 6378137.0
    return (
        math.cos(math.radians(lat)) * earth_circumference / (tile_pixels * (1 << zoom))
    )


def to_quadkey(x: int, y: int, zoom: int) -> str:
    if zoom <= 0:
        raise Invalid("quadkey zoom must be positive")
    n = 1 << zoom
    if not (0 <= x < n and 0 <= y < n):
        raise Invalid("tile coordinates fall outside the zoom level's grid")
    digits = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        digits.append(str(digit))
    return "".join(digits)


def from_quadkey(quadkey: str) -> tuple[int, int, int]:
    if not quadkey or any(c not in "0123" for c in quadkey):
        raise Invalid("a quadkey is a non-empty string of digits 0 to 3")
    x = y = 0
    zoom = len(quadkey)
    for i, digit in enumerate(quadkey):
        mask = 1 << (zoom - 1 - i)
        d = int(digit)
        if d & 1:
            x |= mask
        if d & 2:
            y |= mask
    return (x, y, zoom)
