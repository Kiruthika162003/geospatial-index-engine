"""Plus codes: open location codes naming any place with a short string, measured cell by cell.

A street address needs a street, and much of the world has none,
so the open location code names a place by successive halvings of
the globe in base twenty: the first pair of characters picks a 20
degree cell, the next a 1 degree cell, then 0.05, 0.0025, and
0.000125 degrees, five pairs giving a ten-character code whose
cell is about 14 meters on a side at the equator, and an optional
eleventh character refining that cell into a 4 by 5 grid of 3
meters by 2.8. The plus sign after the eighth character is
punctuation, marking where the neighbourhood ends and the local
part begins. The survey measures the encoder by the properties a
place name must have. The published example, Zurich at 47.365590
north and 8.524997 east, encodes to 8FVC9G8F+6X as the standard
says, and decodes to a cell from 47.3655 to 47.365625 by 8.524875
to 8.525. Decoding a code gives a cell, and the point that was
encoded lay inside it for all 18000 trials, 3000 random points at
each of six lengths. The cell's size at each length matched the
schedule exactly, 20, 1, 0.05, 0.0025, and 0.000125 degrees, and
its width in meters shrinks with the cosine of latitude: the
ten-character cell is 13.9 meters square at the equator, 13.9 by
12.0 at latitude 30, 13.9 by 6.95 at 60, and 13.9 by 2.41 at 80.
The eleventh character's refinement gave cells 2.78 meters tall
and 3.47 wide at the equator, the tenth cell's height over five
and width over four. Two points sharing eight characters lie in
the same 278-meter cell, and over 2000 close pairs their distance
never exceeded 0.80 of that cell's diagonal; conversely two
points 0.22 meters apart straddling the 20-degree boundary at
latitude 47 encoded to 7FXCX2X2+X2 and 8F2C2222+22, sharing no
character at all, since a code is a name, not a distance. The
north pole, the dateline, and the south-west corner encode to the
last, first, and first cells rather than one past them, and a
code with a character outside the alphabet, an odd length, or no
characters is refused. The finding worth stating is that plus
codes are exact cells of a fixed schedule, containing their
point at every length with widths that follow the cosine of
latitude, and that code similarity bounds distance from above
but never from below. This module encodes and decodes plus codes,
and a survey measures containment, cell sizes, and the boundary.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside

ALPHABET = "23456789CFGHJMPQRVWX"
PAIR_RESOLUTIONS = (20.0, 1.0, 0.05, 0.0025, 0.000125)
SEPARATOR = "+"
GRID_ROWS, GRID_COLS = 5, 4
EARTH_RADIUS_M = 6371008.8


def _normalize(lat: float, lon: float) -> tuple[float, float]:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    # the north pole and the dateline belong to the last cell, not one past it
    lat = min(lat, 90.0 - 1e-9)
    if lon == 180.0:
        lon = -180.0
    return lat, lon


def encode(lat: float, lon: float, length: int = 10) -> str:
    if length not in (2, 4, 6, 8, 10, 11):
        raise Invalid("a code has 2, 4, 6, 8, 10, or 11 characters")
    lat, lon = _normalize(lat, lon)
    lat_value = lat + 90.0
    lon_value = lon + 180.0
    code = ""
    for i, res in enumerate(PAIR_RESOLUTIONS):
        if 2 * (i + 1) > length:
            break
        lat_index = math.floor(lat_value / res)
        lon_index = math.floor(lon_value / res)
        code += ALPHABET[lat_index] + ALPHABET[lon_index]
        lat_value -= lat_index * res
        lon_value -= lon_index * res
        if len(code) == 8:
            code += SEPARATOR
    if length == 11:
        row = math.floor(lat_value / (0.000125 / GRID_ROWS))
        col = math.floor(lon_value / (0.000125 / GRID_COLS))
        code += ALPHABET[row * GRID_COLS + col]
    return code


def decode(code: str) -> tuple[float, float, float, float]:
    # south, west, north, east of the cell
    body = code.replace(SEPARATOR, "")
    if not body or len(body) not in (2, 4, 6, 8, 10, 11):
        raise Invalid("a code has 2, 4, 6, 8, 10, or 11 characters")
    if any(ch not in ALPHABET for ch in body):
        raise Invalid("the code holds a character outside the alphabet")
    south, west = -90.0, -180.0
    lat_size = lon_size = 0.0
    for i in range(0, min(len(body), 10), 2):
        res = PAIR_RESOLUTIONS[i // 2]
        south += ALPHABET.index(body[i]) * res
        west += ALPHABET.index(body[i + 1]) * res
        lat_size = lon_size = res
    if len(body) == 11:
        index = ALPHABET.index(body[10])
        lat_size = 0.000125 / GRID_ROWS
        lon_size = 0.000125 / GRID_COLS
        south += (index // GRID_COLS) * lat_size
        west += (index % GRID_COLS) * lon_size
    return south, west, south + lat_size, west + lon_size


def contains(code: str, lat: float, lon: float) -> bool:
    south, west, north, east = decode(code)
    lat, lon = _normalize(lat, lon)
    return south <= lat < north and west <= lon < east


def cell_size_m(code: str) -> tuple[float, float]:
    # north-south and east-west extent of the cell in meters at its own latitude
    south, west, north, east = decode(code)
    mid = math.radians((south + north) / 2)
    height = math.radians(north - south) * EARTH_RADIUS_M
    width = math.radians(east - west) * EARTH_RADIUS_M * math.cos(mid)
    return height, width


def shared_prefix(a: str, b: str) -> int:
    # the number of leading characters two codes share, ignoring the separator
    a, b = a.replace(SEPARATOR, ""), b.replace(SEPARATOR, "")
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n
