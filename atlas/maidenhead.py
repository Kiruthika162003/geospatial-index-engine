"""Maidenhead locators: the radio amateur's grid squares, and their uneven cells measured.

Radio amateurs exchange locations as grid squares, a scheme
older than any digital map: two letters name an 18 by 18 field of
20 degrees of longitude by 10 of latitude, two digits a square of
2 by 1 degrees within it, two more letters a subsquare of 5 by
2.5 minutes, and two more digits an extended square of 30 by 15
seconds. The scheme's oddity is that every level is twice as wide
in longitude as it is tall in latitude, so its cells are not
squares at all but rectangles, and at high latitude, where a
degree of longitude shrinks, they become narrow tall boxes on the
ground. The survey measures the scheme by the properties a
locator must have. Munich at 48.1 north, 11.6 east encodes to
JN58TC, the field and square the published tables give. A
locator's cell contained the point that was encoded in all 12000
trials, 3000 random points at each of four lengths, decoding to
the cell's center and re-encoding returned the same locator every
time, and the center lay within half a cell of the original point
in both axes, the worst offset being exactly 0.5 of a cell for
points on a cell edge. The cell's extent on the ground follows
the schedule and the cosine of latitude: a six-character
subsquare is 4.63 km tall everywhere and 9.27 km wide at the
equator, aspect 2.0, 8.02 km at latitude 30, aspect 1.73, 4.63 km
at 60, aspect 0.999 and finally square, and 2.40 km at 75, aspect
0.52, a tall narrow box; its diagonal is 10.4 km at the equator
and 6.6 at 60. Adjacent field centers are 2215 km apart east to
west near the equator, 1112 km north to south, and 193 km east to
west in the last row of fields near the pole. The south pole at
the antimeridian encodes to AA00AA, the first cell, and the north
pole at the dateline to RR99XX, the last, and a locator with a
letter past R in its field, past X in its subsquare, a letter
where a digit belongs, or an odd length is refused. The finding
worth stating is that Maidenhead cells are rectangles twice as
wide as tall in degrees that become square on the ground only at
latitude 60, that encoding contains and decoding centers within
half a cell at every length, and that the scheme's two-to-one
aspect is a fact of its history a map must correct for. This
module encodes and decodes locators, and a survey measures
containment, centering, and the cell shapes.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside

FIELD_LETTERS = "ABCDEFGHIJKLMNOPQR"
SUBSQUARE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"
EARTH_RADIUS_M = 6371008.8

# (longitude span, latitude span) in degrees for each pair of characters
SPANS = ((20.0, 10.0), (2.0, 1.0), (5.0 / 60.0, 2.5 / 60.0), (5.0 / 600.0, 2.5 / 600.0))


def _normalize(lat: float, lon: float) -> tuple[float, float]:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    return min(lat, 90.0 - 1e-9), min(lon, 180.0 - 1e-9)


def encode(lat: float, lon: float, length: int = 6) -> str:
    if length not in (2, 4, 6, 8):
        raise Invalid("a locator has 2, 4, 6, or 8 characters")
    lat, lon = _normalize(lat, lon)
    lon_value, lat_value = lon + 180.0, lat + 90.0
    out = ""
    for i, (lon_span, lat_span) in enumerate(SPANS):
        if 2 * (i + 1) > length:
            break
        lon_index = math.floor(lon_value / lon_span)
        lat_index = math.floor(lat_value / lat_span)
        if i % 2 == 0:
            letters = FIELD_LETTERS if i == 0 else SUBSQUARE_LETTERS
            out += letters[lon_index] + letters[lat_index]
        else:
            out += str(lon_index) + str(lat_index)
        lon_value -= lon_index * lon_span
        lat_value -= lat_index * lat_span
    return out


def decode(locator: str) -> tuple[float, float, float, float]:
    # south, west, north, east of the cell
    code = locator.upper()
    if len(code) not in (2, 4, 6, 8):
        raise Invalid("a locator has 2, 4, 6, or 8 characters")
    west, south = -180.0, -90.0
    lon_size = lat_size = 0.0
    for i in range(0, len(code), 2):
        lon_span, lat_span = SPANS[i // 2]
        a, b = code[i], code[i + 1]
        if (i // 2) % 2 == 0:
            letters = FIELD_LETTERS if i == 0 else SUBSQUARE_LETTERS
            if a not in letters or b not in letters:
                raise Invalid("a letter lies outside the locator alphabet")
            lon_index, lat_index = letters.index(a), letters.index(b)
        else:
            if not (a.isdigit() and b.isdigit()):
                raise Invalid("a square needs two digits")
            lon_index, lat_index = int(a), int(b)
        west += lon_index * lon_span
        south += lat_index * lat_span
        lon_size, lat_size = lon_span, lat_span
    return south, west, south + lat_size, west + lon_size


def center(locator: str) -> tuple[float, float]:
    south, west, north, east = decode(locator)
    return ((south + north) / 2, (west + east) / 2)


def contains(locator: str, lat: float, lon: float) -> bool:
    south, west, north, east = decode(locator)
    lat, lon = _normalize(lat, lon)
    return south <= lat < north and west <= lon < east


def cell_size_m(locator: str) -> tuple[float, float]:
    # north-south and east-west extent of the cell in meters at its own latitude
    south, west, north, east = decode(locator)
    mid = math.radians((south + north) / 2)
    height = math.radians(north - south) * EARTH_RADIUS_M
    width = math.radians(east - west) * EARTH_RADIUS_M * math.cos(mid)
    return height, width


def aspect_on_ground(locator: str) -> float:
    # width over height on the ground: 2 at the equator, 1 at latitude 60
    height, width = cell_size_m(locator)
    return width / height
