"""Viewport: fitting a map window to a region, with the zoom, the scale, and the tile count.

A map on a screen is a window of some pixels at a zoom level
centered somewhere, and the questions a map application asks are
arithmetic on the web Mercator tile pyramid: how many meters does
one pixel cover here, which zoom fits this bounding box in this
window, how many tiles does the window need, and how long is a
scale bar of a given pixel length. At zoom z the world is 256
times 2 to the z pixels wide, so a pixel at the equator covers
40075 km over that, 156543 meters at zoom 0, halving per level,
and at latitude phi it covers cos(phi) times less on the ground,
since Mercator stretches the map by the secant there. The survey
measures the arithmetic against the ground: the great-circle
distance between two pixels a hundred apart, computed by
unprojecting both, sat 0.112 percent below the meters per pixel
times a hundred at every zoom from 8 to 18 and every latitude from
0 to 70, a constant that is not the projection's doing but the
radius's, since the great-circle distance uses the mean sphere of
6371.0088 km while the Mercator constant of 40075 km is the
equatorial circumference of 6378.137, and the ratio of the two
is 0.99888; at low zoom the Mercator scale changes across the
hundred pixels and the gap widens, to 1.9 percent at zoom 1 and
latitude 30 and 5.6 at latitude 70, where a hundred pixels is a
fifth of the world. The fit zoom for a bounding box is the
largest integer zoom at which the box's projected width and
height both fit the window, read from the projection rather than
from degrees since a degree of longitude is more pixels near the
poles, and on 500 random boxes and windows the box fit at the fit
zoom and overflowed at one more every time; the world fits a 1024
by 768 window at zoom 1 and a 10-kilometer box round London an
800 by 600 window at zoom 12. The tile count of a window was
guessed at the window's tile span plus one in each axis at worst,
and the worst is the rule: a 1024 by 768 window, four tiles by
three, needed 20 tiles, five by four, at every one of 2000 random
centers and zooms, since the aligned minimum of 12 needs the
window's edges on tile boundaries in both axes at once, which a
random center never gives. The finding worth stating is that
meters per pixel follow 156543 times cos(phi) over 2 to the z
with the ground reading 0.112 percent short by the choice of
sphere radius, that the fit zoom is the level where a box's
projected extent first fits, and that a window needs its tile
span plus one in each axis, so viewport arithmetic is exact once
it is done in projected pixels and not in degrees. This module
computes viewport arithmetic, and a survey measures it against
the ground.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside
from atlas.haversine import haversine

TILE = 256
EQUATOR_M = 40_075_016.686
MAX_LAT = 85.05112878


def _check(lat: float, lon: float) -> None:
    if not -MAX_LAT <= lat <= MAX_LAT:
        raise Outside("latitude must lie within the Mercator limits")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")


def meters_per_pixel(lat: float, zoom: float) -> float:
    if zoom < 0:
        raise Invalid("zoom cannot be negative")
    return EQUATOR_M * math.cos(math.radians(lat)) / (TILE * 2.0**zoom)


def to_pixels(lat: float, lon: float, zoom: float) -> tuple[float, float]:
    _check(lat, lon)
    scale = TILE * 2.0**zoom
    x = (lon + 180.0) / 360.0 * scale
    phi = math.radians(lat)
    y = (1.0 - math.log(math.tan(phi) + 1.0 / math.cos(phi)) / math.pi) / 2.0 * scale
    return x, y


def from_pixels(x: float, y: float, zoom: float) -> tuple[float, float]:
    scale = TILE * 2.0**zoom
    lon = x / scale * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / scale
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lon


def ground_distance_of_pixels(lat: float, lon: float, zoom: float, pixels: float) -> float:
    # the great-circle distance in meters between a point and one `pixels` to its east
    x, y = to_pixels(lat, lon, zoom)
    lat2, lon2 = from_pixels(x + pixels, y, zoom)
    return haversine(lat, lon, lat2, lon2) * 1000.0


def fit_zoom(south: float, west: float, north: float, east: float, width: int, height: int):
    # the largest integer zoom at which the box fits the window in projected pixels
    if width < 1 or height < 1:
        raise Invalid("the window needs positive pixel dimensions")
    if south >= north or west >= east:
        raise Invalid("the box must have positive extent")
    x0, y0 = to_pixels(north, west, 0)
    x1, y1 = to_pixels(south, east, 0)
    box_w, box_h = x1 - x0, y1 - y0
    if box_w <= 0 or box_h <= 0:
        raise Invalid("the box has no projected extent")
    zoom = math.floor(math.log2(min(width / box_w, height / box_h)))
    return max(0, zoom)


def fits(
    south: float, west: float, north: float, east: float, width: int, height: int, zoom: int
) -> bool:
    x0, y0 = to_pixels(north, west, zoom)
    x1, y1 = to_pixels(south, east, zoom)
    return x1 - x0 <= width and y1 - y0 <= height


def tile_count(lat: float, lon: float, zoom: int, width: int, height: int) -> int:
    # tiles a window centered on the point needs at an integer zoom
    if zoom < 0 or zoom != int(zoom):
        raise Invalid("the zoom must be a non-negative integer")
    cx, cy = to_pixels(lat, lon, zoom)
    left, right = cx - width / 2, cx + width / 2
    top, bottom = cy - height / 2, cy + height / 2
    cols = math.floor((right - 1e-9) / TILE) - math.floor(left / TILE) + 1
    rows = math.floor((bottom - 1e-9) / TILE) - math.floor(top / TILE) + 1
    return cols * rows


def scale_bar_meters(lat: float, zoom: float, pixels: float) -> float:
    return meters_per_pixel(lat, zoom) * pixels
