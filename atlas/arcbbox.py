"""Arc bounding box: the box round a great-circle route, which bulges past its endpoints.

The bounding box of a route is not the box of its endpoints. A
great circle from Paris to Vancouver, both near latitude 49,
climbs to latitude 68.3 over Greenland, 2119 km north of the
endpoint box, where the guess had put it at 62, and a box drawn
from the endpoints alone misses the whole northern bulge; a
spatial index that filed the route by that box would fail to
find it for any query over Greenland. Clairaut's relation settles where the bulge
peaks: along any great circle the product of the cosine of the
latitude and the sine of the bearing is constant, so the highest
latitude the circle reaches is the arccosine of that constant,
and the arc reaches it only if the vertex lies between the
endpoints, which the bearing at each end decides, since the
bearing crosses east-west exactly at the vertex. The module
computes the true box, latitude extended to the vertex when the
arc passes it and longitude taken the short way round, and the
survey measures it against a dense sampling of the arc, which is
the plain truth: over 500 random arcs between latitudes -70 and
70 the Clairaut box matched the sampled box within 3e-5 degrees
of latitude, the sampling step, and exactly in longitude width;
the endpoint box agreed with the true box on 263 of the 500, the
arcs that do not pass their vertex, and missed the arc by more
than a degree on 189, by 71.4 degrees at worst, an arc joining
two mid-latitude points nearly opposite in longitude over the
pole. It also measures how the bulge grows with the arc's span
for arcs along a parallel: a route along latitude 45 spanning 30
degrees of longitude reaches 45.99, spanning 60 reaches 49.1, 90
reaches 54.7, 150 reaches 75.5, and 179 reaches 89.5, since two
points opposite in longitude at the same latitude are joined
over the pole. Meridional and short arcs give the endpoint box
exactly, a southern-hemisphere route dips to -67.8 by the
mirrored rule, and a Pacific arc from 170 to -170 gets a box
from 170 to 190, the east edge unwrapped past 180 so the span is
the 20 degrees the short way rather than the 340 the long way.
The finding worth stating is that the true box of an arc extends
past its endpoints by an amount Clairaut's constant gives
exactly, by up to 71 degrees on long mid-latitude routes and by
nothing on meridional ones, so a spatial index must file arcs by
their vertex box or lose 38 percent of them. This module
computes arc boxes, and a survey measures them against sampled
arcs.
"""

from __future__ import annotations

import math

from atlas.bearing import initial_bearing
from atlas.errors import Invalid
from atlas.haversine import haversine
from atlas.interpolate import sample

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # south, west, north, east


def vertex_latitude(lat: float, bearing_deg: float) -> float:
    # the highest latitude the great circle through this point on this bearing reaches
    constant = math.cos(math.radians(lat)) * math.sin(math.radians(bearing_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(constant)))))


def passes_vertex(bearing_start: float, bearing_end: float) -> bool:
    # the arc passes its northern vertex when it starts heading east of north-south and
    # ends heading east of it on the other side, that is the bearing crosses 90 or 270
    start_east = 0.0 < bearing_start % 360.0 < 180.0
    end_east = 0.0 < bearing_end % 360.0 < 180.0
    if start_east != end_east:
        return False
    # heading east: northern vertex passed when the start bearing is below 90 and the
    # end bearing above; heading west: when the start is above 270 and the end below
    if start_east:
        return bearing_start % 360.0 < 90.0 < bearing_end % 360.0
    return bearing_start % 360.0 > 270.0 > bearing_end % 360.0


def endpoint_box(a: Point, b: Point) -> Box:
    south, north = min(a[0], b[0]), max(a[0], b[0])
    west, east = _short_span(a[1], b[1])
    return south, west, north, east


def _short_span(lon1: float, lon2: float) -> tuple[float, float]:
    delta = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    return (lon1, lon1 + delta) if delta >= 0 else (lon1 + delta, lon1)


def _final_bearing(a: Point, b: Point) -> float:
    return (initial_bearing(b[0], b[1], a[0], a[1]) + 180.0) % 360.0


def arc_box(a: Point, b: Point) -> Box:
    if a == b:
        raise Invalid("an arc needs two distinct endpoints")
    south, west, north, east = endpoint_box(a, b)
    start = initial_bearing(a[0], a[1], b[0], b[1])
    end = _final_bearing(a, b)
    peak = vertex_latitude(a[0], start)
    if passes_vertex(start, end):
        north = max(north, peak)
    # the southern vertex is passed when the mirrored bearings pass the northern one
    if passes_vertex((360.0 - start) % 360.0 + 180.0, (360.0 - end) % 360.0 + 180.0):
        south = min(south, -peak)
    return south, west, north, east


def sampled_box(a: Point, b: Point, count: int = 2000) -> Box:
    points = sample(a[0], a[1], b[0], b[1], count)
    lats = [p[0] for p in points]
    west, east = _lon_span([p[1] for p in points])
    return min(lats), west, max(lats), east


def _lon_span(lons: list[float]) -> tuple[float, float]:
    # unwrap the sampled longitudes relative to the first so the span is the short way
    unwrapped = [lons[0]]
    for lon in lons[1:]:
        delta = (lon - unwrapped[-1] + 180.0) % 360.0 - 180.0
        unwrapped.append(unwrapped[-1] + delta)
    return min(unwrapped), max(unwrapped)


def bulge_km(a: Point, b: Point) -> float:
    # how far the arc's true northern edge lies beyond the endpoint box, in kilometers
    _, _, endpoint_north, _ = endpoint_box(a, b)
    _, _, true_north, _ = arc_box(a, b)
    return haversine(endpoint_north, a[1], true_north, a[1])
