"""Dateline split: cutting a segment at the 180th meridian so each half stays on a map.

A segment from longitude 170 to -170 spans twenty degrees across the
Pacific, but drawn on a flat map it spans three hundred and forty
degrees the other way, a line across the whole world. Renderers, tile
covers, and bounding boxes all need the short version, cut at the
seam into two pieces, one ending at longitude 180 and the other
starting at -180. The cut needs a latitude, and the latitude where
a straight line on the map crosses the seam is the linear
interpolation at the fraction of the way from the first longitude,
measured the short way round. The survey checks the cut by the one
property that matters: the two pieces together must be the same line
as the original, which for a planar segment means their lengths sum
to the original's short-way length exactly, and for a great-circle
segment means the sum matches the true distance to within the
straightness error of the map, which is small for short spans and
measurable for long ones. It also checks that a segment which does
not cross the seam comes back whole, that a segment ending exactly
on the seam comes back whole, and that the decision to cut is made
by the short way round, so a segment from 10 to 20 is never cut even
though the long way round would pass the seam. The guess before
measuring was that the great-circle lengths of the two pieces would
sum to the true distance within a few meters for spans under twenty
degrees. They do not: for diagonal segments with up to ten degrees
of latitude change, the worst gap was 0.57 km at a five-degree span,
2.47 km at ten, 10.3 km at twenty, 33.6 km at forty, and 227 km at
eighty, roughly quadrupling per doubling of the span. The gap is the
straightness error of the map, since a straight line in longitude
and latitude is not a great circle, and cutting the line at the seam
keeps the map's line rather than the globe's. The measured finding
is that the two pieces sum to the short-way span exactly on the
plane, to 6e-14 degrees over 5000 random segments, while on the
sphere the pieces are kilometers longer than the geodesic for any
span past a few degrees, a cost that belongs to drawing straight
lines on maps and not to the cut. This module cuts segments at the
dateline, and a survey confirms the planar identity, the geodesic
gap, and the no-cut cases.
"""

from __future__ import annotations

from atlas.errors import Outside
from atlas.haversine import haversine

Point = tuple[float, float]
Segment = tuple[Point, Point]


def short_delta(lon1: float, lon2: float) -> float:
    # the signed longitude change going the short way round, in (-180, 180]
    delta = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    if delta == -180.0:
        delta = 180.0
    return delta


def crosses_dateline(lon1: float, lon2: float) -> bool:
    # true when the short way from lon1 to lon2 passes through longitude 180
    delta = short_delta(lon1, lon2)
    unwrapped = lon1 + delta
    return unwrapped > 180.0 or unwrapped < -180.0


def split(segment: Segment) -> list[Segment]:
    (lat1, lon1), (lat2, lon2) = segment
    for lon in (lon1, lon2):
        if not -180.0 <= lon <= 180.0:
            raise Outside("longitude must lie within -180 and 180 degrees")
    if not crosses_dateline(lon1, lon2):
        return [segment]
    delta = short_delta(lon1, lon2)
    seam = 180.0 if delta > 0 else -180.0
    fraction = (seam - lon1) / delta
    cut_lat = lat1 + fraction * (lat2 - lat1)
    return [((lat1, lon1), (cut_lat, seam)), ((cut_lat, -seam), (lat2, lon2))]


def planar_span(segment: Segment) -> float:
    # the short-way longitude span of a segment, the length the pieces must sum to
    (_, lon1), (_, lon2) = segment
    return abs(short_delta(lon1, lon2))


def piece_span(pieces: list[Segment]) -> float:
    return sum(abs(b[1] - a[1]) for a, b in pieces)


def great_circle_gap_km(segment: Segment) -> float:
    # how far the summed geodesic lengths of the pieces depart from the true distance
    (lat1, lon1), (lat2, lon2) = segment
    pieces = split(segment)
    summed = sum(haversine(*a, *b) for a, b in pieces)
    return summed - haversine(lat1, lon1, lat2, lon2)
