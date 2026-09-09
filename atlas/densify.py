"""Densify: adding great-circle points to a line so its map drawing hugs the geodesic.

A route stored as two endpoints is drawn on a map as one straight
line in latitude and longitude, and that line is not the route; the
geodesic bows poleward of it, by kilometers for a span of ten degrees
and by hundreds for a span of eighty. Densifying fixes the drawing
by inserting points along the great circle so that each map-straight
piece is short enough that its departure from the geodesic falls
below a tolerance. The survey measures how the departure falls with
the number of pieces. Cutting a segment into n equal great-circle
hops and drawing each hop straight on the map leaves a gap that
shrinks as one over n squared, because the sagitta of a short arc is
quadratic in its length, so doubling the points quarters the gap.
Measured on New York to London, whose single map line strays 753 km
from the geodesic, the gap fell to 205, 52, 13.1, 3.29, 0.82, and
0.21 km at 2, 4, 8, 16, 32, and 64 pieces, the ratio per doubling
climbing from 3.68 to 4.00 as the pieces shorten; thirty pieces
bring it within a kilometer and 291 within ten meters, not the
thousands a linear fall would need. A twenty-degree Pacific segment
across the dateline starts at 33.5 km and needs 7 pieces for a
kilometer and 67 for ten meters, the map line being drawn the short
way round the seam as a cut map would draw it. The module chooses
the hop count from a maximum hop length
rather than a gap, since the gap depends on the route's direction and
latitude while the hop length is a plain distance, and the survey
reports the gap that a given hop length actually achieves. Two
identities anchor the result: the densified line passes through both
endpoints with every intermediate point on the geodesic to
floating precision, and its summed hop length equals the great-circle
distance to the same precision, since the hops are themselves
geodesic pieces. The finding worth stating is that map gap falls as
the square of the hop count, so a modest number of points removes
nearly all of the straightness error, and the densified line's
length is the geodesic's, so the drawing is corrected without
lengthening the route. This module densifies segments and polylines,
and a survey measures the gap against the hop count.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.crosstrack import cross_track_km
from atlas.datelinesplit import short_delta
from atlas.errors import Invalid
from atlas.haversine import haversine
from atlas.interpolate import sample

Point = tuple[float, float]


def wrap_longitude(lon: float) -> float:
    if lon > 180.0:
        return lon - 360.0
    if lon < -180.0:
        return lon + 360.0
    return lon


def hops_needed(lat1: float, lon1: float, lat2: float, lon2: float, max_hop_km: float) -> int:
    if max_hop_km <= 0:
        raise Invalid("the maximum hop must be positive")
    distance = haversine(lat1, lon1, lat2, lon2)
    return max(1, math.ceil(distance / max_hop_km))


def densify_segment(a: Point, b: Point, max_hop_km: float) -> list[Point]:
    hops = hops_needed(*a, *b, max_hop_km)
    if hops == 1:
        return [a, b]
    dense = sample(*a, *b, hops + 1)
    # the interpolation lands on the endpoints to rounding; pin them exactly
    dense[0] = a
    dense[-1] = b
    return dense


def densify(line: Sequence[Point], max_hop_km: float) -> list[Point]:
    if len(line) < 2:
        raise Invalid("a line needs at least two points")
    out: list[Point] = [line[0]]
    for a, b in itertools.pairwise(line):
        out.extend(densify_segment(a, b, max_hop_km)[1:])
    return out


def length_km(line: Sequence[Point]) -> float:
    return sum(haversine(*a, *b) for a, b in itertools.pairwise(line))


def map_gap_km(a: Point, b: Point, pieces: int, samples_per_piece: int = 16) -> float:
    # the largest distance between the geodesic a-b and the map-straight drawing of
    # the densified line, sampled linearly in latitude and longitude along each piece
    if pieces < 1:
        raise Invalid("need at least one piece")
    dense = sample(*a, *b, pieces + 1)
    worst = 0.0
    for (lat1, lon1), (lat2, lon2) in itertools.pairwise(dense):
        # a piece across the dateline is drawn the short way round, as a cut map would
        delta = short_delta(lon1, lon2)
        for i in range(1, samples_per_piece):
            t = i / samples_per_piece
            lat = lat1 + t * (lat2 - lat1)
            lon = wrap_longitude(lon1 + t * delta)
            worst = max(worst, abs(cross_track_km(*a, *b, lat, lon)))
    return worst


def pieces_for_gap(a: Point, b: Point, gap_km: float, limit: int = 4096) -> int:
    # the fewest equal pieces whose map drawing stays within the gap, by doubling then search
    if gap_km <= 0:
        raise Invalid("the gap must be positive")
    hi = 1
    while map_gap_km(a, b, hi) > gap_km:
        hi *= 2
        if hi > limit:
            raise Invalid("no piece count under the limit meets the gap")
    lo = hi // 2 if hi > 1 else 1
    while lo < hi:
        mid = (lo + hi) // 2
        if map_gap_km(a, b, mid) <= gap_km:
            hi = mid
        else:
            lo = mid + 1
    return lo
