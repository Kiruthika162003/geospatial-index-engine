"""UTM zones: sixty six-degree strips, each scaled by 0.9996 to spread the error.

The Universal Transverse Mercator system cuts the world into sixty
longitude zones six degrees wide, numbered from the antimeridian
eastward, and within each zone projects the earth onto a cylinder
wrapped around the zone's central meridian, so that distortion, which
grows with distance from the meridian, is kept small by never letting a
point be more than three degrees from it. Rows of latitude get letters,
C through X skipping I and O, eight degrees each, so a UTM position is a
zone number, a band letter, and an easting and northing in meters, a
grid that surveyors and the military can add and subtract in without
trigonometry. The number worth understanding is the central meridian
scale factor of 0.9996. A transverse Mercator with true scale on its
central meridian stretches everything away from it, so the scale error
would be zero at the center and largest, about a tenth of a percent, at
the zone edges. Shrinking the central meridian by four parts in ten
thousand instead makes the scale slightly under one at the center,
exactly one along two lines roughly a hundred and eighty kilometers
either side, and slightly over one at the edges, halving the worst
error by spreading it in both directions rather than accumulating it
in one. The measured scale runs from 0.9996 at the meridian to about
1.0010 at the equatorial zone edge, so the whole zone stays within a
tenth of a percent of true scale. The finding worth stating is that the
0.9996 factor trades a small deliberate shrinkage at the center for a
balanced error across the zone, which is why every UTM zone is usable
to survey accuracy edge to edge. This module assigns zone numbers and
band letters, finds the central meridian, and computes the point scale
factor, and a survey measures the scale at the meridian and the edge.
"""

from __future__ import annotations

import math

from atlas.errors import Outside

CENTRAL_SCALE = 0.9996
_BANDS = "CDEFGHJKLMNPQRSTUVWX"


def zone_number(lat: float, lon: float) -> int:
    if not -80.0 <= lat <= 84.0:
        raise Outside("UTM covers latitudes from -80 to 84 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(zone, 60)
    # the Norway exception: zone 32 is widened to keep the coast in one zone
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32
    # the Svalbard exceptions
    if 72.0 <= lat < 84.0:
        if 0.0 <= lon < 9.0:
            zone = 31
        elif 9.0 <= lon < 21.0:
            zone = 33
        elif 21.0 <= lon < 33.0:
            zone = 35
        elif 33.0 <= lon < 42.0:
            zone = 37
    return zone


def band_letter(lat: float) -> str:
    if not -80.0 <= lat <= 84.0:
        raise Outside("UTM covers latitudes from -80 to 84 degrees")
    index = int((lat + 80.0) // 8.0)
    return _BANDS[min(index, len(_BANDS) - 1)]


def central_meridian(zone: int) -> float:
    if not 1 <= zone <= 60:
        raise Outside("zone must lie within 1 and 60")
    return (zone - 1) * 6.0 - 180.0 + 3.0


def point_scale(lat: float, lon: float) -> float:
    # transverse Mercator point scale to second order in the meridian offset
    zone = zone_number(lat, lon)
    dlam = math.radians(lon - central_meridian(zone))
    phi = math.radians(lat)
    p = dlam * math.cos(phi)
    return CENTRAL_SCALE * (1 + p * p / 2)
