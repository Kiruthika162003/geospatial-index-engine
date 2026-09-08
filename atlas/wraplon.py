"""Coordinate normalization: wrap longitude around the globe, reflect latitude over the pole.

Coordinates arrive dirty. A longitude of two hundred is a real place,
one hundred and sixty west, and a longitude of negative five hundred
and forty is that same place three turns further round; a latitude of
one hundred is not a real place, since latitude does not wrap, but it
is what you get by walking ten degrees past the pole, and the honest
reading of it is eighty degrees on the far side of the pole with the
longitude flipped by half a turn. Normalizing to the canonical ranges
is a small job with two traps the survey pins. For longitude, a naive
modulo into zero to three hundred and sixty and then shifting down
mishandles the seam: exactly one hundred and eighty must stay one
hundred and eighty, or map to negative one hundred and eighty, by a
fixed convention, not flip depending on floating-point residue, and
negative inputs must come out in range too, which a bare remainder
operator in some languages does not guarantee. The modulo of the
shifted value, longitude plus one hundred and eighty taken modulo
three hundred and sixty then shifted back, handles every case in one
expression with the convention that positive one hundred and eighty
maps to negative one hundred and eighty. For latitude, wrapping is
wrong, since the pole is an end, not a seam: going past ninety means
reflecting back to one hundred and eighty minus the value and turning
the longitude by half a turn, so the point stays where the walker
actually stands. The finding worth stating, as a measurement, is that
the normalizer is idempotent, applying it twice changes nothing, that
it preserves the actual location, measured by a great-circle distance
of zero between the raw and normalized readings for every input tried,
and that it treats the seam by a fixed rule. This module normalizes
longitude and latitude, and a survey checks idempotence and location
preservation across wild inputs.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid


def wrap_longitude(lon: float) -> float:
    if not math.isfinite(lon):
        raise Invalid("longitude must be finite")
    return (lon + 180.0) % 360.0 - 180.0


def normalize(lat: float, lon: float) -> tuple[float, float]:
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise Invalid("coordinates must be finite")
    # bring latitude into one full cycle of -180..180 first
    lat = (lat + 180.0) % 360.0 - 180.0
    if lat > 90.0:
        lat = 180.0 - lat  # over the north pole: reflect and flip longitude
        lon += 180.0
    elif lat < -90.0:
        lat = -180.0 - lat  # over the south pole
        lon += 180.0
    return (lat, wrap_longitude(lon))


def to_unit_vector(lat: float, lon: float) -> tuple[float, float, float]:
    # a location on the sphere independent of how its coordinates are written
    phi, lam = math.radians(lat), math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))
