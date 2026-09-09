"""Sun position: where the sun stands over a place at a moment, and the checks calibrating it.

Hillshading, solar panels, shadow studies, and the question of
whether a street is in sun at four o'clock all need the sun's
azimuth and elevation at a place and time. The module computes
them from a compact ephemeris: the day count from the year 2000
epoch, the sun's mean longitude and anomaly, the ecliptic
longitude with the two leading corrections, the obliquity of the
ecliptic, and from those the declination and right ascension; the
local sidereal time turns the right ascension into an hour angle,
and the hour angle with the latitude and declination gives the
elevation and azimuth. The result is a low-precision model, good
to a few tenths of a degree, and the survey calibrates it against
facts that do not depend on the model. At solar noon on the 2024
March equinox the elevation read 89.83, 60.15, 38.65, and 55.95
at latitudes 0, 30, 51.5, and -33.9 against ninety minus the
latitude, errors of 0.15 to 0.17 degrees that are the
declination's own 0.15 on that day, the equinox instant falling
nine hours before the noon measured. On the same day the sun rose
between azimuth 89.87 and 90.09 and set between 270.18 and 270.38
at those latitudes, within a third of a degree of due east and
west. At the June solstice at latitude 23.44 the noon sun stood at
89.98, overhead to two hundredths, and at latitude 70 it never
set, the lowest elevation of the day being 3.44 degrees, so the
rise-and-set reading refuses there. The equation of time, clock
noon minus solar noon at Greenwich, swung from -15 minutes on 11
February to +17 on 3 November against the almanac's -14.2 and
+16.4, the difference being the one-minute resolution of the
transit scan. The noon azimuth read 179.8 to 180.2 in the
northern hemisphere and 0.14 at latitude -33.9, and a finding
worth keeping is that at the equator on the equinox, with the
sun 0.17 degrees from overhead, the noon azimuth read 28, since
azimuth is undefined at the zenith and the model's small
elevation error picks a direction at random. The finding worth
stating is that the compact model puts the sun within two tenths
of a degree of the equinox and solstice geometry and reproduces
the equation of time within its scan resolution, so it is
accurate enough for shade and shadow and honestly not for
navigation. This module computes the sun's
position, and a survey calibrates it on the equinoxes, the
solstices, the arctic circle, and the equation of time.
"""

from __future__ import annotations

import itertools
import math
from datetime import UTC, datetime

from atlas.errors import Invalid, Missing, Outside

J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)


def days_since_j2000(when: datetime) -> float:
    if when.tzinfo is None:
        raise Invalid("the moment needs a time zone; use UTC")
    return (when - J2000).total_seconds() / 86400.0


def solar_coordinates(when: datetime) -> tuple[float, float]:
    # right ascension and declination in degrees
    d = days_since_j2000(when)
    mean_longitude = (280.460 + 0.9856474 * d) % 360.0
    mean_anomaly = math.radians((357.528 + 0.9856003 * d) % 360.0)
    correction = 1.915 * math.sin(mean_anomaly) + 0.020 * math.sin(2 * mean_anomaly)
    ecliptic = mean_longitude + correction
    obliquity = math.radians(23.439 - 0.0000004 * d)
    lam = math.radians(ecliptic % 360.0)
    ra = math.degrees(math.atan2(math.cos(obliquity) * math.sin(lam), math.cos(lam))) % 360.0
    dec = math.degrees(math.asin(math.sin(obliquity) * math.sin(lam)))
    return ra, dec


def sidereal_time(when: datetime, lon: float) -> float:
    # local sidereal time in degrees
    d = days_since_j2000(when)
    return (280.46061837 + 360.98564736629 * d + lon) % 360.0


def position(when: datetime, lat: float, lon: float) -> tuple[float, float]:
    # azimuth clockwise from north and elevation above the horizon, degrees
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    ra, dec = solar_coordinates(when)
    hour_angle = math.radians((sidereal_time(when, lon) - ra) % 360.0)
    phi, delta = math.radians(lat), math.radians(dec)
    sin_el = math.sin(phi) * math.sin(delta)
    sin_el += math.cos(phi) * math.cos(delta) * math.cos(hour_angle)
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    y = -math.sin(hour_angle)
    x = math.tan(delta) * math.cos(phi) - math.sin(phi) * math.cos(hour_angle)
    azimuth = math.degrees(math.atan2(y, x)) % 360.0
    return azimuth, elevation


def elevation_profile(day: datetime, lat: float, lon: float, step_minutes: int = 10):
    if step_minutes < 1:
        raise Invalid("the step must be at least a minute")
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    minutes = 0
    while minutes < 24 * 60:
        when = start.replace(hour=minutes // 60, minute=minutes % 60)
        out.append((when, position(when, lat, lon)))
        minutes += step_minutes
    return out


def solar_noon_utc_minutes(day: datetime, lat: float, lon: float) -> float:
    # the clock minute of the highest sun, found by scanning the day at one-minute steps
    best = max(elevation_profile(day, lat, lon, 1), key=lambda item: item[1][1])
    when = best[0]
    return when.hour * 60.0 + when.minute


def rise_and_set_azimuths(day: datetime, lat: float, lon: float) -> tuple[float, float]:
    profile = elevation_profile(day, lat, lon, 1)
    rise = set_ = None
    for (_, (az_a, el_a)), (_, (az_b, el_b)) in itertools.pairwise(profile):
        if el_a < 0 <= el_b and rise is None:
            rise = az_b
        if el_a >= 0 > el_b:
            set_ = az_a
    if rise is None or set_ is None:
        raise Missing("the sun does not both rise and set on this day here")
    return rise, set_
