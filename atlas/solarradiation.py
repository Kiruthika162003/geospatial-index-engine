"""A 45 degree south slope triples the winter beam at latitude 45 and loses a fifth in summer.

Direct beam radiation on a surface is the solar constant times the
cosine of the incidence angle, integrated over the daylight minutes
and, when an atmosphere is asked for, cut by a transmittance raised
to the air mass. On the winter solstice at latitude 45 flat ground
receives 10.07 megajoules a square metre a day and a south-facing
slope of 45 degrees 30.93, 3.07 times as much; at the equinox the
ratio is 1.43 and at the summer solstice 0.796, since the high sun
strikes the tilted plane obliquely. A north-facing slope of 30
degrees at that latitude receives no direct beam at all on the
winter solstice, because the noon elevation is 21.56 degrees; a
slope of 25 is dark too and one of 20 is not. At latitude 65 the
winter day lasts 2.88 hours with a noon elevation of 1.56 degrees,
flat ground receives 0.256 and the south slope 10.03, 39 times as
much. At the equator the same south slope receives 1.19 times the
flat ground in December and 0.293 times in June.

The guess that the best annual tilt equals the latitude held only
without an atmosphere, and then within five degrees: 0, 30, 40 and
55 at latitudes 0, 30, 45 and 60. With a transmittance of 0.7 the
best tilts fall to 0, 25, 35 and 45, since the air mass punishes the
low winter sun that a steep tilt was chasing. A sweep on every
seventh day at ten minute steps reads 0, 30, 45 and 55 dry and 0,
25, 35 and 45 hazy, the tilts of 40 and 45 at latitude 45 too close
to separate at that resolution. Annual flat totals
read 13,096, 9,679, 6,752 and 5,486 at latitudes 0, 45, 65 and 90
without air, the pole at 42 percent of the equator, and 7,818,
4,980, 2,921 and 1,710 with air, the pole at 22 percent. The two
minute step reads 34.3412 against 34.3413 at one minute; a one hour
step reads 34.145, 0.57 percent low.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

SOLAR_CONSTANT = 1361.0
Vector = tuple[float, float, float]


def declination(day_of_year: int) -> float:
    if not 1 <= day_of_year <= 366:
        raise Invalid("the day of year runs from 1 to 366")
    return 23.44 * math.sin(math.radians(360 * (284 + day_of_year) / 365))


def sun_vector(lat: float, decl: float, hour_angle: float) -> Vector:
    # east, north, up components of the unit vector toward the sun
    phi, delta, omega = map(math.radians, (lat, decl, hour_angle))
    up = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(omega)
    east = -math.cos(delta) * math.sin(omega)
    north = math.cos(phi) * math.sin(delta) - math.sin(phi) * math.cos(delta) * math.cos(omega)
    return east, north, up


def surface_normal(slope: float, aspect: float) -> Vector:
    if not 0 <= slope <= 90:
        raise Invalid("the slope runs from 0 to 90 degrees")
    beta, gamma = math.radians(slope), math.radians(aspect)
    return math.sin(beta) * math.sin(gamma), math.sin(beta) * math.cos(gamma), math.cos(beta)


def elevation(lat: float, decl: float, hour_angle: float) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, sun_vector(lat, decl, hour_angle)[2]))))


def air_mass(elev: float) -> float:
    if elev <= 0:
        return math.inf
    return 1 / math.sin(math.radians(elev))


def daily_beam(
    lat: float,
    day_of_year: int,
    slope: float = 0.0,
    aspect: float = 180.0,
    transmittance: float | None = None,
    step_minutes: int = 2,
) -> float:
    if not -90 <= lat <= 90:
        raise Invalid("latitude runs from -90 to 90")
    if step_minutes <= 0 or 1440 % step_minutes:
        raise Invalid("the step must divide a day")
    if transmittance is not None and not 0 < transmittance <= 1:
        raise Invalid("transmittance is a fraction")
    decl = declination(day_of_year)
    normal = surface_normal(slope, aspect)
    total = 0.0
    for minute in range(0, 1440, step_minutes):
        hour_angle = (minute / 4.0) - 180.0
        sun = sun_vector(lat, decl, hour_angle)
        if sun[2] <= 0:
            continue
        incidence = sum(s * n for s, n in zip(sun, normal, strict=True))
        if incidence <= 0:
            continue
        strength = SOLAR_CONSTANT
        if transmittance is not None:
            strength *= transmittance ** air_mass(math.degrees(math.asin(sun[2])))
        total += strength * incidence * step_minutes * 60
    return total / 1e6


def day_length_hours(lat: float, day_of_year: int) -> float:
    phi, delta = math.radians(lat), math.radians(declination(day_of_year))
    cos_omega = -math.tan(phi) * math.tan(delta)
    if cos_omega <= -1:
        return 24.0
    if cos_omega >= 1:
        return 0.0
    return 2 * math.degrees(math.acos(cos_omega)) / 15


def annual_beam(
    lat: float, slope: float = 0.0, aspect: float = 180.0, day_step: int = 1, **kwargs
) -> float:
    if day_step <= 0:
        raise Invalid("the day step must be positive")
    days = range(1, 366, day_step)
    return sum(daily_beam(lat, day, slope, aspect, **kwargs) for day in days) * 365 / len(days)


def best_tilt(
    lat: float, aspect: float = 180.0, step: int = 5, day_step: int = 1, **kwargs
) -> tuple[int, float]:
    if step <= 0:
        raise Invalid("the tilt step must be positive")
    best = (0, -1.0)
    for tilt in range(0, 91, step):
        total = annual_beam(lat, tilt, aspect, day_step, **kwargs)
        if total > best[1]:
            best = (tilt, total)
    return best


def noon_elevation(lat: float, day_of_year: int) -> float:
    return elevation(lat, declination(day_of_year), 0.0)


def shaded_all_day(lat: float, day_of_year: int, slope: float, aspect: float) -> bool:
    return daily_beam(lat, day_of_year, slope, aspect) == 0.0


SOLSTICE_SUMMER = 172
SOLSTICE_WINTER = 355
EQUINOX_SPRING = 80
