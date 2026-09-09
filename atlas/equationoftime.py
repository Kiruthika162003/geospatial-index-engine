"""The sundial runs 16.5 minutes fast on day 303 and 14.6 slow on day 44, a 31 minute analemma.

The equation of time is the difference between the sun's crossing
of the meridian and clock noon, from the tilt of the axis and the
ellipse of the orbit. The three-term approximation used here peaks
at +16.45 minutes on day 303, the end of October, and bottoms at
-14.60 on day 44, the middle of February, crossing zero on days
106, 165, 243 and 358. Plotted against the declination it draws the
analemma, 31.05 minutes wide, whose two branches cross where the
zero crossings of April and August sit 1.68 degrees apart in
declination, near 9 degrees north; the other two crossings sit at
the solstices, 23.3 and -23.4.

The guess that the approximation agrees with a scan of the sun's
elevation to the minute was wrong by a minute: against the solar
noon found by scanning each day of 2024 at one-minute steps it
disagrees by 0.584 minutes on average and 1.786 at worst, the scan
reading 16 on day 306 where the formula reads 16.40 and -14 on day
45 against -14.59, since the scan itself is quantised to the minute
and the formula drops the smaller harmonics. Clock noon and solar
noon part by the zone as well: on day 45 at Greenwich the sun
crosses 14.59 minutes after noon; on day 172 in Berlin, 13.4 east in
zone 1, 7.9 minutes after; in Madrid, 3.7 west in the same zone,
76.3 minutes after, and 58.35 on day 303.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from atlas.errors import Invalid
from atlas.sunposition import solar_noon_utc_minutes


def equation_of_time_minutes(day_of_year: int) -> float:
    if not 1 <= day_of_year <= 366:
        raise Invalid("the day of year runs from 1 to 366")
    b = 2 * math.pi * (day_of_year - 81) / 364
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def declination(day_of_year: int) -> float:
    if not 1 <= day_of_year <= 366:
        raise Invalid("the day of year runs from 1 to 366")
    return 23.44 * math.sin(math.radians(360 * (284 + day_of_year) / 365))


def solar_noon_clock_offset_minutes(day_of_year: int, lon: float, zone_hours: float) -> float:
    # minutes after 12:00 clock time at which the sun crosses the meridian
    if not -180 <= lon <= 180:
        raise Invalid("longitude runs from -180 to 180")
    zone_meridian = 15 * zone_hours
    return 4 * (zone_meridian - lon) - equation_of_time_minutes(day_of_year)


def extremes(year_days: int = 365) -> tuple[tuple[int, float], tuple[int, float]]:
    values = [(day, equation_of_time_minutes(day)) for day in range(1, year_days + 1)]
    high = max(values, key=lambda v: v[1])
    low = min(values, key=lambda v: v[1])
    return high, low


def zero_crossings(year_days: int = 365) -> list[int]:
    days = []
    previous = equation_of_time_minutes(1)
    for day in range(2, year_days + 1):
        current = equation_of_time_minutes(day)
        if (previous < 0) != (current < 0):
            days.append(day)
        previous = current
    return days


def analemma(step_days: int = 1) -> list[tuple[float, float]]:
    if step_days < 1:
        raise Invalid("the step must be positive")
    return [
        (equation_of_time_minutes(day), declination(day)) for day in range(1, 366, step_days)
    ]


def analemma_width_minutes() -> float:
    points = analemma()
    return max(p[0] for p in points) - min(p[0] for p in points)


def analemma_crossing_declination() -> float:
    # the declination where the figure crosses itself: the two zero crossings of the
    # equation of time closest in declination
    crossings = zero_crossings()
    decls = [declination(d) for d in crossings]
    best = math.inf
    for i in range(len(decls)):
        for j in range(i + 1, len(decls)):
            best = min(best, abs(decls[i] - decls[j]))
    return best


def measured_from_positions(year: int, lon: float = 0.0) -> list[tuple[int, float]]:
    out = []
    start = datetime(year, 1, 1, tzinfo=UTC)
    for day in range(1, 366):
        when = start + timedelta(days=day - 1)
        noon = solar_noon_utc_minutes(when, 0.0, lon)
        out.append((day, 720.0 - noon))
    return out


def worst_disagreement(year: int) -> tuple[float, float]:
    measured = measured_from_positions(year)
    worst = 0.0
    total = 0.0
    for day, minutes in measured:
        diff = minutes - equation_of_time_minutes(day)
        worst = max(worst, abs(diff))
        total += abs(diff)
    return worst, total / len(measured)
