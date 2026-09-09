"""A south slope read against flat ground on the solstices, and a north slope in the dark.

The drill integrates the direct beam over the winter and summer
solstice days at latitude 45 on flat ground, on a south-facing slope
of 45 degrees, and on a north-facing slope of 30 degrees. The guess
before measuring was that a steep south slope would roughly double
the winter beam and hold level with flat ground in summer, since the
sun still crosses the slope's sky. The measurement refutes both
halves: the south slope reads 3.07 times the flat ground in winter
and 0.80 of it in summer, since the high summer sun strikes the
tilted plane obliquely. The north slope of 30 degrees reads exactly
zero on the winter solstice, because the noon sun sits at 21.56
degrees and never clears the slope's own horizon, while the same
slope in summer reads 37.06, six sevenths of flat ground. The
survey keeps the doubling guess beside the 3.07.
"""

from __future__ import annotations

from atlas.solarradiation import SOLSTICE_SUMMER, SOLSTICE_WINTER, daily_beam, noon_elevation
from atlas.surveys.survey import Survey


def run() -> Survey:
    winter_flat = daily_beam(45, SOLSTICE_WINTER)
    winter_south = daily_beam(45, SOLSTICE_WINTER, 45, 180)
    winter_north = daily_beam(45, SOLSTICE_WINTER, 30, 0)
    summer_flat = daily_beam(45, SOLSTICE_SUMMER)
    summer_south = daily_beam(45, SOLSTICE_SUMMER, 45, 180)
    summer_north = daily_beam(45, SOLSTICE_SUMMER, 30, 0)
    winter_ratio = winter_south / winter_flat
    summer_ratio = summer_south / summer_flat
    readings = {
        "winter_flat_mj": round(winter_flat, 3),
        "winter_south_45_mj": round(winter_south, 3),
        "winter_north_30_mj": round(winter_north, 3),
        "winter_south_over_flat": round(winter_ratio, 4),
        "summer_south_over_flat": round(summer_ratio, 4),
        "summer_north_30_mj": round(summer_north, 3),
        "winter_noon_elevation": round(noon_elevation(45, SOLSTICE_WINTER), 3),
    }
    holds = (
        abs(winter_ratio - 3.073) < 0.01
        and abs(summer_ratio - 0.796) < 0.01
        and winter_north == 0.0
        and summer_north > 0.8 * summer_flat
    )
    return Survey(
        surveyor="tiltwatch",
        finding=(
            "a 45 degree south slope at latitude 45 took 3.07 times the winter beam of "
            "flat ground and 0.80 of the summer beam, and a 30 degree north slope took "
            "none at all on the winter solstice"
        ),
        readings=readings,
        holds=holds,
    )
