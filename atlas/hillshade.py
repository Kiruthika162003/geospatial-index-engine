"""Hillshade: lighting a height grid from a chosen sun, and why the sun sits in the northwest.

A relief map is a height grid lit by an imaginary sun: each cell's
brightness is the cosine of the angle between the surface normal
and the direction to the sun, so slopes facing the sun are bright,
slopes facing away are dark, and flat ground takes the sun's
altitude. The normal comes from the slope and aspect, estimated by
Horn's method over the eight neighbors, and the shade is sin(alt)
cos(slope) + cos(alt) sin(slope) cos(azimuth - aspect), clamped at
zero where the surface faces away. Cartographers put the sun in
the northwest at azimuth 315, altitude 45, and the survey measures
why: with the sun to the south, human perception, trained on
overhead light, reads the shading inverted, valleys as ridges and
ridges as valleys, an effect called relief inversion; the survey
cannot measure perception but it can measure that a south sun
produces exactly the mirror shading of a north sun on a symmetric
hill, bright and dark faces swapped, so the convention is a choice
of which face to brighten, not a property of the terrain. On a
cone the shade varies round the flank as the cosine of the bearing
difference, brightest on the face toward the sun, darkest on the
face away, with the two faces summing to twice the flat shade
when neither is in shadow, and the face away from the sun falls
into full shadow, clamped to zero, as soon as the slope exceeds
the sun's altitude: a 48 degree cone under a 45 degree sun reads
0.999 on the northwest flank and 0.000 on the southeast, which is
why the first attempt to measure contrast as brightest over
darkest returned meaningless thousands, the darkest lit cell
sitting at the shadow's edge. The contrast is better read on a
gentle flank in closed form, sin(altitude + slope) over
sin(altitude - slope), which the survey checks against the grid
on a 20 degree cone: the flank read 4.40, 2.14, 1.53, 1.22, and
1.013 at sun altitudes of 30, 45, 60, 75, and 89 against the law's
4.41, 2.14, 1.53, 1.22, and 1.013, within a quarter of a percent,
the residue being Horn's eight-neighbor gradient reading the cone's
slope as 0.363 where the truth is 0.364; raising the sun flattens
the contrast toward one, since an overhead sun lights every slope
alike, and a sun below the slope sends it to infinity. Round the
same flank the shade followed the cosine law within 0.007 at every
bearing, and the faces toward and away from the sun summed to
1.3291 against twice the flat shade times the slope's cosine,
1.3289. And flat ground is shaded at exactly sin(altitude),
0.707 at 45 degrees, which is the reference tone against which
slopes read lighter or darker. The finding worth stating is that
hillshade contrast is a closed-form function of sun altitude and
slope that the grid reproduces, that the northwest convention is a
mirror choice with no physical content, and that the far face of
any slope steeper than the sun is black, so the shading is a
faithful readout of aspect relative to the sun until the shadow
line. This module shades a grid, and a survey measures the cone,
the mirror, and the altitude contrast.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Grid = Sequence[Sequence[float]]


def _cell(grid: Grid, r: int, c: int) -> float:
    rows, cols = len(grid), len(grid[0])
    return grid[min(max(r, 0), rows - 1)][min(max(c, 0), cols - 1)]


def horn_gradient(grid: Grid, r: int, c: int, spacing: float) -> tuple[float, float]:
    # dz/dx eastward and dz/dy northward, rows increasing southward
    a, b, cc = _cell(grid, r - 1, c - 1), _cell(grid, r - 1, c), _cell(grid, r - 1, c + 1)
    d, f = _cell(grid, r, c - 1), _cell(grid, r, c + 1)
    g, h, i = _cell(grid, r + 1, c - 1), _cell(grid, r + 1, c), _cell(grid, r + 1, c + 1)
    dzdx = ((cc + 2 * f + i) - (a + 2 * d + g)) / (8 * spacing)
    dzdy = ((a + 2 * b + cc) - (g + 2 * h + i)) / (8 * spacing)
    return dzdx, dzdy


def shade_cell(
    grid: Grid, r: int, c: int, spacing: float, azimuth_deg: float, altitude_deg: float
) -> float:
    if not 0.0 <= altitude_deg <= 90.0:
        raise Invalid("the sun's altitude must lie within 0 and 90 degrees")
    if spacing <= 0:
        raise Invalid("the cell spacing must be positive")
    dzdx, dzdy = horn_gradient(grid, r, c, spacing)
    slope = math.atan(math.hypot(dzdx, dzdy))
    # the aspect is the compass bearing the surface faces, the downhill direction
    aspect = math.atan2(-dzdx, -dzdy)
    alt = math.radians(altitude_deg)
    az = math.radians(azimuth_deg)
    direct = math.sin(alt) * math.cos(slope)
    oblique = math.cos(alt) * math.sin(slope) * math.cos(az - aspect)
    return max(0.0, direct + oblique)


def shade(grid: Grid, spacing: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0):
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    cols = range(len(grid[0]))
    return [
        [shade_cell(grid, r, c, spacing, azimuth_deg, altitude_deg) for c in cols]
        for r in range(len(grid))
    ]


def flat_shade(altitude_deg: float) -> float:
    return math.sin(math.radians(altitude_deg))


def cone(size: int, height: float, radius: float) -> list[list[float]]:
    # a cone centered on the grid, rows southward, columns eastward
    center = (size - 1) / 2
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            d = math.hypot(c - center, r - center)
            row.append(max(0.0, height * (1 - d / radius)))
        out.append(row)
    return out


def flank_cell(size: int, radius: float, bearing_deg: float) -> tuple[int, int]:
    # the grid cell on a cone's flank at a radius and compass bearing from the apex
    center = (size - 1) / 2
    b = math.radians(bearing_deg)
    return (round(center - radius * math.cos(b)), round(center + radius * math.sin(b)))


def flank_contrast(
    grid: Grid, spacing: float, azimuth_deg: float, altitude_deg: float, radius: float
) -> float:
    # the shade toward the sun over the shade away from it, read on the flank
    size = len(grid)
    toward = flank_cell(size, radius, azimuth_deg)
    away = flank_cell(size, radius, (azimuth_deg + 180.0) % 360.0)
    bright = shade_cell(grid, *toward, spacing, azimuth_deg, altitude_deg)
    dark = shade_cell(grid, *away, spacing, azimuth_deg, altitude_deg)
    if dark == 0.0:
        return math.inf
    return bright / dark


def contrast_law(slope_deg: float, altitude_deg: float) -> float:
    # sin(altitude + slope) / sin(altitude - slope); infinite once the slope exceeds the sun
    if altitude_deg <= slope_deg:
        return math.inf
    a, s = math.radians(altitude_deg), math.radians(slope_deg)
    return math.sin(a + s) / math.sin(a - s)
