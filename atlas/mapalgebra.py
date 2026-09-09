"""Map algebra: arithmetic on rasters cell by cell and over neighbourhoods, bias measured.

A raster analysis is a sequence of algebra: elevation minus base
level, rainfall times a runoff coefficient, a land-cover class
mapped to a friction value, a focal mean to smooth noise, a
focal range to find edges. The local operations act cell by cell
and carry no surprises beyond the treatment of missing cells,
which propagate through arithmetic as missing and are skipped by
reclassification. The focal operations act over a window of
cells and carry a bias that the survey measures. A focal mean
over a square window of radius r reproduces a plane exactly,
since the window's cells are symmetric about its center and the
plane's deviations cancel, and the survey reads the error at
zero to floating precision away from the edges. On a curved
surface the focal mean is biased by the curvature times the
mean squared offset of the window's cells, which for a square
window of radius r is r (r + 1) over 3 in each axis: a bowl of
curvature c in both axes is read high by c r (r + 1) / 3, so a
radius-one window on a bowl with second derivative 2 reads 4/3
high everywhere, and the survey confirms the number to nine
places at radius one, 4.0 at radius two, and 8.0 at three, with
the plane's interior error at 2e-15. At the edges the window is
truncated, and the guess about the two rules was half wrong.
Skipping the missing cells biases an edge cell of a plane by half
the slope times the radius, as guessed: the left edge of a plane
sloping 0.5 per cell read -0.75 where the truth was -1.0, a
quarter high. Padding with the edge value was guessed to bias it
the other way, and it does not: the same cell read -0.833, a
sixth high, the same direction at two thirds the size, since the
padded copies of the edge column pull the mean toward the edge
value, which is itself on the low side of the window's center;
at the top edge the two rules read 6.85 and 6.90 against 7.0, and
at the corner 2.10 and 2.067 against 2.0. The focal range on a
plane was guessed at 2r times the slope's larger component and is
2r times the sum of both components, 3.2 at radius two for
slopes of 0.5 and 0.3, since the window's extremes sit at
opposite corners, and the focal maximum of a plane is the plane
at the window's far corner, exact on every interior cell. The
finding worth stating is that a focal mean is exact on planes
and biased on curved surfaces by curvature times r (r + 1) / 3
exactly, and that both edge rules bias it the same way, padding
by two thirds as much, so a smoothed raster is a smoothed plane
plus a curvature term and an edge band a map must account for.
This module implements local and focal operations, and a survey
measures the curvature bias and the edge rules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from atlas.errors import Invalid

Raster = Sequence[Sequence[float | None]]


def _check(raster: Raster) -> tuple[int, int]:
    if not raster or not raster[0]:
        raise Invalid("the raster is empty")
    return len(raster), len(raster[0])


def local(
    op: Callable[[float, float], float], a: Raster, b: Raster
) -> list[list[float | None]]:
    rows, cols = _check(a)
    if (rows, cols) != _check(b):
        raise Invalid("the rasters differ in shape")

    def combine(x: float | None, y: float | None) -> float | None:
        return None if x is None or y is None else op(x, y)

    return [[combine(a[r][c], b[r][c]) for c in range(cols)] for r in range(rows)]


def scale(raster: Raster, factor: float, offset: float = 0.0) -> list[list[float | None]]:
    _check(raster)
    return [[None if v is None else v * factor + offset for v in row] for row in raster]


def reclassify(raster: Raster, breaks: Sequence[float], classes: Sequence[float]):
    # value below the first break gets classes[0], between breaks i-1 and i gets classes[i]
    _check(raster)
    if len(classes) != len(breaks) + 1:
        raise Invalid("need one more class than breaks")
    if list(breaks) != sorted(breaks):
        raise Invalid("breaks must ascend")

    def classify(v: float | None) -> float | None:
        if v is None:
            return None
        for i, b in enumerate(breaks):
            if v < b:
                return classes[i]
        return classes[-1]

    return [[classify(v) for v in row] for row in raster]


def _window(raster: Raster, r: int, c: int, radius: int, pad: bool) -> list[float]:
    rows, cols = len(raster), len(raster[0])
    values = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if pad:
                rr, cc = min(max(rr, 0), rows - 1), min(max(cc, 0), cols - 1)
            elif not (0 <= rr < rows and 0 <= cc < cols):
                continue
            v = raster[rr][cc]
            if v is not None:
                values.append(v)
    return values


def focal(
    raster: Raster, radius: int, reducer: Callable[[list[float]], float], pad: bool = False
) -> list[list[float | None]]:
    rows, cols = _check(raster)
    if radius < 1:
        raise Invalid("the window radius must be at least one")
    out: list[list[float | None]] = []
    for r in range(rows):
        row: list[float | None] = []
        for c in range(cols):
            values = _window(raster, r, c, radius, pad)
            row.append(reducer(values) if values else None)
        out.append(row)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def value_range(values: list[float]) -> float:
    return max(values) - min(values)


def curvature_bias(curvature: float, radius: int) -> float:
    # the focal mean's excess on a surface with second derivative `curvature` in both axes
    return curvature * radius * (radius + 1) / 3.0


def sampled(field: Callable[[float, float], float], rows: int, cols: int):
    return [[field(float(c), float(r)) for c in range(cols)] for r in range(rows)]
