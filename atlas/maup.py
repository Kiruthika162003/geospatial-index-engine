"""The modifiable areal unit problem: the same points, different cells, different answers.

Aggregate individual records to areas, say income and rent by
neighbourhood, and the correlation between the two aggregated
variables depends on the areas chosen: coarser cells give a
higher correlation, since averaging over larger areas smooths
away the individual scatter that dilutes it, and shifting the
cell grid by half a cell gives a different correlation again
with the same cells at the same size. This is the modifiable
areal unit problem, and the survey measures both of its faces on
5000 points built to have a known structure: each carries two
values sharing a smooth spatial component, the second twice the
first, plus independent noise of 1.5, so their correlation at the
point level was 0.42. Aggregated to square cells the correlation
of the cell means read 0.559 at cells of 2 units, 0.889 at 5,
0.970 at 10, 0.990 at 20, 0.964 at 33, and 0.997 at 50, rising
toward one as the noise averages out and the shared component
remains, the dip at 33 coming from the ragged fifteen cells that
size leaves. Holding the cell at 10 and shifting the grid's origin
by tenths of a cell moved the correlation between 0.854 and
0.970, a spread of 0.116, and at 20 between 0.951 and 0.990, a
spread of 0.039, the zoning face, smaller than the scale face
but not small. The guess about the regression slope was wrong in
an instructive way: the guess was that the point-level slope is
the true slope and aggregation drifts it, but the point-level
slope read 0.586 against a true 2.0, attenuated by the noise in
the predictor, the errors-in-variables bias, and aggregation
restored it, 0.848 at cells of 2, 1.68 at 5, 1.92 at 10, 1.98 at
20, and 2.03 at 50, since averaging shrinks the predictor's noise
faster than its signal. The accounting held to 1e-16: the mean of
every variable over all points equalled the count-weighted mean
of its cell means at every size. The finding worth stating is
that aggregation raises a point-level correlation of 0.42 to 0.99
at twenty-unit cells and the grid's placement alone moves it by
0.12 at ten-unit cells, and that the same aggregation undoes the
attenuation of a noisy predictor's slope from 0.59 toward its
true 2, so a correlation between areal aggregates is a property
of the areas as much as of the people in them, and a slope from
individual records can be the more misleading of the two. This
module aggregates point values to cells and reads the
correlation and slope, and a survey measures the scale and zoning
effects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Record = tuple[float, float, float, float]  # x, y, first value, second value


def correlation(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n != len(b) or n < 3:
        raise Invalid("correlation needs two equal sequences of at least three values")
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        raise Invalid("a constant sequence has no correlation")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (sa * sb)


def slope(a: Sequence[float], b: Sequence[float]) -> float:
    # the least-squares change in b per unit of a
    n = len(a)
    if n != len(b) or n < 2:
        raise Invalid("a slope needs two equal sequences of at least two values")
    ma, mb = sum(a) / n, sum(b) / n
    sxx = sum((x - ma) ** 2 for x in a)
    if sxx == 0:
        raise Invalid("a constant predictor has no slope")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / sxx


def aggregate(records: Sequence[Record], cell: float, shift: float = 0.0):
    # cell means of both values, with the count per cell, keyed by cell
    if cell <= 0:
        raise Invalid("the cell size must be positive")
    sums: dict[tuple[int, int], list[float]] = {}
    for x, y, u, v in records:
        key = (math.floor((x + shift) / cell), math.floor((y + shift) / cell))
        entry = sums.setdefault(key, [0.0, 0.0, 0])
        entry[0] += u
        entry[1] += v
        entry[2] += 1
    return {key: (s[0] / s[2], s[1] / s[2], s[2]) for key, s in sums.items()}


def cell_correlation(records: Sequence[Record], cell: float, shift: float = 0.0) -> float:
    cells = aggregate(records, cell, shift)
    return correlation([c[0] for c in cells.values()], [c[1] for c in cells.values()])


def cell_slope(records: Sequence[Record], cell: float, shift: float = 0.0) -> float:
    cells = aggregate(records, cell, shift)
    return slope([c[0] for c in cells.values()], [c[1] for c in cells.values()])


def weighted_mean_identity(records: Sequence[Record], cell: float) -> tuple[float, float]:
    # the gap between the point mean and the count-weighted mean of cell means, per value
    cells = aggregate(records, cell)
    n = len(records)
    weighted_u = sum(u * c for u, _, c in cells.values()) / n
    weighted_v = sum(v * c for _, v, c in cells.values()) / n
    mean_u = sum(r[2] for r in records) / n
    mean_v = sum(r[3] for r in records) / n
    return abs(weighted_u - mean_u), abs(weighted_v - mean_v)


def synthetic(count: int, noise: float, rng, extent: float = 100.0) -> list[Record]:
    # two values sharing a smooth component of unit scale plus independent noise
    out = []
    for _ in range(count):
        x, y = rng.uniform(0, extent), rng.uniform(0, extent)
        shared = math.sin(x / 15.0) + math.cos(y / 12.0)
        out.append((x, y, shared + rng.gauss(0, noise), 2.0 * shared + rng.gauss(0, noise)))
    return out
