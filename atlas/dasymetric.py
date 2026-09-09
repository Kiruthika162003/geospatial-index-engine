"""Class weights cut the areal error 11 to 22 fold; swapped weights are worse than none.

Areal weighting spreads a zone's total evenly over its cells;
dasymetric mapping spreads it in proportion to ancillary weights,
here a settled mask or a per-class density. On a synthetic town of
13,037 people living on 306 of 1600 cells, 69 of them in a dense
core, with sixteen source zones of 10 by 10 cells, every method
conserves the zone totals to 1e-10. The cell root mean square error
reads 15.51 for areal weighting, 11.24 for a binary settled mask, and
3.285 for class weights of 25 and 100 matching the two densities.
Re-aggregated to target zones of 5 by 5 cells the errors read 214.4,
142.9 and 18.5, and to 10 by 10 zones shifted by half a zone 489.4,
318.2 and 23.5; to 20 by 20 zones, which nest the sources exactly,
every method reads 0.0. The guess that any weighting beats none was
wrong: swapping the two class weights reads 22.15 by cell and 305.0
by target zone, worse than areal weighting, while a flat weight on
both classes is the binary mask again.

Corrupting a tenth, a quarter, half and all of the class labels at
random lifts the class-weighted target error from 18.5 to 52.5, 82.0,
123.1 and 165.4, and the binary mask's from 142.9 to 140.3, 157.4,
162.5 and 198.9. Coarser source zones hurt areal weighting and barely
touch the dasymetric estimate: at source blocks of 10, 20 and 40
cells the target errors read 214.4, 410.4 and 439.8 against 18.5,
18.6 and 20.0, ratios of 11.6, 22.0 and 22.0.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Grid = list[list[float]]
Zones = list[list[int]]
Mask = list[list[bool]]


def _check(grid: list[list]) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return len(grid), len(grid[0])


def zone_totals(truth: Grid, zones: Zones) -> dict[int, float]:
    rows, cols = _check(truth)
    totals: dict[int, float] = {}
    for r in range(rows):
        for c in range(cols):
            totals[zones[r][c]] = totals.get(zones[r][c], 0.0) + truth[r][c]
    return totals


def areal_weighting(totals: dict[int, float], zones: Zones) -> Grid:
    rows, cols = _check(zones)
    counts: dict[int, int] = {}
    for row in zones:
        for z in row:
            counts[z] = counts.get(z, 0) + 1
    out = []
    for r in range(rows):
        out.append([totals.get(zones[r][c], 0.0) / counts[zones[r][c]] for c in range(cols)])
    return out


def dasymetric(totals: dict[int, float], zones: Zones, weights: Grid) -> Grid:
    rows, cols = _check(zones)
    if (rows, cols) != _check(weights):
        raise Invalid("zones and weights must share a shape")
    if any(w < 0 for row in weights for w in row):
        raise Invalid("weights must not be negative")
    mass: dict[int, float] = {}
    counts: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            z = zones[r][c]
            mass[z] = mass.get(z, 0.0) + weights[r][c]
            counts[z] = counts.get(z, 0) + 1
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            z = zones[r][c]
            if mass[z] > 0:
                row.append(totals.get(z, 0.0) * weights[r][c] / mass[z])
            else:
                row.append(totals.get(z, 0.0) / counts[z])
        out.append(row)
    return out


def mask_weights(mask: Mask) -> Grid:
    return [[1.0 if v else 0.0 for v in row] for row in mask]


def class_weights(classes: list[list[int]], per_class: dict[int, float]) -> Grid:
    return [[per_class.get(v, 0.0) for v in row] for row in classes]


def aggregate(grid: Grid, zones: Zones) -> dict[int, float]:
    return zone_totals(grid, zones)


def block_zones(size: int, block: int, offset: int = 0) -> Zones:
    if block < 1 or size < 1:
        raise Invalid("size and block must be positive")
    per_row = math.ceil((size + offset) / block)
    out = []
    for r in range(size):
        band = ((r + offset) // block) * per_row
        out.append([band + (c + offset) // block for c in range(size)])
    return out


def rms_by_zone(estimate: Grid, truth: Grid, zones: Zones) -> float:
    est = aggregate(estimate, zones)
    real = aggregate(truth, zones)
    keys = set(est) | set(real)
    return math.sqrt(sum((est.get(k, 0.0) - real.get(k, 0.0)) ** 2 for k in keys) / len(keys))


def rms_by_cell(estimate: Grid, truth: Grid) -> float:
    rows, cols = _check(truth)
    total = sum((estimate[r][c] - truth[r][c]) ** 2 for r in range(rows) for c in range(cols))
    return math.sqrt(total / (rows * cols))


def total(grid: Grid) -> float:
    return sum(sum(row) for row in grid)


def synthetic_town(size: int, rng: random.Random) -> tuple[Grid, list[list[int]]]:
    if size < 8:
        raise Invalid("the town needs at least eight cells a side")
    classes = [[0] * size for _ in range(size)]
    truth = [[0.0] * size for _ in range(size)]
    cx, cy = size * 0.4, size * 0.55
    for r in range(size):
        for c in range(size):
            d = math.hypot(r - cy, c - cx)
            if d < size * 0.12:
                classes[r][c] = 2
                truth[r][c] = rng.uniform(80, 120)
            elif d < size * 0.3 and rng.random() < 0.6:
                classes[r][c] = 1
                truth[r][c] = rng.uniform(15, 35)
    return truth, classes


def corrupted(classes: list[list[int]], fraction: float, rng: random.Random) -> list[list[int]]:
    if not 0 <= fraction <= 1:
        raise Invalid("fraction must lie in [0, 1]")
    out = []
    for row in classes:
        fresh = []
        for v in row:
            fresh.append(rng.choice([0, 1, 2]) if rng.random() < fraction else v)
        out.append(fresh)
    return out
