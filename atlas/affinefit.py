"""Three points fit an affine map exactly; leave-one-out on four reads twelve times the noise.

Georeferencing fits a map from image to ground coordinates through
control points, here a six-parameter affine map by least squares or
a four-parameter similarity of scale, rotation and shift. Control
points jittered by a sigma of 1 on each axis leave a fit residual
that follows root two times sigma times root of one minus the
parameters over twice the points: three points read 0.0 exactly,
since three fix the six parameters, and 4, 6, 10, 20 and 50 points
read 0.719, 0.998, 1.156, 1.277 and 1.391 against the law's 0.707,
1.0, 1.183, 1.304 and 1.371, rising toward the noise's own root two.
The guess that leaving one point out and predicting it is always the
honest error was wrong at small counts: with four points each fit
runs through three noisy points and extrapolates, reading 16.5, and
six points read 2.45, ten 1.68, twenty 1.50 and fifty 1.48, settling
toward 1.41 from above. Ten points recover a scale of 1.9862 against
1.9875 and a rotation of 29.96 degrees against 30.

The similarity fit cannot absorb a shear: on the same sheared truth
its residual reads 3.97, 6.48, 8.55, 9.16, 9.84 and 10.12 at 3 to 50
points, far over the four-parameter law of 0.82 to 1.39, and on
clean points a shear of 0.05 leaves 10.19 and 0.2 leaves 40.8 where
the affine fit leaves 1e-12; with no shear both leave 0. Collinear
control points are refused, since they fix no transform.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Affine = tuple[float, float, float, float, float, float]


def apply(t: Affine, p: Point) -> Point:
    a, b, c, d, e, f = t
    return a * p[0] + b * p[1] + c, d * p[0] + e * p[1] + f


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    m = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            raise Invalid("the control points do not fix the transform")
        for r in range(n):
            if r != col:
                factor = m[r][col] / m[col][col]
                for k in range(col, n + 1):
                    m[r][k] -= factor * m[col][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_affine(source: list[Point], target: list[Point]) -> Affine:
    if len(source) != len(target):
        raise Invalid("one target per source point")
    if len(source) < 3:
        raise Invalid("an affine fit needs three control points")
    rows = [[x, y, 1.0] for x, y in source]
    normal = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    xs = _solve(
        normal, [sum(r[i] * t[0] for r, t in zip(rows, target, strict=True)) for i in range(3)]
    )
    ys = _solve(
        normal, [sum(r[i] * t[1] for r, t in zip(rows, target, strict=True)) for i in range(3)]
    )
    return xs[0], xs[1], xs[2], ys[0], ys[1], ys[2]


def fit_similarity(source: list[Point], target: list[Point]) -> Affine:
    # scale, rotation and shift only: four parameters
    if len(source) != len(target):
        raise Invalid("one target per source point")
    if len(source) < 2:
        raise Invalid("a similarity fit needs two control points")
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        rows.append([x, -y, 1.0, 0.0])
        rhs.append(u)
        rows.append([y, x, 0.0, 1.0])
        rhs.append(v)
    normal = [[sum(r[i] * r[j] for r in rows) for j in range(4)] for i in range(4)]
    a, b, c, d = _solve(
        normal, [sum(r[i] * v for r, v in zip(rows, rhs, strict=True)) for i in range(4)]
    )
    return a, -b, c, b, a, d


def residuals(t: Affine, source: list[Point], target: list[Point]) -> list[float]:
    return [math.dist(apply(t, s), g) for s, g in zip(source, target, strict=True)]


def rms(values: list[float]) -> float:
    if not values:
        raise Invalid("no residuals")
    return math.sqrt(sum(v * v for v in values) / len(values))


def leave_one_out(source: list[Point], target: list[Point], fitter=fit_affine) -> list[float]:
    if len(source) < 4:
        raise Invalid("leave-one-out needs a spare point beyond the three that fix the fit")
    out = []
    for i in range(len(source)):
        rest_s = source[:i] + source[i + 1 :]
        rest_t = target[:i] + target[i + 1 :]
        t = fitter(rest_s, rest_t)
        out.append(math.dist(apply(t, source[i]), target[i]))
    return out


def true_transform(scale: float, angle_deg: float, shift: Point, shear: float = 0.0) -> Affine:
    a = math.radians(angle_deg)
    return (
        scale * math.cos(a),
        -scale * math.sin(a) + shear,
        shift[0],
        scale * math.sin(a),
        scale * math.cos(a),
        shift[1],
    )


def control_points(n: int, rng: random.Random, side: float = 1000.0) -> list[Point]:
    return [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]


def jitter(points: list[Point], sigma: float, rng: random.Random) -> list[Point]:
    return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma)) for x, y in points]


def expected_rms(sigma: float, n: int, parameters: int = 6) -> float:
    # each axis loses parameters over 2 degrees of freedom to the fit
    if 2 * n <= parameters:
        return 0.0
    return sigma * math.sqrt(2) * math.sqrt(1 - parameters / (2 * n))


def scale_of(t: Affine) -> float:
    return math.sqrt(abs(t[0] * t[4] - t[1] * t[3]))


def rotation_of(t: Affine) -> float:
    return math.degrees(math.atan2(t[3], t[0]))
