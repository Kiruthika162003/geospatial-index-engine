"""A warp that costs lockstep 3578 costs DTW 46, and a window under the shift gives it back.

Two tracks over the same path at different speeds are the same route
to a person and far apart to a point-by-point sum. This module reads
dynamic time warping beside the lockstep sum on a sine track of 200
points resampled by arc length, path length 109.24. Reparametrizing
the track by u to the gamma leaves DTW at 2.72 for gamma 1, where the
only change is the resampling itself, and reads 37.48, 45.97 and 60.36
for gamma 1.5, 2 and 3, while the lockstep sum reads 2163, 3578 and
5287. The warping path is 229, 250 and 277 steps long and its largest
index shift is 29, 50 and 77. A stall of 50 repeated points costs DTW
exactly 0.0 and shifts the path by 50.

The guess that a Sakoe-Chiba window is a mild speedup was wrong: it is
a hard bound on the shift. On the gamma 2 warp, whose shift is 50, a
window of 0 reads the lockstep sum 3577.71, windows of 5, 10, 20 and
40 read 3061.86, 2572.58, 1680.55 and 344.17, and a window of 50 reads
the unwindowed 45.97 exactly. The guess that DTW smooths noise was
also wrong. Gaussian jitter of 0.1 on every point reads DTW 24.793
and lockstep 24.793, the same to three places, against the law
200 sigma root(pi/2) of 25.07; at sigma 0.5 DTW reads 118.81 to the
lockstep 124.14, and at 1.0 239.10 to 249.01, a saving of 4 percent
from the diagonal steps a jitter can buy.

DTW is not a metric. On 12 random walks of 30 steps, 12 of the 1320
ordered triples break the triangle inequality; a window of 3 leaves 8
broken and a window of 0, which is the lockstep sum, leaves none. A
sine track sampled evenly in x sits 25.87 from the same track sampled
evenly by arc length under DTW and 91.17 in lockstep, so the sampling
rule alone is a warp.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Track = list[Point]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dtw(a: Track, b: Track, window: int | None = None) -> float:
    if not a or not b:
        raise Invalid("both tracks need at least one point")
    if window is not None and window < 0:
        raise Invalid("the window must not be negative")
    n, m = len(a), len(b)
    w = max(window, abs(n - m)) if window is not None else max(n, m)
    previous = [math.inf] * (m + 1)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = [math.inf] * (m + 1)
        low, high = max(1, i - w), min(m, i + w)
        for j in range(low, high + 1):
            best = min(previous[j], current[j - 1], previous[j - 1])
            current[j] = _dist(a[i - 1], b[j - 1]) + best
        previous = current
    return previous[m]


def dtw_path(a: Track, b: Track) -> list[tuple[int, int]]:
    if not a or not b:
        raise Invalid("both tracks need at least one point")
    n, m = len(a), len(b)
    cost = [[math.inf] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
            cost[i][j] = _dist(a[i - 1], b[j - 1]) + best
    i, j = n, m
    path = [(n - 1, m - 1)]
    while i > 1 or j > 1:
        options = []
        if i > 1 and j > 1:
            options.append((cost[i - 1][j - 1], i - 1, j - 1))
        if i > 1:
            options.append((cost[i - 1][j], i - 1, j))
        if j > 1:
            options.append((cost[i][j - 1], i, j - 1))
        _, i, j = min(options)
        path.append((i - 1, j - 1))
    path.reverse()
    return path


def lockstep(a: Track, b: Track) -> float:
    if len(a) != len(b) or not a:
        raise Invalid("lockstep needs two tracks of the same positive length")
    return sum(_dist(p, q) for p, q in zip(a, b, strict=True))


def path_length(track: Track) -> float:
    return sum(_dist(track[i], track[i + 1]) for i in range(len(track) - 1))


def resample(track: Track, n: int) -> Track:
    if len(track) < 2:
        raise Invalid("resampling needs at least two points")
    if n < 2:
        raise Invalid("resample to at least two points")
    total = path_length(track)
    if total == 0:
        raise Invalid("the track has no length")
    stations = [total * k / (n - 1) for k in range(n)]
    out: Track = []
    seg = 0
    walked = 0.0
    for s in stations:
        while seg < len(track) - 2 and walked + _dist(track[seg], track[seg + 1]) < s:
            walked += _dist(track[seg], track[seg + 1])
            seg += 1
        length = _dist(track[seg], track[seg + 1])
        t = 0.0 if length == 0 else min(max((s - walked) / length, 0.0), 1.0)
        out.append(
            (
                track[seg][0] + t * (track[seg + 1][0] - track[seg][0]),
                track[seg][1] + t * (track[seg + 1][1] - track[seg][1]),
            )
        )
    return out


def warped(track: Track, n: int, gamma: float) -> Track:
    if gamma <= 0:
        raise Invalid("gamma must be positive")
    fine = resample(track, 2001)
    out = []
    for k in range(n):
        u = (k / (n - 1)) ** gamma
        out.append(fine[min(round(u * 2000), 2000)])
    return out


def stalled(track: Track, at: int, repeats: int) -> Track:
    if not (0 <= at < len(track)):
        raise Invalid("the stall must sit on the track")
    return track[: at + 1] + [track[at]] * repeats + track[at + 1 :]


def sine_track(n: int, length: float = 100.0, amplitude: float = 10.0) -> Track:
    if n < 2:
        raise Invalid("a track needs at least two points")
    out = []
    for k in range(n):
        t = k / (n - 1)
        out.append((length * t, amplitude * math.sin(2 * math.pi * t)))
    return out


def noisy(track: Track, sigma: float, rng: random.Random) -> Track:
    return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma)) for x, y in track]


def random_track(n: int, rng: random.Random, step: float = 1.0) -> Track:
    x = y = 0.0
    heading = rng.uniform(0, 2 * math.pi)
    out = [(x, y)]
    for _ in range(n - 1):
        heading += rng.gauss(0, 0.3)
        x += step * math.cos(heading)
        y += step * math.sin(heading)
        out.append((x, y))
    return out


def triangle_violations(tracks: list[Track], window: int | None = None) -> tuple[int, int]:
    if len(tracks) < 3:
        raise Invalid("at least three tracks are needed")
    seen = 0
    broken = 0
    d = {}
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            d[i, j] = d[j, i] = dtw(tracks[i], tracks[j], window)
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            for k in range(len(tracks)):
                if len({i, j, k}) < 3:
                    continue
                seen += 1
                if d[i, k] > d[i, j] + d[j, k] + 1e-9:
                    broken += 1
    return broken, seen


def largest_shift(path: list[tuple[int, int]]) -> int:
    return max(abs(i - j) for i, j in path)
