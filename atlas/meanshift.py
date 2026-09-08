"""Mean shift: climb each point to a density peak, and the peaks are the clusters.

Mean shift clusters without being told how many clusters to find. It
treats the points as samples from an unknown density and seeks the
density's peaks, the modes, by a simple hill climb. Place a window of
fixed bandwidth on a point, compute the mean of all points inside the
window, move the window to that mean, and repeat; each step shifts the
window toward denser ground, since the mean of a lopsided neighborhood
lies on its heavier side, and the shifts shrink to nothing when the
window is centered on a local maximum of density, a mode. Run the
climb from every point, and points that climb to the same mode belong
to the same cluster; the number of distinct modes is the number of
clusters, discovered rather than supplied, the property it shares with
DBSCAN and lacks in k-means. The bandwidth is the one parameter and it
plays the role epsilon plays for DBSCAN: too small and every little
bump in the sample is its own mode, fragmenting the data; too large and
neighboring peaks merge under one wide window until a single mode
swallows everything. Between those, the climbs from a few
well-separated blobs land on a few well-separated modes, one per blob,
and the modes sit at the blob centers where the density is highest.
Because each point climbs independently, the method is embarrassingly
parallel but not cheap, every step being a pass over the points, so it
suits modest sets. The finding worth stating is that mean shift's mode
count is governed by bandwidth, running from one mode per point at
tiny bandwidth to one mode overall at huge bandwidth and passing
through the true blob count between, with each mode landing at a blob
center. This module runs mean shift with a flat kernel, and a survey
measures the mode count against bandwidth on separated blobs.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Point = tuple[float, float]


def _shift_once(center: Point, points: list[Point], bandwidth: float) -> Point:
    sx = sy = 0.0
    n = 0
    for x, y in points:
        if math.hypot(x - center[0], y - center[1]) <= bandwidth:
            sx += x
            sy += y
            n += 1
    if n == 0:
        return center
    return (sx / n, sy / n)


def climb(start: Point, points: list[Point], bandwidth: float, max_iters: int = 200) -> Point:
    center = start
    for _ in range(max_iters):
        moved = _shift_once(center, points, bandwidth)
        if math.hypot(moved[0] - center[0], moved[1] - center[1]) < 1e-6:
            return moved
        center = moved
    return center


def cluster(points: list[Point], bandwidth: float) -> tuple[list[int], list[Point]]:
    if points is None:
        raise Invalid("points must not be None")
    if bandwidth <= 0:
        raise Invalid("bandwidth must be positive")
    modes: list[Point] = []
    labels: list[int] = []
    merge_radius = bandwidth / 2
    for p in points:
        mode = climb(p, points, bandwidth)
        found = -1
        for i, m in enumerate(modes):
            if math.hypot(m[0] - mode[0], m[1] - mode[1]) <= merge_radius:
                found = i
                break
        if found < 0:
            modes.append(mode)
            found = len(modes) - 1
        labels.append(found)
    return labels, modes
