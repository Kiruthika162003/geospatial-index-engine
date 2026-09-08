"""k-means: Lloyd's alternation drives inertia down to a local optimum set by the seed.

k-means partitions points into k groups, each represented by a
centroid, minimizing inertia, the total squared distance from every
point to its group's centroid. Lloyd's algorithm reaches a minimum by
alternating two steps: assign each point to its nearest centroid, then
move each centroid to the mean of the points assigned to it. Both steps
can only lower inertia, so the algorithm converges, and it converges
quickly. Two properties define its behavior and both are worth stating
as measurements rather than warnings. First, inertia decreases
monotonically across iterations and never rises, which is why the loop
can stop when it stops improving; that monotone descent is the
algorithm's correctness guarantee. Second, the minimum it reaches is
only a local one, and which local minimum depends entirely on the
initial centroids, so two runs from different random seeds on the same
data can converge to different partitions with different final inertia,
sometimes markedly different when the data has several plausible
groupings. This is the price of the speed, and it is why practical
k-means runs several times and keeps the best, or seeds carefully with
k-means++ to spread the initial centroids apart. There is also a
shape assumption baked in: because each point goes to the nearest
centroid, the groups are separated by straight bisectors, so k-means
carves space into convex cells and cannot recover a curved or nested
cluster the way a density method can. The finding worth stating is that
Lloyd's inertia falls monotonically to a local optimum whose value
depends on the seed, so the algorithm is fast and correct in its
descent but not deterministic in its answer. This module runs Lloyd's
k-means returning the assignment and inertia, and a survey confirms
inertia never rises within a run and varies across seeds.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _assign(points: list[Point], centers: list[Point]) -> list[int]:
    labels = []
    for p in points:
        best = min(range(len(centers)), key=lambda c: _dist2(p, centers[c]))
        labels.append(best)
    return labels


def _recenter(points: list[Point], labels: list[int], k: int, old: list[Point]) -> list[Point]:
    sums = [[0.0, 0.0, 0] for _ in range(k)]
    for p, label in zip(points, labels, strict=True):
        sums[label][0] += p[0]
        sums[label][1] += p[1]
        sums[label][2] += 1
    centers = []
    for i in range(k):
        if sums[i][2] == 0:
            centers.append(old[i])  # keep an emptied center in place
        else:
            centers.append((sums[i][0] / sums[i][2], sums[i][1] / sums[i][2]))
    return centers


def inertia(points: list[Point], labels: list[int], centers: list[Point]) -> float:
    return sum(_dist2(p, centers[label]) for p, label in zip(points, labels, strict=True))


def cluster(
    points: list[Point], initial_centers: list[Point], max_iters: int = 100
) -> tuple[list[int], list[Point], list[float]]:
    if points is None or initial_centers is None:
        raise Invalid("points and initial_centers must not be None")
    if not points:
        raise Invalid("points must not be empty")
    if not initial_centers:
        raise Invalid("at least one initial center is required")
    k = len(initial_centers)
    centers = list(initial_centers)
    history: list[float] = []
    labels = _assign(points, centers)
    history.append(inertia(points, labels, centers))
    for _ in range(max_iters):
        centers = _recenter(points, labels, k, centers)
        new_labels = _assign(points, centers)
        history.append(inertia(points, new_labels, centers))
        if new_labels == labels:
            labels = new_labels
            break
        labels = new_labels
    return labels, centers, history
