"""DBSCAN: cluster by density, discovering how many clusters there are and what is noise.

DBSCAN groups points by density rather than by distance to a center,
which lets it do two things partitioning methods like k-means cannot:
find clusters of any shape, and refuse to force every point into a
cluster. It has two parameters, a radius epsilon and a minimum count.
A point is a core point if at least the minimum number of points lie
within epsilon of it, including itself; core points that are within
epsilon of each other, directly or through a chain of core points,
belong to the same cluster, and non-core points within epsilon of a
core point join it as border points. Everything else, points in no
dense neighborhood, is labeled noise and left in no cluster at all.
The consequences are what make it worth the trouble. The number of
clusters is discovered from the data, not supplied in advance, so the
algorithm answers how many groups there are rather than being told;
and because clusters grow by chaining through dense regions, a long
curved or ring-shaped cluster is found whole, where k-means would slice
it because it assumes round blobs around centers. The two parameters
trade in an understandable way: too small an epsilon and every point is
too lonely to be core, so all of it becomes noise; too large and every
point reaches every other, so the whole set collapses into one cluster.
The finding worth stating is that DBSCAN's cluster count is an output
governed by epsilon, running from all-noise at tiny epsilon to
one-cluster at large epsilon and passing through the true count in
between, so the same data yields different, predictable structure as
the density threshold moves. This module runs DBSCAN, and a survey
measures how the discovered cluster count responds to epsilon on
well-separated blobs.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]
NOISE = -1


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def cluster(points: list[Point], eps: float, min_points: int) -> list[int]:
    if points is None:
        raise Invalid("points must not be None")
    if eps <= 0:
        raise Invalid("eps must be positive")
    if min_points < 1:
        raise Invalid("min_points must be at least 1")
    eps2 = eps * eps
    n = len(points)
    labels = [0] * n  # 0 means unvisited; real labels start at 1, NOISE is -1
    cluster_id = 0

    def neighbors(i: int) -> list[int]:
        return [j for j in range(n) if _dist2(points[i], points[j]) <= eps2]

    for i in range(n):
        if labels[i] != 0:
            continue
        neigh = neighbors(i)
        if len(neigh) < min_points:
            labels[i] = NOISE
            continue
        cluster_id += 1
        labels[i] = cluster_id
        seeds = [j for j in neigh if j != i]
        k = 0
        while k < len(seeds):
            j = seeds[k]
            k += 1
            if labels[j] == NOISE:
                labels[j] = cluster_id  # a border point of this cluster
            if labels[j] != 0:
                continue
            labels[j] = cluster_id
            j_neigh = neighbors(j)
            if len(j_neigh) >= min_points:
                seeds.extend(j_neigh)
    return labels


def count_clusters(labels: list[int]) -> int:
    return len({label for label in labels if label != NOISE})
