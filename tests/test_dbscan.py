from __future__ import annotations

import math
import random

import pytest

from atlas.dbscan import NOISE, cluster, count_clusters
from atlas.errors import Invalid


def _blobs(rng):
    def blob(cx, cy, n):
        return [(cx + rng.gauss(0, 0.3), cy + rng.gauss(0, 0.3)) for _ in range(n)]

    return blob(0, 0, 50) + blob(10, 0, 50) + blob(5, 10, 50)


class TestDiscovery:
    def test_it_finds_the_true_number_of_blobs(self):
        rng = random.Random(41)
        labels = cluster(_blobs(rng), eps=1.0, min_points=4)
        assert count_clusters(labels) == 3
        assert labels.count(NOISE) == 0

    def test_eps_runs_from_all_noise_to_one_cluster(self):
        rng = random.Random(41)
        pts = _blobs(rng)
        assert count_clusters(cluster(pts, eps=0.05, min_points=4)) == 0  # all noise
        assert count_clusters(cluster(pts, eps=0.5, min_points=4)) == 3  # true count
        assert count_clusters(cluster(pts, eps=20.0, min_points=4)) == 1  # all merged

    def test_a_tiny_eps_marks_everything_noise(self):
        rng = random.Random(41)
        labels = cluster(_blobs(rng), eps=0.05, min_points=4)
        assert all(label == NOISE for label in labels)


class TestArbitraryShapes:
    def test_a_ring_is_one_cluster_not_split(self):
        ring = [(5 * math.cos(i * 0.1), 5 * math.sin(i * 0.1)) for i in range(63)]
        labels = cluster(ring, eps=1.0, min_points=2)
        assert count_clusters(labels) == 1


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            cluster(None, 1.0, 4)

    def test_a_non_positive_eps_is_refused(self):
        with pytest.raises(Invalid):
            cluster([(0, 0)], 0, 4)

    def test_a_bad_min_points_is_refused(self):
        with pytest.raises(Invalid):
            cluster([(0, 0)], 1.0, 0)
