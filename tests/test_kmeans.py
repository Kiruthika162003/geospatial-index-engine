from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.kmeans import cluster, inertia


def _blobs(rng):
    def blob(cx, cy, n):
        return [(cx + rng.gauss(0, 0.5), cy + rng.gauss(0, 0.5)) for _ in range(n)]

    return blob(0, 0, 100) + blob(10, 0, 100) + blob(5, 8, 100)


class TestMonotoneDescent:
    def test_inertia_never_rises_within_a_run(self):
        rng = random.Random(59)
        pts = _blobs(rng)
        for _ in range(200):
            init = random.Random(rng.random()).sample(pts, 3)
            _, _, history = cluster(pts, init)
            for i in range(1, len(history)):
                assert history[i] <= history[i - 1] + 1e-9

    def test_the_reported_inertia_matches_a_recomputation(self):
        rng = random.Random(3)
        pts = _blobs(rng)
        labels, centers, history = cluster(pts, rng.sample(pts, 3))
        assert history[-1] == pytest.approx(inertia(pts, labels, centers))


class TestSeedDependence:
    def test_different_seeds_reach_different_local_optima(self):
        rng = random.Random(59)
        pts = _blobs(rng)
        finals = {round(cluster(pts, random.Random(s).sample(pts, 3))[2][-1], 2)
                  for s in range(50)}
        # the local optima genuinely differ, and by a lot
        assert len(finals) > 3
        assert max(finals) > min(finals) * 5

    def test_the_true_centers_give_a_low_inertia(self):
        rng = random.Random(59)
        pts = _blobs(rng)
        _, _, history = cluster(pts, [(0, 0), (10, 0), (5, 8)])
        # seeding at the real blob centers reaches the good optimum
        assert history[-1] < 200


class TestRefusals:
    def test_none_points_is_refused(self):
        with pytest.raises(Invalid):
            cluster(None, [(0, 0)])

    def test_no_centers_is_refused(self):
        with pytest.raises(Invalid):
            cluster([(0, 0)], [])
