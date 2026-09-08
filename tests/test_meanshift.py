from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.meanshift import climb, cluster

CENTERS = [(0, 0), (10, 0), (5, 8)]


def _blobs(rng):
    def blob(cx, cy, n):
        return [(cx + rng.gauss(0, 0.4), cy + rng.gauss(0, 0.4)) for _ in range(n)]

    return blob(0, 0, 60) + blob(10, 0, 60) + blob(5, 8, 60)


class TestModeCountFollowsBandwidth:
    def test_tiny_bandwidth_fragments_into_many_modes(self):
        rng = random.Random(77)
        _, modes = cluster(_blobs(rng), 0.05)
        assert len(modes) > 100  # measured 159 of 180 points

    def test_a_moderate_bandwidth_finds_the_three_blobs(self):
        rng = random.Random(77)
        pts = _blobs(rng)
        assert len(cluster(pts, 1.5)[1]) == 3
        assert len(cluster(pts, 3.0)[1]) == 3

    def test_a_huge_bandwidth_merges_everything(self):
        rng = random.Random(77)
        _, modes = cluster(_blobs(rng), 50.0)
        assert len(modes) == 1

    def test_the_modes_land_on_the_blob_centers(self):
        rng = random.Random(77)
        _, modes = cluster(_blobs(rng), 1.5)
        worst = max(
            min(math.hypot(m[0] - c[0], m[1] - c[1]) for c in CENTERS) for m in modes
        )
        assert worst < 0.3  # measured 0.124


class TestClimb:
    def test_a_point_climbs_toward_its_blob_center(self):
        rng = random.Random(77)
        pts = _blobs(rng)
        mode = climb((0.8, 0.8), pts, 1.5)
        assert math.hypot(mode[0], mode[1]) < 0.3

    def test_every_point_gets_a_label_within_the_mode_count(self):
        rng = random.Random(77)
        pts = _blobs(rng)
        labels, modes = cluster(pts, 1.5)
        assert len(labels) == len(pts)
        assert set(labels) == set(range(len(modes)))


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            cluster(None, 1.0)

    def test_a_non_positive_bandwidth_is_refused(self):
        with pytest.raises(Invalid):
            cluster([(0, 0)], 0)
