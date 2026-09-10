from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.mortonjoin import (
    brute_join,
    expected_pairs,
    keys,
    recall,
    seam_share,
    sweep_join,
    uniform,
    window_for_recall,
)


@pytest.fixture(scope="module")
def cloud():
    return uniform(1000, random.Random(811))


class TestRecall:
    @pytest.mark.parametrize(
        ("distance", "pairs", "law", "seam", "recalls", "window"),
        [
            (10.0, 153, 156.9, 0.0131, (0.7386, 0.9477, 0.9869), 4),
            (30.0, 1422, 1412.3, 0.0373, (0.3664, 0.8066, 0.9501), 32),
        ],
    )
    def test_the_window_sets_the_recall_and_the_seam_is_not_the_story(
        self, cloud, distance, pairs, law, seam, recalls, window
    ):
        truth = brute_join(cloud, distance)
        assert len(truth) == pairs
        assert expected_pairs(1000, distance) == pytest.approx(law, abs=0.1)
        assert seam_share(cloud, 1000.0, distance) == pytest.approx(seam, abs=1e-4)
        for w, r in zip((1, 8, 64), recalls, strict=True):
            read, count, compares = recall(cloud, 1000.0, distance, w)
            assert read == pytest.approx(r, abs=1e-4)
            assert count == pairs
            assert compares == w * 1000
        assert window_for_recall(cloud, 1000.0, distance, 0.9) == window
        assert 1 - recalls[1] > seam

    def test_the_sweep_never_invents_a_pair(self, cloud):
        truth = brute_join(cloud, 10.0)
        assert sweep_join(cloud, 1000.0, 10.0, 16) <= truth
        assert len(keys(cloud, 1000.0)) == 1000


class TestRefusals:
    def test_bad_sides_windows_and_targets(self, cloud):
        with pytest.raises(Invalid):
            keys(cloud, 0.0)
        with pytest.raises(Invalid):
            sweep_join(cloud, 1000.0, 10.0, 0)
        with pytest.raises(Invalid):
            recall([(0.0, 0.0), (500.0, 500.0)], 1000.0, 1.0, 4)
        with pytest.raises(Invalid):
            window_for_recall(cloud, 1000.0, 10.0, 0.0)
