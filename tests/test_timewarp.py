from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.timewarp import (
    dtw,
    dtw_path,
    largest_shift,
    lockstep,
    noisy,
    path_length,
    random_track,
    resample,
    sine_track,
    stalled,
    triangle_violations,
    warped,
)


@pytest.fixture(scope="module")
def base():
    return resample(sine_track(400), 200)


class TestWarps:
    def test_the_arc_length_base(self, base):
        assert len(base) == 200
        assert path_length(base) == pytest.approx(109.237, abs=1e-3)
        assert dtw(sine_track(200), base) == pytest.approx(25.87, abs=0.01)
        assert lockstep(sine_track(200), base) == pytest.approx(91.17, abs=0.01)

    @pytest.mark.parametrize(
        ("gamma", "warp", "step", "shift", "steps"),
        [
            (1.0, 2.7172, 2.717, 0, 200),
            (1.5, 37.4766, 2163.454, 29, 229),
            (2.0, 45.9686, 3577.715, 50, 250),
            (3.0, 60.3648, 5287.437, 77, 277),
        ],
    )
    def test_dtw_absorbs_the_warp_that_lockstep_pays_for(
        self, base, gamma, warp, step, shift, steps
    ):
        other = warped(base, 200, gamma)
        assert dtw(base, other) == pytest.approx(warp, abs=1e-3)
        assert lockstep(base, other) == pytest.approx(step, abs=1e-2)
        path = dtw_path(base, other)
        assert len(path) == steps
        assert largest_shift(path) == shift
        assert path[0] == (0, 0) and path[-1] == (199, 199)

    def test_a_stall_costs_nothing(self, base):
        other = stalled(base, 100, 50)
        assert len(other) == 250
        assert dtw(base, other) == 0.0
        assert largest_shift(dtw_path(base, other)) == 50


class TestTheWindow:
    def test_a_window_below_the_shift_binds_and_at_the_shift_releases(self, base):
        other = warped(base, 200, 2.0)
        assert dtw(base, other, 0) == pytest.approx(lockstep(base, other), abs=1e-6)
        readings = [dtw(base, other, w) for w in (5, 10, 20, 40)]
        assert readings == pytest.approx([3061.86, 2572.58, 1680.55, 344.17], abs=0.01)
        assert dtw(base, other, 50) == pytest.approx(dtw(base, other), abs=1e-9)
        assert dtw(base, other, 60) == pytest.approx(dtw(base, other), abs=1e-9)


class TestNoiseAndMetric:
    def test_jitter_is_not_absorbed(self, base):
        rng = random.Random(300)
        for sigma, warp, step in ((0.1, 24.793, 24.793), (0.5, 118.807, 124.144)):
            warps, steps = [], []
            for _ in range(10):
                other = noisy(base, sigma, rng)
                warps.append(dtw(base, other))
                steps.append(lockstep(base, other))
            assert sum(warps) / 10 == pytest.approx(warp, abs=1e-3)
            assert sum(steps) / 10 == pytest.approx(step, abs=1e-3)
            law = 200 * sigma * math.sqrt(math.pi / 2)
            assert sum(steps) / 10 == pytest.approx(law, rel=0.02)

    def test_twelve_triples_break_the_triangle_and_lockstep_breaks_none(self):
        rng = random.Random(301)
        tracks = [random_track(30, rng) for _ in range(12)]
        assert triangle_violations(tracks) == (12, 1320)
        assert triangle_violations(tracks, 3) == (8, 1320)
        assert triangle_violations(tracks, 0) == (0, 1320)


class TestRefusals:
    def test_bad_tracks_windows_and_warps(self):
        with pytest.raises(Invalid):
            dtw([], [(0, 0)])
        with pytest.raises(Invalid):
            dtw([(0, 0)], [(0, 0)], -1)
        with pytest.raises(Invalid):
            lockstep([(0, 0)], [(0, 0), (1, 1)])
        with pytest.raises(Invalid):
            resample([(0, 0)], 5)
        with pytest.raises(Invalid):
            resample([(0, 0), (0, 0)], 5)
        with pytest.raises(Invalid):
            warped(sine_track(10), 10, 0)
        with pytest.raises(Invalid):
            stalled(sine_track(10), 10, 1)
        with pytest.raises(Invalid):
            triangle_violations([sine_track(5), sine_track(5)])
