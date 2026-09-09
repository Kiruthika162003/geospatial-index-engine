from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.trackcompress import (
    position_at,
    spatial,
    straight_with_stop,
    time_ratio,
    worst_errors,
)

STEADY = [(10.0 * t, 0.0, float(t)) for t in range(60)]


class TestKnownTracks:
    def test_a_steady_straight_track_keeps_only_its_ends(self):
        assert spatial(STEADY, 1.0) == [0, 59]
        assert time_ratio(STEADY, 1.0) == [0, 59]

    def test_a_stop_is_invisible_to_the_spatial_rule_and_kept_by_the_time_rule(self):
        track = straight_with_stop(10.0, 30, 60, 30)
        assert spatial(track, 1.0) == [0, 119]
        assert time_ratio(track, 1.0) == [0, 30, 90, 119]
        midpoint = track[60]
        estimate = position_at(track, [0, 119], midpoint[2])
        assert math.dist(estimate, midpoint[:2]) == pytest.approx(2.5, abs=0.1)
        assert worst_errors(track, [0, 119]) == pytest.approx((0.0, 151.26), abs=0.01)
        assert worst_errors(track, [0, 30, 90, 119]) == pytest.approx((0.0, 0.0), abs=1e-9)

    def test_a_bending_track_keeps_its_corners_under_both_rules(self):
        bend, x, y, t = [], 0.0, 0.0, 0.0
        for heading in (0, 90, 180, 270):
            for _ in range(25):
                bend.append((x, y, t))
                x += 10 * math.cos(math.radians(heading))
                y += 10 * math.sin(math.radians(heading))
                t += 1
        assert spatial(bend, 1.0) == time_ratio(bend, 1.0) == [0, 25, 50, 75, 99]


class TestARealLookingTrack:
    def test_the_time_rule_keeps_more_at_tight_tolerance_and_never_mislocates_by_48(self):
        rng = random.Random(253)
        track, x, y, t, heading, speed = [], 0.0, 0.0, 0.0, 0.0, 10.0
        for i in range(600):
            if i % 100 == 0:
                heading += rng.uniform(-90, 90)
                speed = rng.uniform(4, 16)
            track.append((x + rng.gauss(0, 0.5), y + rng.gauss(0, 0.5), t))
            x += speed * math.cos(math.radians(heading))
            y += speed * math.sin(math.radians(heading))
            t += 1
        assert len(spatial(track, 1.0)) == 146
        assert len(time_ratio(track, 1.0)) == 248
        assert worst_errors(track, time_ratio(track, 1.0))[1] <= 1.0 + 1e-9
        assert worst_errors(track, spatial(track, 1.0))[1] == pytest.approx(1.96, abs=0.01)
        assert worst_errors(track, spatial(track, 10.0))[1] == pytest.approx(48.12, abs=0.05)
        assert worst_errors(track, time_ratio(track, 10.0))[1] == pytest.approx(2.02, abs=0.05)


class TestRefusals:
    def test_bad_tracks_tolerances_and_times_are_refused(self):
        with pytest.raises(Invalid):
            spatial([(0, 0, 0)], 1.0)
        with pytest.raises(Invalid):
            spatial(STEADY, -1)
        with pytest.raises(Invalid):
            position_at(STEADY, [0, 59], 100.0)
