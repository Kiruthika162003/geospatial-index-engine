from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.hand import (
    floodplain_fraction,
    hand,
    mean,
    noisy,
    stream_fraction,
    streams,
    v_valley,
    valley_law,
)
from atlas.sinkfill import priority_flood

CLEAN = v_valley(61, 0.5)


class TestCleanValley:
    def test_the_axis_is_the_stream_and_the_height_reads_the_wall(self):
        heights, channel = hand(CLEAN, 30)
        assert stream_fraction(channel) == pytest.approx(0.0492, abs=1e-4)
        assert mean(heights) == pytest.approx(7.1311, abs=1e-3)
        assert all(v >= 0 for row in heights for v in row)
        for level, fraction, law in (
            (0.5, 0.0771, 0.0492),
            (2.0, 0.1803, 0.1475),
            (5.0, 0.377, 0.3443),
        ):
            assert floodplain_fraction(heights, level) == pytest.approx(fraction, abs=1e-4)
            assert valley_law(61, 0.5, level) == pytest.approx(law, abs=1e-4)
            assert fraction > law

    @pytest.mark.parametrize(
        ("threshold", "stream", "height", "flood"),
        [
            (5, 0.8689, 0.1639, 1.0),
            (200, 0.0156, 7.6239, 0.1459),
            (1000, 0.0121, 7.6452, 0.1389),
        ],
    )
    def test_the_threshold_is_the_whole_story(self, threshold, stream, height, flood):
        heights, channel = hand(CLEAN, threshold)
        assert stream_fraction(channel) == pytest.approx(stream, abs=1e-4)
        assert mean(heights) == pytest.approx(height, abs=1e-3)
        assert floodplain_fraction(heights, 2.0) == pytest.approx(flood, abs=1e-4)


class TestNoise:
    @pytest.mark.parametrize(
        ("sigma", "stream", "height", "flood", "filled_flood"),
        [
            (0.1, 0.1819, 2.451, 0.4974, 0.5042),
            (0.5, 0.0202, 2.025, 0.5396, 0.627),
            (1.0, 0.0019, 2.1399, 0.4652, 0.5485),
        ],
    )
    def test_rills_not_sinks(self, sigma, stream, height, flood, filled_flood):
        dem = noisy(CLEAN, sigma, random.Random(700))
        heights, channel = hand(dem, 30)
        assert stream_fraction(channel) == pytest.approx(stream, abs=1e-4)
        assert mean(heights) == pytest.approx(height, abs=1e-3)
        assert floodplain_fraction(heights, 2.0) == pytest.approx(flood, abs=1e-4)
        assert all(v >= 0 for row in heights for v in row)
        filled, _ = hand(priority_flood(dem, 1e-6), 30)
        assert floodplain_fraction(filled, 2.0) == pytest.approx(filled_flood, abs=1e-4)


class TestRefusals:
    def test_bad_grids_and_thresholds(self):
        with pytest.raises(Invalid):
            streams([], 1)
        with pytest.raises(Invalid):
            streams(CLEAN, 0)
