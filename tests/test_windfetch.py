from __future__ import annotations

import math

import pytest

from atlas.errors import Invalid
from atlas.windfetch import (
    bay,
    channel,
    channel_law,
    circle_law,
    effective_fetch,
    fetch,
    fetch_rose,
    lake,
    longest_fetch,
    shore_point,
)

EAST = math.pi / 2


@pytest.fixture(scope="module")
def pond():
    water = lake(201, 90.0)
    return water, shore_point(water, 100, 100, 3 * math.pi / 2)


class TestTheLake:
    def test_the_effective_fetch_is_0_884_of_the_chord(self, pond):
        water, (r, c) = pond
        assert (r, c) == (100, 10)
        assert fetch(water, r, c, EAST) == 180.0
        assert effective_fetch(water, r, c, EAST) == pytest.approx(159.114, abs=1e-3)
        assert circle_law(90.0) == pytest.approx(159.753, abs=1e-3)
        assert longest_fetch(water, r, c) == (180.0, 90.0)
        assert longest_fetch(water, 100, 100)[0] == 90.0
        assert min(fetch_rose(water, 100, 100, 36)) == 89.0

    def test_the_rose_reads_the_chords(self, pond):
        water, (r, c) = pond
        rose = fetch_rose(water, r, c, 36)
        assert rose[:4] == [0.0] * 4
        assert rose[15:] == [0.0] * 21
        assert [round(v) for v in rose[5:10]] == [137, 155, 168, 177, 180]
        for k in (5, 6, 7, 8):
            theta = math.radians(k * 10 - 90)
            assert rose[k] == pytest.approx(180 * math.cos(theta), abs=2.0)

    def test_spread_and_step(self, pond):
        water, (r, c) = pond
        settings = ((45, 15), (90, 6), (30, 6))
        readings = [effective_fetch(water, r, c, EAST, 1.0, s, t) for s, t in settings]
        assert readings == pytest.approx([158.319, 130.293, 170.068], abs=1e-3)


class TestChannelsAndBays:
    @pytest.mark.parametrize(
        ("width", "effective", "law"),
        [
            (5, 34.952, 35.242),
            (11, 44.684, 45.207),
            (21, 61.288, 61.815),
            (41, 94.632, 95.032),
        ],
    )
    def test_a_channels_fetch_is_a_fraction_of_its_length(self, width, effective, law):
        water = channel(61, 401, width)
        assert fetch(water, 30, 0, EAST) == 400.0
        assert effective_fetch(water, 30, 0, EAST) == pytest.approx(effective, abs=1e-3)
        assert channel_law(400, width) == pytest.approx(law, abs=1e-3)
        assert abs(effective / law - 1) < 0.012

    def test_a_coarse_step_reads_the_channel_high(self):
        water = channel(61, 401, 11)
        coarse = effective_fetch(water, 30, 0, EAST, 1.0, 45, 15)
        assert coarse == pytest.approx(76.963, abs=1e-3)

    def test_the_bay_mouth(self):
        water = bay(101, 21, 60)
        assert fetch(water, 50, 0, EAST) == 100.0
        assert effective_fetch(water, 50, 0, EAST) == pytest.approx(73.539, abs=1e-3)
        assert fetch(water, 30, 0, EAST) == 59.0
        assert effective_fetch(water, 30, 0, EAST) == pytest.approx(70.358, abs=1e-3)


class TestRefusals:
    def test_bad_masks_points_and_spreads(self):
        with pytest.raises(Invalid):
            fetch([], 0, 0, 0.0)
        with pytest.raises(Invalid):
            fetch(lake(9, 3.0), 20, 0, 0.0)
        with pytest.raises(Invalid):
            fetch(lake(9, 3.0), 4, 4, 0.0, 0.0)
        with pytest.raises(Invalid):
            fetch_rose(lake(9, 3.0), 4, 4, 0)
        with pytest.raises(Invalid):
            effective_fetch(lake(9, 3.0), 4, 4, 0.0, 1.0, 45, 60)
        with pytest.raises(Invalid):
            channel(5, 5, 6)
        with pytest.raises(Invalid):
            bay(9, 0, 4)
        with pytest.raises(Invalid):
            shore_point(lake(9, 3.0), 0, 0, 0.0)
        with pytest.raises(Invalid):
            lake(2, 1.0)
