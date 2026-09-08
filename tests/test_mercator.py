from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Outside
from atlas.mercator import MAX_LATITUDE, area_inflation, forward, inverse


class TestProjection:
    def test_the_center_maps_to_the_square_center(self):
        assert forward(0, 0) == pytest.approx((0.5, 0.5))

    def test_the_corners_map_to_the_unit_square(self):
        assert forward(0, -180) == pytest.approx((0.0, 0.5))
        assert forward(MAX_LATITUDE, 180) == pytest.approx((1.0, 0.0), abs=1e-9)

    def test_it_round_trips(self):
        rng = random.Random(35)
        worst = 0.0
        for _ in range(100000):
            lat = rng.uniform(-MAX_LATITUDE, MAX_LATITUDE)
            lon = rng.uniform(-180, 180)
            la, lo = inverse(*forward(lat, lon))
            worst = max(worst, abs(la - lat), abs(lo - lon))
        assert worst < 1e-9


class TestAreaInflation:
    def test_the_inflation_follows_the_secant_squared_law(self):
        assert area_inflation(0) == pytest.approx(1.0)
        assert area_inflation(45) == pytest.approx(2.0)
        assert area_inflation(60) == pytest.approx(4.0)
        assert area_inflation(75) == pytest.approx(14.928, abs=0.01)

    def test_the_vertical_stretch_is_the_secant_of_latitude(self):
        _, y_base_a = forward(0, 0)
        _, y_base_b = forward(1e-4, 0)
        base = abs(y_base_b - y_base_a)
        for lat in (30, 45, 60):
            _, y1 = forward(lat, 0)
            _, y2 = forward(lat + 1e-4, 0)
            stretch = abs(y2 - y1) / base
            assert stretch == pytest.approx(1 / math.cos(math.radians(lat)), rel=1e-4)


class TestRefusals:
    def test_a_latitude_past_the_limit_is_refused(self):
        with pytest.raises(Outside):
            forward(86, 0)

    def test_a_point_outside_the_unit_square_is_refused(self):
        with pytest.raises(Outside):
            inverse(1.5, 0.5)
