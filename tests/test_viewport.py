from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.viewport import (
    fit_zoom,
    fits,
    from_pixels,
    ground_distance_of_pixels,
    meters_per_pixel,
    scale_bar_meters,
    tile_count,
    to_pixels,
)


class TestMetersPerPixel:
    def test_the_formula_and_the_ground_reading_0_112_percent_short(self):
        assert meters_per_pixel(0, 0) == pytest.approx(156543.03, abs=0.01)
        expected = 156543.03 * math.cos(math.radians(45)) / 1024
        assert meters_per_pixel(45, 10) == pytest.approx(expected, rel=1e-6)
        for zoom in (8, 12, 18):
            for lat in (0, 30, 60, 70):
                ground = ground_distance_of_pixels(lat, 10.0, zoom, 100)
                formula = meters_per_pixel(lat, zoom) * 100
                assert 100 * (ground / formula - 1) == pytest.approx(-0.1118, abs=1e-3)
        assert pytest.approx(0.99888, abs=1e-5) == 6371.0088 / 6378.137

    def test_the_gap_widens_at_low_zoom_where_the_scale_changes_across_the_span(self):
        for lat, gap in ((70, -5.646), (30, -1.865)):
            ground = ground_distance_of_pixels(lat, 10.0, 1, 100)
            formula = meters_per_pixel(lat, 1) * 100
            assert 100 * (ground / formula - 1) == pytest.approx(gap, abs=0.01)


class TestFitZoom:
    def test_the_box_fits_at_the_fit_zoom_and_overflows_at_one_more(self):
        rng = random.Random(243)
        for _ in range(200):
            south, west = rng.uniform(-70, 60), rng.uniform(-170, 160)
            north, east = south + rng.uniform(0.01, 20), west + rng.uniform(0.01, 20)
            width, height = rng.randint(200, 1600), rng.randint(200, 1200)
            zoom = fit_zoom(south, west, north, east, width, height)
            assert fits(south, west, north, east, width, height, zoom)
            assert not fits(south, west, north, east, width, height, zoom + 1)

    def test_known_fits_and_the_round_trip(self):
        assert fit_zoom(-80, -180, 80, 180, 1024, 768) == 1
        assert fit_zoom(51.45, -0.2, 51.55, -0.05, 800, 600) == 12
        lat, lon = from_pixels(*to_pixels(51.5, -0.1, 12), 12)
        assert (lat, lon) == pytest.approx((51.5, -0.1), abs=1e-9)


class TestTiles:
    def test_a_window_needs_its_span_plus_one_in_each_axis(self):
        rng = random.Random(243)
        assert tile_count(51.5, -0.1, 12, 1024, 768) == 20
        counts = set()
        for _ in range(500):
            lat, lon, zoom = rng.uniform(-60, 60), rng.uniform(-180, 180), rng.randint(3, 15)
            counts.add(tile_count(lat, lon, zoom, 1024, 768))
        assert counts == {20}

    def test_the_scale_bar(self):
        assert scale_bar_meters(51.5, 14, 100) == pytest.approx(594.8, abs=0.1)


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            meters_per_pixel(0, -1)
        with pytest.raises(Invalid):
            fit_zoom(1, 1, 0, 0, 100, 100)
        with pytest.raises(Invalid):
            fit_zoom(0, 0, 1, 1, 0, 100)
        with pytest.raises(Outside):
            to_pixels(89, 0, 1)
        with pytest.raises(Invalid):
            tile_count(0, 0, 1.5, 100, 100)
