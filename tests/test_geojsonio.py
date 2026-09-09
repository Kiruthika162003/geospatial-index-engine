from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.geojsonio import (
    dumps,
    feature,
    is_counterclockwise,
    linestring_object,
    loads,
    point_object,
    polygon_object,
)
from atlas.shoelace import signed_area


def _area(ring) -> float:
    return abs(signed_area([(lon, lat) for lat, lon in ring]))


class TestTheRightHandRule:
    def test_half_the_rings_are_reversed_and_nothing_else_changes(self):
        rng = random.Random(244)
        reversed_outer = reversed_hole = 0
        for _ in range(1000):
            k = rng.randint(3, 9)
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
            ring = [(50 + 2 * math.sin(a), 10 + 2 * math.cos(a)) for a in angles]
            if rng.random() < 0.5:
                ring.reverse()
            hole = [(50 + 0.2 * math.sin(a), 10 + 0.2 * math.cos(a)) for a in angles]
            if rng.random() < 0.5:
                hole.reverse()
            _, rings = loads(dumps(polygon_object([ring, hole])))
            reversed_outer += not is_counterclockwise(ring)
            reversed_hole += is_counterclockwise(hole)
            assert is_counterclockwise(rings[0])
            assert not is_counterclockwise(rings[1])
            assert _area(rings[0]) == _area(ring)
            assert set(rings[0]) == set(ring)
            assert set(rings[1]) == set(hole)
            rng.uniform(-90, 90)
            rng.uniform(-180, 180)
            for _ in range(4):
                rng.uniform(-90, 90)
                rng.uniform(-180, 180)
        assert 0.45 < reversed_outer / 1000 < 0.55
        assert 0.45 < reversed_hole / 1000 < 0.55


class TestRoundTrip:
    def test_points_and_lines_return_to_the_bit_with_axes_swapped(self):
        rng = random.Random(245)
        for _ in range(500):
            p = (rng.uniform(-90, 90), rng.uniform(-180, 180))
            obj = point_object(p)
            assert obj["coordinates"] == [p[1], p[0]]
            assert loads(dumps(obj)) == ("Point", p)
            line = [(rng.uniform(-90, 90), rng.uniform(-180, 180)) for _ in range(4)]
            assert loads(dumps(linestring_object(line))) == ("LineString", line)

    def test_features_carry_properties_and_parse_to_their_geometry(self):
        text = dumps(feature(point_object((51.5, -0.1)), {"name": "London"}))
        assert '"coordinates":[-0.1,51.5]' in text
        assert loads(text) == ("Point", (51.5, -0.1))


class TestRefusals:
    @pytest.mark.parametrize(
        "text",
        [
            '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1]]]}',
            '{"type":"Polygon","coordinates":[]}',
            '{"type":"Point","coordinates":[1]}',
            '{"type":"Blob","coordinates":[]}',
            '{"type":"Feature"}',
            "not json",
            "[1,2]",
        ],
    )
    def test_forbidden_text_is_refused(self, text):
        with pytest.raises(Invalid):
            loads(text)

    def test_bad_geometries_are_refused_on_the_way_out(self):
        with pytest.raises(Invalid):
            polygon_object([[(0, 0), (1, 1), (2, 2)]])
        with pytest.raises(Invalid):
            polygon_object([])
        with pytest.raises(Invalid):
            linestring_object([(0, 0)])
