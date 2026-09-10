from __future__ import annotations

import random

import pytest

from atlas.centroidassignment import (
    by_area,
    by_centroid,
    cell_of,
    misplaced_law,
    misplaced_share,
    simulated_share,
    square_zones,
    tiled_zones,
    zone_centroid,
)
from atlas.errors import Invalid


class TestTheLaw:
    @pytest.mark.parametrize(
        ("ratio", "law", "simulated", "crowd"),
        [
            (0.1, 0.0494, 0.0502, 0.0449),
            (0.5, 0.2344, 0.2348, 0.2102),
            (1.0, 0.4375, 0.4411, 0.4169),
            (4.0, 0.9375, 0.9375, 0.8978),
        ],
    )
    def test_a_lone_zone_follows_the_law_and_a_crowd_misplaces_less(
        self, ratio, law, simulated, crowd
    ):
        assert misplaced_law(ratio) == pytest.approx(law, abs=1e-4)
        assert simulated_share(ratio, random.Random(751), 4000) == pytest.approx(
            simulated, abs=1e-4
        )
        assert abs(simulated - law) < 0.004
        rng = random.Random(750)
        for r in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            zones = square_zones(r * 10.0, 400, rng)
            values = [rng.uniform(1, 10) for _ in zones]
            if r == ratio:
                assert misplaced_share(zones, values, 10.0) == pytest.approx(crowd, abs=1e-4)
                assert crowd <= law + 1e-9


class TestTilings:
    def test_a_tiling_misplaces_only_at_the_edge(self):
        assert misplaced_share(tiled_zones(10.0), [1.0] * 10000, 10.0) == pytest.approx(
            0.0, abs=1e-9
        )
        assert misplaced_share(
            tiled_zones(10.0, 1000.0, 2.5), [1.0] * 10000, 10.0
        ) == pytest.approx(0.005, abs=1e-4)
        assert misplaced_share(
            tiled_zones(10.0, 1000.0, 5.0), [1.0] * 10000, 10.0
        ) == pytest.approx(0.01, abs=1e-4)

    def test_area_splitting_conserves(self):
        split = by_area([(0, 0, 10, 10)], [100.0], 4.0)
        assert sum(split.values()) == pytest.approx(100.0)
        assert split[(0, 0)] == pytest.approx(16.0)
        assert split[(2, 2)] == pytest.approx(4.0)
        assert by_centroid([(0, 0, 10, 10)], [100.0], 4.0) == {(1, 1): 100.0}
        assert zone_centroid((0, 0, 10, 10)) == (5.0, 5.0)
        assert cell_of((5.0, 5.0), 4.0) == (1, 1)


class TestRefusals:
    def test_bad_zones_cells_and_values(self):
        with pytest.raises(Invalid):
            zone_centroid((0, 0, 0, 5))
        with pytest.raises(Invalid):
            cell_of((0.0, 0.0), 0.0)
        with pytest.raises(Invalid):
            by_centroid([(0, 0, 1, 1)], [1.0, 2.0], 1.0)
        with pytest.raises(Invalid):
            by_area([(0, 0, 1, 1)], [1.0], 0.0)
        with pytest.raises(Invalid):
            misplaced_share([(0, 0, 1, 1)], [0.0], 1.0)
        with pytest.raises(Invalid):
            square_zones(0.0, 1, random.Random(1))
        with pytest.raises(Invalid):
            misplaced_law(0.0)
