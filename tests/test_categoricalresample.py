from __future__ import annotations

import random

import pytest

from atlas.categoricalresample import (
    fragments,
    landscape,
    lost_classes,
    majority,
    majority_survival_law,
    nearest,
    road_scene,
    share_error,
    shares,
)
from atlas.errors import Invalid


@pytest.fixture(scope="module")
def scene():
    return landscape(96, random.Random(600), 4, 6, 0.05)


class TestTheLandscape:
    def test_the_scene(self, scene):
        read = shares(scene)
        assert read[4] == pytest.approx(0.0511, abs=1e-4)
        assert len(read) == 5
        assert fragments(scene) == 559

    @pytest.mark.parametrize(
        ("factor", "near_error", "sprinkle", "major_error", "near_patches", "major_patches"),
        [
            (2, 0.0043, 0.0525, 0.0511, 243, 131),
            (3, 0.0098, 0.0566, 0.0511, 183, 131),
            (4, 0.0237, 0.0608, 0.0957, 167, 116),
            (8, 0.0566, 0.0486, 0.0637, 85, 80),
        ],
    )
    def test_nearest_keeps_the_minority_and_majority_erases_it(
        self, scene, factor, near_error, sprinkle, major_error, near_patches, major_patches
    ):
        near, major = nearest(scene, factor), majority(scene, factor)
        assert share_error(scene, near) == pytest.approx(near_error, abs=1e-4)
        assert lost_classes(scene, near) == 0
        assert shares(near)[4] == pytest.approx(sprinkle, abs=1e-4)
        assert share_error(scene, major) == pytest.approx(major_error, abs=1e-4)
        assert lost_classes(scene, major) == 1
        assert 4 not in shares(major)
        assert (fragments(near), fragments(major)) == (near_patches, major_patches)

    def test_the_binomial_law_and_the_seed_average(self):
        assert majority_survival_law(2, 0.05) == pytest.approx(0.00048, abs=1e-5)
        assert majority_survival_law(3, 0.05) == pytest.approx(3e-5, abs=1e-5)
        assert majority_survival_law(1, 0.05) == pytest.approx(0.05)
        near, major = [], []
        for seed in range(10):
            sc = landscape(96, random.Random(610 + seed), 4, 6, 0.05)
            near.append(share_error(sc, nearest(sc, 4)))
            major.append(share_error(sc, majority(sc, 4)))
        assert sum(near) / 10 == pytest.approx(0.0274, abs=1e-4)
        assert sum(major) / 10 == pytest.approx(0.0968, abs=1e-4)


class TestRoads:
    def test_thin_roads_survive_nearest_only_by_phase(self):
        roads = road_scene(96, random.Random(601), 4)
        assert shares(roads)[9] == pytest.approx(0.0417, abs=1e-4)
        readings = [shares(nearest(roads, f)).get(9, 0.0) for f in (2, 3, 4)]
        assert readings == pytest.approx([0.0208, 0.0, 0.0833], abs=1e-4)
        assert all(shares(majority(roads, f)).get(9, 0.0) == 0.0 for f in (2, 3, 4))


class TestRefusals:
    def test_bad_grids_factors_and_landscapes(self):
        with pytest.raises(Invalid):
            nearest([], 2)
        with pytest.raises(Invalid):
            majority([[0, 1, 2]], 2)
        with pytest.raises(Invalid):
            landscape(8, random.Random(1), 1, 2)
        with pytest.raises(Invalid):
            majority_survival_law(0, 0.5)
