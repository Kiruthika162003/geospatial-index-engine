from __future__ import annotations

import random

import pytest

from atlas.dasymetric import (
    areal_weighting,
    block_zones,
    class_weights,
    corrupted,
    dasymetric,
    mask_weights,
    rms_by_cell,
    rms_by_zone,
    synthetic_town,
    total,
    zone_totals,
)
from atlas.errors import Invalid

RIGHT = {0: 0.0, 1: 25.0, 2: 100.0}


@pytest.fixture(scope="module")
def town():
    truth, classes = synthetic_town(40, random.Random(360))
    source = block_zones(40, 10)
    return truth, classes, source, zone_totals(truth, source)


class TestTheTown:
    def test_the_scene(self, town):
        truth, classes, _source, totals = town
        assert total(truth) == pytest.approx(13037.1, abs=0.1)
        assert sum(1 for row in classes for v in row if v) == 306
        assert sum(1 for row in classes for v in row if v == 2) == 69
        assert len(totals) == 16

    @pytest.mark.parametrize(
        ("name", "cell", "fine", "shifted"),
        [
            ("areal", 15.51, 214.408, 489.42),
            ("binary", 11.238, 142.863, 318.238),
            ("classes", 3.285, 18.469, 23.536),
            ("swapped", 22.146, 305.017, 622.373),
        ],
    )
    def test_errors_by_method_and_conservation(self, town, name, cell, fine, shifted):
        truth, classes, source, totals = town
        weights = {
            "binary": mask_weights([[v > 0 for v in r] for r in classes]),
            "classes": class_weights(classes, RIGHT),
            "swapped": class_weights(classes, {0: 0.0, 1: 100.0, 2: 25.0}),
        }
        if name == "areal":
            estimate = areal_weighting(totals, source)
        else:
            estimate = dasymetric(totals, source, weights[name])
        assert total(estimate) == pytest.approx(total(truth), abs=1e-8)
        assert rms_by_cell(estimate, truth) == pytest.approx(cell, abs=1e-3)
        assert rms_by_zone(estimate, truth, block_zones(40, 5)) == pytest.approx(fine, abs=1e-3)
        moved = rms_by_zone(estimate, truth, block_zones(40, 10, 5))
        assert moved == pytest.approx(shifted, abs=1e-3)
        assert rms_by_zone(estimate, truth, block_zones(40, 20)) == pytest.approx(0.0, abs=1e-9)

    def test_a_flat_weight_is_the_mask_again(self, town):
        truth, classes, source, totals = town
        flat = dasymetric(totals, source, class_weights(classes, {0: 0.0, 1: 1.0, 2: 1.0}))
        binary = dasymetric(totals, source, mask_weights([[v > 0 for v in r] for r in classes]))
        assert rms_by_cell(flat, truth) == pytest.approx(rms_by_cell(binary, truth), abs=1e-9)


class TestDamageAndScale:
    @pytest.mark.parametrize(
        ("fraction", "classed", "binary"),
        [(0.1, 52.496, 140.316), (0.5, 123.078, 162.477), (1.0, 165.432, 198.899)],
    )
    def test_corrupted_labels(self, town, fraction, classed, binary):
        truth, classes, source, totals = town
        bad = corrupted(classes, fraction, random.Random(361))
        target = block_zones(40, 5)
        estimate = dasymetric(totals, source, class_weights(bad, RIGHT))
        assert rms_by_zone(estimate, truth, target) == pytest.approx(classed, abs=1e-3)
        mask = dasymetric(totals, source, mask_weights([[v > 0 for v in r] for r in bad]))
        assert rms_by_zone(mask, truth, target) == pytest.approx(binary, abs=1e-3)

    @pytest.mark.parametrize(
        ("block", "areal", "dasy"),
        [(5, 0.0, 0.0), (20, 410.436, 18.641), (40, 439.796, 20.023)],
    )
    def test_coarser_sources_hurt_areal_weighting_only(self, town, block, areal, dasy):
        truth, classes, _, _ = town
        source = block_zones(40, block)
        totals = zone_totals(truth, source)
        target = block_zones(40, 5)
        assert rms_by_zone(areal_weighting(totals, source), truth, target) == pytest.approx(
            areal, abs=1e-3
        )
        estimate = dasymetric(totals, source, class_weights(classes, RIGHT))
        assert rms_by_zone(estimate, truth, target) == pytest.approx(dasy, abs=1e-3)


class TestRefusals:
    def test_bad_shapes_weights_and_towns(self):
        with pytest.raises(Invalid):
            zone_totals([], [])
        with pytest.raises(Invalid):
            dasymetric({0: 1.0}, [[0, 0]], [[1.0]])
        with pytest.raises(Invalid):
            dasymetric({0: 1.0}, [[0, 0]], [[1.0, -1.0]])
        with pytest.raises(Invalid):
            block_zones(4, 0)
        with pytest.raises(Invalid):
            synthetic_town(4, random.Random(1))
        with pytest.raises(Invalid):
            corrupted([[0]], 2.0, random.Random(1))
        empty = dasymetric({0: 10.0}, [[0, 0]], [[0.0, 0.0]])
        assert empty == [[5.0, 5.0]]
