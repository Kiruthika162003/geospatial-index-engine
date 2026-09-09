from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.bboxquantize import (
    Quantizer,
    bits_for_cell,
    contains_original,
    expected_inflation,
    false_hit_rate,
    inflation,
    mean_inflation,
    random_boxes,
)
from atlas.errors import Invalid

EXTENT = BBox(0, 0, 1000, 1000)


@pytest.fixture(scope="module")
def scene():
    rng = random.Random(470)
    boxes = {size: random_boxes(2000, rng, size) for size in (1.0, 5.0, 20.0, 100.0)}
    return boxes, random_boxes(200, random.Random(471), 50.0)


class TestFat:
    @pytest.mark.parametrize(
        ("size", "readings"),
        [
            (1.0, ((4, 4604.44), (8, 23.95), (12, 1.55), (16, 1.03))),
            (5.0, ((4, 203.02), (8, 3.17), (12, 1.1), (16, 1.01))),
            (100.0, ((4, 2.74), (8, 1.08), (12, 1.0), (16, 1.0))),
        ],
    )
    def test_inflation_follows_the_one_cell_law(self, scene, size, readings):
        boxes, _ = scene
        for bits, reading in readings:
            quantizer = Quantizer(EXTENT, bits)
            read = mean_inflation(boxes[size], quantizer)
            assert read == pytest.approx(reading, abs=0.01)
            law = expected_inflation(size, 1000 / quantizer.steps)
            assert read == pytest.approx(law, rel=0.025)
            assert all(contains_original(quantizer.fatten(b), b) for b in boxes[size][:300])


class TestFalseHits:
    @pytest.mark.parametrize(
        ("size", "rates"),
        [
            (1.0, (0.8066, 0.4111, 0.1309, 0.0401, 0.0128, 0.0019)),
            (20.0, (0.7329, 0.3269, 0.0928, 0.0259, 0.0066, 0.0005)),
            (100.0, (0.4929, 0.1686, 0.0454, 0.0115, 0.0029, 0.0002)),
        ],
    )
    def test_never_a_miss_and_the_false_share_by_bits(self, scene, size, rates):
        boxes, queries = scene
        for bits, rate in zip((4, 6, 8, 10, 12, 16), rates, strict=True):
            quantizer = Quantizer(EXTENT, bits)
            read, true_hits, fat_hits, misses = false_hit_rate(boxes[size], queries, quantizer)
            assert read == pytest.approx(rate, abs=1e-4)
            assert misses == 0
            assert fat_hits >= true_hits


class TestCodes:
    def test_encoding_bytes_and_bits(self):
        q = Quantizer(EXTENT, 8)
        code = q.encode(BBox(10.2, 20.7, 33.3, 41.1))
        assert code == (2, 5, 9, 11)
        fat = q.decode(code)
        assert fat.contains_box(BBox(10.2, 20.7, 33.3, 41.1))
        assert inflation(fat, BBox(10.2, 20.7, 33.3, 41.1)) > 1
        assert q.bytes_per_box() == 4.0
        assert Quantizer(EXTENT, 16).bytes_per_box() == 8.0
        assert bits_for_cell(1000, 1.0) == 10
        assert bits_for_cell(1000, 0.01) == 17
        assert inflation(fat, BBox(1, 1, 1, 1)) == float("inf")

    def test_refusals(self):
        with pytest.raises(Invalid):
            Quantizer(EXTENT, 0)
        with pytest.raises(Invalid):
            Quantizer(BBox(0, 0, 0, 5), 8)
        with pytest.raises(Invalid):
            Quantizer(EXTENT, 8).encode(BBox(-1, 0, 5, 5))
