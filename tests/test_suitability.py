from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.suitability import (
    gradient_layer,
    noise_layer,
    normalise,
    overlap,
    overlay,
    perturbed_weights,
    rank_correlation,
    smooth_layer,
    stability,
    top_cells,
)

WEIGHTS = [3.0, 2.0, 1.0, 1.0]


@pytest.fixture(scope="module")
def scene():
    rng = random.Random(640)
    layers = [
        smooth_layer(48, rng),
        smooth_layer(48, rng),
        noise_layer(48, rng),
        gradient_layer(48),
    ]
    return layers, overlay(layers, WEIGHTS)


class TestWobble:
    @pytest.mark.parametrize(
        ("fraction", "readings"),
        [
            (0.05, (0.9684, 0.9477, 0.8956, 0.719)),
            (0.1, (0.9631, 0.9279, 0.8684, 0.7201)),
            (0.25, (0.9808, 0.9612, 0.9187, 0.7755)),
        ],
    )
    def test_the_top_cells_churn_while_the_ranking_holds(self, scene, fraction, readings):
        layers, _ = scene
        read = tuple(
            stability(layers, WEIGHTS, fraction, a, random.Random(641))
            for a in (0.05, 0.1, 0.2, 0.5)
        )
        assert read == pytest.approx(readings, abs=1e-4)

    @pytest.mark.parametrize(
        ("amount", "mean", "worst"),
        [(0.05, 0.9993, 0.998), (0.1, 0.9973, 0.9919), (0.5, 0.9413, 0.8462)],
    )
    def test_rank_correlation_under_wobble(self, scene, amount, mean, worst):
        layers, base = scene
        rng = random.Random(642)
        rhos = [
            rank_correlation(base, overlay(layers, perturbed_weights(WEIGHTS, amount, rng)))
            for _ in range(20)
        ]
        assert sum(rhos) / 20 == pytest.approx(mean, abs=1e-4)
        assert min(rhos) == pytest.approx(worst, abs=1e-4)


class TestDropping:
    @pytest.mark.parametrize(
        ("drop", "kept", "rho"),
        [(0, 0.3295, 0.6195), (1, 0.3181, 0.6769), (2, 0.6312, 0.9182), (3, 0.749, 0.9276)],
    )
    def test_dropping_a_layer(self, scene, drop, kept, rho):
        layers, base = scene
        weights = list(WEIGHTS)
        weights[drop] = 0.0
        other = overlay(layers, weights)
        assert overlap(top_cells(base, 0.1), top_cells(other, 0.1)) == pytest.approx(
            kept, abs=1e-4
        )
        assert rank_correlation(base, other) == pytest.approx(rho, abs=1e-4)

    def test_equal_weights_and_noise_alone(self, scene):
        layers, base = scene
        equal = overlay(layers, [1, 1, 1, 1])
        assert overlap(top_cells(base, 0.1), top_cells(equal, 0.1)) == pytest.approx(
            0.5646, abs=1e-4
        )
        assert rank_correlation(base, equal) == pytest.approx(0.9015, abs=1e-4)
        assert overlap(top_cells(base, 0.1), top_cells(layers[2], 0.1)) == pytest.approx(
            0.0824, abs=1e-4
        )


class TestPiecesAndRefusals:
    def test_normalise_and_refusals(self):
        assert normalise([[2.0, 4.0], [6.0, 8.0]]) == [[0.0, 1 / 3], [2 / 3, 1.0]]
        assert normalise([[5.0, 5.0]]) == [[0.0, 0.0]]
        with pytest.raises(Invalid):
            overlay([], [])
        with pytest.raises(Invalid):
            overlay([[[1.0]]], [1.0, 2.0])
        with pytest.raises(Invalid):
            overlay([[[1.0]]], [0.0])
        with pytest.raises(Invalid):
            overlay([[[1.0]], [[1.0, 2.0]]], [1.0, 1.0])
        with pytest.raises(Invalid):
            top_cells([[1.0]], 0.0)
        with pytest.raises(Invalid):
            overlap(set(), {(0, 0)})
        with pytest.raises(Invalid):
            perturbed_weights([1.0], -0.1, random.Random(1))
        with pytest.raises(Invalid):
            rank_correlation([[1.0]], [[1.0, 2.0]])
