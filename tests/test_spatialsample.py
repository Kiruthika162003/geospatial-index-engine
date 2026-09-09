from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.spatialsample import (
    bias_and_scatter,
    bowl,
    estimation_error,
    noise_field,
    plane,
    sample_mean,
    simple_random,
    stratified,
    stripes,
    systematic,
)

SIDE, N, TRIALS = 10, 100, 400


def _designs(rng):
    return {
        "random": lambda: simple_random(N, rng),
        "systematic": lambda: systematic(SIDE, rng),
        "stratified": lambda: stratified(SIDE, rng),
    }


class TestSmoothFields:
    def test_on_a_plane_the_grid_is_no_better_than_random_and_stratified_is_tenfold(self):
        designs = _designs(random.Random(227))
        errors = {d: estimation_error(plane, fn, 3.5, TRIALS) for d, fn in designs.items()}
        assert errors["random"] == pytest.approx(0.107, abs=0.005)
        assert errors["systematic"] == pytest.approx(0.105, abs=0.005)
        assert errors["stratified"] == pytest.approx(0.0101, abs=0.001)
        assert 0.9 < errors["random"] / errors["systematic"] < 1.1
        assert errors["random"] / errors["stratified"] > 9

    def test_on_a_bowl_the_grid_wins_tenfold_and_stratified_fourfold(self):
        rng = random.Random(227)
        designs = _designs(rng)
        for fn in designs.values():
            estimation_error(plane, fn, 3.5, TRIALS)
        errors = {d: estimation_error(bowl, fn, 1 / 6, TRIALS) for d, fn in designs.items()}
        assert errors["random"] == pytest.approx(0.01054, abs=0.001)
        assert errors["systematic"] == pytest.approx(0.00103, abs=0.0002)
        assert errors["stratified"] == pytest.approx(0.00245, abs=0.0003)
        assert errors["random"] / errors["systematic"] > 8
        assert 3.5 < errors["random"] / errors["stratified"] < 5


class TestPeriodicFields:
    def test_at_the_grids_own_period_the_grid_scatters_by_the_amplitude(self):
        designs = _designs(random.Random(229))
        field = stripes(0.1)
        readings = {d: bias_and_scatter(field, fn, 0.0, TRIALS) for d, fn in designs.items()}
        assert readings["systematic"][1] > 0.6
        assert readings["random"][1] < 0.1
        assert readings["stratified"][1] < 0.1
        assert abs(readings["systematic"][0]) < 0.1

    def test_at_an_unrelated_period_the_designs_agree(self):
        designs = _designs(random.Random(230))
        field = stripes(0.137)
        scatters = [bias_and_scatter(field, fn, 0.0, TRIALS)[1] for fn in designs.values()]
        assert max(scatters) / min(scatters) < 1.25


class TestNoise:
    def test_white_noise_makes_the_designs_indistinguishable(self):
        errors = []
        for fn in _designs(random.Random(231)).values():
            errors.append(estimation_error(noise_field(random.Random(228)), fn, 0.0, TRIALS))
        assert errors == pytest.approx([0.0983] * 3, abs=1e-3)
        assert errors[0] == pytest.approx(1 / math.sqrt(N), abs=0.005)


class TestRefusals:
    def test_bad_counts_and_empty_samples_are_refused(self):
        rng = random.Random(0)
        with pytest.raises(Invalid):
            simple_random(0, rng)
        with pytest.raises(Invalid):
            systematic(0, rng)
        with pytest.raises(Invalid):
            sample_mean(plane, [])
        with pytest.raises(Invalid):
            estimation_error(plane, lambda: [(0.5, 0.5)], 1.0, 0)
