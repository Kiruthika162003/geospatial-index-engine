from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.tilebloom import (
    TileBloom,
    independent_positions,
    measured_rate,
    optimal_hashes,
    predicted_rate,
)

N = 5000


def _tiles():
    rng = random.Random(226)
    seen = []
    for _ in range(N):
        seen.append((rng.randint(10, 15), rng.randint(0, 30000), rng.randint(0, 30000)))
    seen_set = set(seen)
    unseen = []
    while len(unseen) < 10000:
        t = (rng.randint(10, 15), rng.randint(0, 30000), rng.randint(0, 30000))
        if t not in seen_set:
            unseen.append(t)
    return seen, unseen


def _filled(bits: int, hashes: int, seen) -> TileBloom:
    f = TileBloom(bits, hashes)
    for t in seen:
        f.add(t)
    return f


class TestAgainstTheFormula:
    @pytest.mark.parametrize(
        ("bits_per_key", "hashes", "predicted", "measured", "fill"),
        [
            (4, 3, 0.1469, 0.1443, 0.526),
            (8, 6, 0.0216, 0.0208, 0.529),
            (16, 11, 0.0005, 0.0007, 0.497),
        ],
    )
    def test_the_rate_follows_the_formula_within_a_tenth_of_a_percent(
        self, bits_per_key, hashes, predicted, measured, fill
    ):
        seen, unseen = _tiles()
        bits = bits_per_key * N
        assert optimal_hashes(bits, N) == hashes
        assert predicted_rate(bits, N, hashes) == pytest.approx(predicted, abs=1e-4)
        f = _filled(bits, hashes, seen)
        assert measured_rate(f, unseen) == pytest.approx(measured, abs=1e-4)
        assert f.fill_fraction() == pytest.approx(fill, abs=1e-3)
        assert all(f.seen(t) for t in seen)

    def test_the_k_curve_bottoms_at_six(self):
        seen, unseen = _tiles()
        rates = {k: measured_rate(_filled(8 * N, k, seen), unseen) for k in (1, 3, 6, 10, 14)}
        assert rates[1] == pytest.approx(0.1161, abs=1e-4)
        assert rates[6] == pytest.approx(0.0208, abs=1e-4)
        assert rates[14] == pytest.approx(0.0701, abs=1e-4)
        assert min(rates, key=rates.get) == 6

    def test_the_double_hash_trick_matches_independent_hashes(self):
        seen, unseen = _tiles()

        class Reference(TileBloom):
            def _positions(self, tile):
                return independent_positions(tile, self.bits, self.hashes)

        reference = Reference(8 * N, 6)
        for t in seen:
            reference.add(t)
        assert measured_rate(reference, unseen) == pytest.approx(0.0205, abs=1e-4)
        assert abs(measured_rate(reference, unseen) - 0.0208) < 0.002

    def test_the_rate_halves_every_1_3_bits(self):
        seen, unseen = _tiles()
        rates = {}
        for b in (8, 10, 12):
            rates[b] = measured_rate(_filled(b * N, optimal_hashes(b * N, N), seen), unseen)
        assert rates[8] / rates[10] == pytest.approx(2.85, abs=0.05)
        assert rates[10] / rates[12] == pytest.approx(2.81, abs=0.05)


class TestRefusals:
    def test_bad_sizes_and_empty_probes_are_refused(self):
        with pytest.raises(Invalid):
            TileBloom(0, 1)
        with pytest.raises(Invalid):
            predicted_rate(10, 0, 1)
        with pytest.raises(Invalid):
            optimal_hashes(0, 5)
        with pytest.raises(Invalid):
            measured_rate(TileBloom(8, 1), [])
