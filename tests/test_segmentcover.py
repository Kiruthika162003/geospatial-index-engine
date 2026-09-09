from __future__ import annotations

import itertools
import random

import pytest

from atlas.errors import Invalid
from atlas.segmentcover import (
    covered_length,
    expected_gaps,
    gaps,
    max_multiplicity,
    merge,
    multiplicity,
    poisson_coverage,
    random_spans,
)


class TestIdentities:
    def test_merged_spans_are_disjoint_and_the_accounting_holds_to_rounding(self):
        rng = random.Random(262)
        for _ in range(150):
            spans = random_spans(rng.randint(1, 40), rng.uniform(0.5, 20), 100.0, rng)
            merged = merge(spans)
            assert all(b0 < a1 for (_, b0), (a1, _) in itertools.pairwise(merged))
            total = covered_length(spans) + sum(b - a for a, b in gaps(spans, 0, 100))
            assert total == pytest.approx(100.0, abs=1e-9)
            integral = sum((b - a) * c for a, b, c in multiplicity(spans))
            assert integral == pytest.approx(sum(b - a for a, b in spans), abs=1e-9)

    def test_small_examples(self):
        assert merge([(1, 3), (2, 5), (7, 8), (8, 9)]) == [(1, 5), (7, 9)]
        assert gaps([(1, 3), (2, 5), (7, 8)], 0, 10) == [(0, 1), (5, 7), (8, 10)]
        assert multiplicity([(0, 4), (2, 6), (3, 5)]) == [
            (0, 2, 1),
            (2, 3, 2),
            (3, 4, 3),
            (4, 5, 2),
            (5, 6, 1),
        ]
        assert max_multiplicity([(0, 4), (2, 6), (3, 5)]) == 3


class TestRandomLaws:
    @pytest.mark.parametrize(
        ("n", "length", "cover", "gap_count"),
        [(5, 10.0, 0.3893, 3.95), (20, 5.0, 0.6253, 8.18), (200, 0.5, 0.6315, 74.34)],
    )
    def test_coverage_follows_poisson_within_the_edge_effect_and_gaps_run_one_high(
        self, n, length, cover, gap_count
    ):
        rng = random.Random(262)
        for _ in range(150):
            random_spans(rng.randint(1, 40), rng.uniform(0.5, 20), 100.0, rng)
        rng = random.Random(262 + n)
        fracs, counts = [], []
        for _ in range(300):
            spans = random_spans(n, length, 100.0, rng)
            fracs.append(covered_length(spans) / 100.0)
            counts.append(len(gaps(spans, 0, 100)))
        mean_cover, mean_gaps = sum(fracs) / 300, sum(counts) / 300
        assert mean_cover == pytest.approx(cover, abs=0.02)
        assert mean_cover == pytest.approx(poisson_coverage(n, length, 100), abs=0.02)
        assert mean_gaps == pytest.approx(gap_count, abs=0.6)
        assert mean_gaps == pytest.approx(expected_gaps(n, length, 100) + 1, abs=0.6)

    def test_at_full_coverage_only_the_leading_gap_remains(self):
        rng = random.Random(264)
        counts = [len(gaps(random_spans(200, 5.0, 100.0, rng), 0, 100)) for _ in range(100)]
        assert sum(counts) / 100 == pytest.approx(1.0, abs=0.05)
        assert expected_gaps(200, 5.0, 100) < 0.02


class TestRefusals:
    def test_backward_spans_empty_lines_and_negative_counts_are_refused(self):
        with pytest.raises(Invalid):
            merge([(3, 1)])
        with pytest.raises(Invalid):
            gaps([], 5, 5)
        with pytest.raises(Invalid):
            random_spans(-1, 1, 10, random.Random(0))
        with pytest.raises(Invalid):
            multiplicity([(2, 1)])
