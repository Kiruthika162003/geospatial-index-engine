from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.idw import best_power
from atlas.tin import TIN, barycentric, flipped, grid_samples, mean_min_angle_deg, rms_error


def plane(x: float, y: float) -> float:
    return 2.0 + 0.5 * x - 0.3 * y


def bowl(x: float, y: float) -> float:
    return (x - 5) ** 2 / 10 + (y - 5) ** 2 / 10


def _layout():
    rng = random.Random(194)
    xy = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(60)]
    xy += [(0, 0), (10, 0), (0, 10), (10, 10)]
    probes = [(rng.uniform(1, 9), rng.uniform(1, 9)) for _ in range(400)]
    return xy, probes


class TestExactness:
    def test_a_plane_is_reproduced_to_floating_precision(self):
        xy, probes = _layout()
        tin = TIN([(x, y, plane(x, y)) for x, y in xy])
        assert len(tin.triangles) == 122
        assert rms_error(tin, plane, probes) < 1e-12
        assert tin.slope(5, 5) == pytest.approx((0.5, -0.3))
        assert all(abs(tin.height(x, y) - plane(x, y)) < 1e-9 for x, y in xy)

    def test_barycentric_weights_at_a_centroid(self):
        weights = barycentric(((0, 0), (3, 0), (0, 3)), 1, 1)
        assert weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


class TestTheQuadraticRate:
    def test_halving_the_spacing_quarters_the_bowl_error(self):
        _, probes = _layout()
        grids = [TIN(grid_samples(bowl, 10.0, n)) for n in (3, 5, 9, 17, 33)]
        errors = [rms_error(grid, bowl, probes) for grid in grids]
        assert errors == pytest.approx([0.9651, 0.2223, 0.05275, 0.01385, 0.003371], rel=1e-3)
        ratios = [errors[i] / errors[i + 1] for i in range(4)]
        assert ratios == pytest.approx([4.34, 4.22, 3.81, 4.11], abs=0.01)


class TestComparisons:
    def test_the_tin_beats_inverse_distance_weighting_by_a_third(self):
        xy, probes = _layout()
        samples = [(x, y, bowl(x, y)) for x, y in xy]
        tin_error = rms_error(TIN(samples), bowl, probes)
        _, idw_error = best_power(samples, bowl, probes, (1, 2, 3, 4, 6))
        assert tin_error == pytest.approx(0.1707, abs=1e-3)
        assert idw_error / tin_error == pytest.approx(1.33, abs=0.01)

    def test_thin_triangles_cost_a_quarter_more(self):
        xy, probes = _layout()
        tin = TIN([(x, y, bowl(x, y)) for x, y in xy])
        thin = flipped(tin)
        changed = sum(1 for a, b in zip(tin.triangles, thin.triangles, strict=True) if a != b)
        assert changed == 100
        assert mean_min_angle_deg(tin.triangles) == pytest.approx(24.0, abs=0.05)
        assert mean_min_angle_deg(thin.triangles) == pytest.approx(17.32, abs=0.05)
        assert rms_error(thin, bowl, probes) / rms_error(tin, bowl, probes) == pytest.approx(
            1.24, abs=0.01
        )
        assert all(thin.locate(*p) for p in probes)

    def test_a_cocircular_grid_is_unchanged_by_flips(self):
        _, probes = _layout()
        grid = TIN(grid_samples(bowl, 10.0, 9))
        before = rms_error(grid, bowl, probes)
        assert rms_error(flipped(grid), bowl, probes) == pytest.approx(before)
        assert mean_min_angle_deg(grid.triangles) == pytest.approx(45.0)


class TestRefusals:
    def test_outside_and_bad_inputs_are_refused(self):
        xy, _ = _layout()
        tin = TIN([(x, y, plane(x, y)) for x, y in xy])
        with pytest.raises(Outside):
            tin.height(20, 20)
        with pytest.raises(Invalid):
            barycentric(((0, 0), (1, 1), (2, 2)), 0.5, 0.5)
        with pytest.raises(Invalid):
            TIN([(0, 0, 1), (1, 1, 2)])
        with pytest.raises(Invalid):
            TIN([(0, 0, 1), (0, 0, 2), (0, 0, 3)])
        with pytest.raises(Invalid):
            grid_samples(bowl, 10.0, 1)
        with pytest.raises(Invalid):
            rms_error(tin, plane, [])
        with pytest.raises(Invalid):
            mean_min_angle_deg([])
