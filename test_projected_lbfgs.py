"""Verify two-metric projection, limited-memory products, and Powell damping."""

import unittest
from collections import deque

import numpy as np

from cdp import AmplitudeLoss, CDPOperator, relative_error
from projected_lbfgs import (
    _hessian_product,
    _inverse_product,
    _positive_pair,
    _restrict_pairs,
    solve_projected_lbfgs,
)
from sp_nmpg import spectral_initialization


class RecordingQuadratic:
    def __init__(self, matrix, linear):
        self.matrix = np.asarray(matrix)
        self.linear = np.asarray(linear)
        self.evaluated_points = []

    def value(self, image):
        self.evaluated_points.append(image.copy())
        return float(0.5 * image @ self.matrix @ image + self.linear @ image)

    def gradient(self, image):
        return self.matrix @ image + self.linear

    def value_and_gradient(self, image):
        return self.value(image), self.gradient(image)


class LimitedMemoryTests(unittest.TestCase):
    def test_products_match_dense_bfgs_updates(self):
        hessian = np.array([[3., 1., 0.5], [1., 4., -1.], [0.5, -1., 2.]])
        steps = [np.array([1., 0., 1.]), np.array([-1., 1., 0.]), np.array([0.2, 0., -1.])]
        pairs = [_positive_pair(step, hessian @ step) for step in steps]
        self.assertTrue(all(pair is not None for pair in pairs))

        inverse_scale = 0.25
        dense = np.eye(3) / inverse_scale
        for pair in pairs:
            b_step = dense @ pair.step
            dense = (dense - np.outer(b_step, b_step) / (pair.step @ b_step)
                     + np.outer(pair.change, pair.change) / (pair.step @ pair.change))
        vector = np.array([-1.5, 2., 0.3])
        np.testing.assert_allclose(
            _inverse_product(vector, pairs, inverse_scale), np.linalg.solve(dense, vector)
        )
        np.testing.assert_allclose(
            _hessian_product(vector, pairs, inverse_scale), dense @ vector
        )

    def test_restriction_rejects_pair_with_negative_reduced_curvature(self):
        original = _positive_pair(np.array([1., 1.]), np.array([-1., 2.]))
        memory = deque([original])
        self.assertEqual(_restrict_pairs(memory, np.array([True, False])), [])
        remaining = _restrict_pairs(memory, np.array([False, True]))
        self.assertEqual(remaining[0].curvature, 2)


class ProjectedLBFGSTests(unittest.TestCase):
    def test_damping_repairs_negative_curvature_and_disabled_mode_skips_it(self):
        loss = RecordingQuadratic([[-0.1]], [0.03])
        damped = solve_projected_lbfgs(loss, [0.5], max_iterations=1)
        undamped = solve_projected_lbfgs(loss, [0.5], damping=False, max_iterations=1)
        repaired, skipped = damped.history[1], undamped.history[1]
        self.assertLess(repaired.raw_curvature, 0)
        self.assertTrue(repaired.damped)
        self.assertFalse(repaired.pair_skipped)
        self.assertGreater(repaired.stored_curvature, 0)
        self.assertAlmostEqual(repaired.stored_curvature, 0.2 * repaired.model_curvature)
        self.assertEqual(repaired.memory_size, 1)
        self.assertFalse(skipped.damped)
        self.assertTrue(skipped.pair_skipped)
        self.assertIsNone(skipped.stored_curvature)
        self.assertEqual(skipped.memory_size, 0)
        np.testing.assert_array_equal(damped.image, undamped.image)

    def test_non_descent_direction_falls_back_and_all_trials_are_feasible(self):
        matrix = [[7.539, 14.754, -4.464], [14.754, 40.636, -12.099], [-4.464, -12.099, 4.204]]
        loss = RecordingQuadratic(matrix, [3.282, 1.387, -1.742])
        result = solve_projected_lbfgs(loss, [0.212, 0.231, 0.224], tolerance=1e-7)
        self.assertEqual(result.stop_reason, "kkt_tolerance")
        fallback_rows = [row for row in result.history if row.fallback == "non_descent"]
        self.assertGreaterEqual(len(fallback_rows), 1)
        self.assertGreaterEqual(result.direction_fallbacks, len(fallback_rows))
        self.assertGreater(result.backtracks, 0)
        self.assertEqual(result.objective_evaluations, len(loss.evaluated_points))
        points = np.array(loss.evaluated_points)
        self.assertTrue(np.all((points >= 0) & (points <= 1)))
        self.assertGreater(result.history[-1].active_variables, 0)
        self.assertTrue(np.all(np.diff([row.objective for row in result.history]) <= 0))

    def test_evaluation_budget_stops_at_last_accepted_point(self):
        loss = RecordingQuadratic([[1.0]], [0.0])
        result = solve_projected_lbfgs(loss, [0.5], max_objective_evaluations=1)
        self.assertEqual(result.stop_reason, "evaluation_budget")
        self.assertEqual(result.objective_evaluations, 1)
        self.assertEqual(result.gradient_evaluations, 1)
        np.testing.assert_array_equal(result.image, [0.5])

    def test_unconstrained_mode_uses_gradient_stationarity_outside_box(self):
        loss = RecordingQuadratic([[1.0]], [-2.0])
        bounded = solve_projected_lbfgs(loss, [0.5])
        unbounded = solve_projected_lbfgs(loss, [0.5], box_constrained=False)
        np.testing.assert_array_equal(bounded.image, [1.0])
        np.testing.assert_array_equal(unbounded.image, [2.0])
        self.assertEqual(unbounded.history[-1].kkt_residual, 0)
        self.assertEqual(unbounded.history[-1].active_variables, 0)

    def test_small_noiseless_cdp_with_and_without_damping(self):
        rng = np.random.default_rng(25)
        truth = rng.random((8, 8))
        truth[0, 0], truth[-1, -1] = 0, 1
        operator = CDPOperator.random(truth.shape, 6, rng)
        observed = operator.intensity(truth)
        initial = spectral_initialization(operator, observed, rng)
        loss = AmplitudeLoss(operator, observed, epsilon=0.01)
        for damping in (True, False):
            with self.subTest(damping=damping):
                result = solve_projected_lbfgs(loss, initial, damping=damping, tolerance=1e-11)
                self.assertEqual(result.stop_reason, "kkt_tolerance")
                self.assertLess(relative_error(truth, result.image), 1e-7)
                self.assertTrue(np.all((result.image >= 0) & (result.image <= 1)))
                self.assertEqual(max(row.memory_size for row in result.history), 10)
                for row in result.history[1:]:
                    if not row.pair_skipped:
                        self.assertGreater(row.stored_curvature, 0)


if __name__ == "__main__":
    unittest.main()
