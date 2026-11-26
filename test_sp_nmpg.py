"""Verify the spectral initializer and the projected line search."""

import unittest

import numpy as np

from cdp import CDPOperator
from sp_nmpg import solve_sp_nmpg, spectral_initialization


class Quadratic:
    def value(self, image):
        return float(50 * np.sum((image - 0.4) ** 2))

    def gradient(self, image):
        return 100 * (image - 0.4)

    def value_and_gradient(self, image):
        return self.value(image), self.gradient(image)


class RecordingRosenbrock:
    def __init__(self):
        self.iterates = []

    def value(self, image):
        x, y = image
        return float((x - 0.8) ** 2 + 100 * (y - x**2) ** 2)

    def gradient(self, image):
        self.iterates.append(image.copy())
        x, y = image
        return np.array([2 * (x - 0.8) - 400 * x * (y - x**2), 200 * (y - x**2)])

    def value_and_gradient(self, image):
        return self.value(image), self.gradient(image)


class SpectralInitializationTests(unittest.TestCase):
    def test_30_truncated_iterations_match_dense_operator(self):
        shape = (2, 3)
        operator = CDPOperator.random(shape, 3, np.random.default_rng(9))
        intensity = np.linspace(0.1, 1.5, 18).reshape(operator.measurement_shape)
        intensity[0, 0, 0] = 100
        weights = intensity.ravel().copy()
        weights[0] = 0
        basis_images = np.eye(6).reshape(6, *shape)
        design = np.column_stack([operator.forward(image).ravel() for image in basis_images])
        spectral_matrix = (design.conj().T @ np.diag(weights) @ design).real
        start = np.random.default_rng(12).standard_normal(6)
        reference = np.linalg.matrix_power(spectral_matrix, 30) @ start
        reference /= np.linalg.norm(reference)
        if np.sum(reference[reference < 0] ** 2) > np.sum(reference[reference > 0] ** 2):
            reference *= -1
        expected = np.clip(np.sqrt(6 * intensity.mean()) * reference, 0, 1).reshape(shape)
        actual = spectral_initialization(operator, intensity, np.random.default_rng(12))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_all_zero_intensity_returns_zero_image(self):
        rng = np.random.default_rng(13)
        operator = CDPOperator.random((2, 3), 2, rng)
        initial = spectral_initialization(operator, np.zeros(operator.measurement_shape), rng)
        np.testing.assert_array_equal(initial, np.zeros(operator.image_shape))


class ProjectedLineSearchTests(unittest.TestCase):
    def test_backtracking_logs_rejected_trial(self):
        result = solve_sp_nmpg(Quadratic(), [0.55])
        first_step = result.history[1]
        self.assertEqual(first_step.bb_step, 1.0)
        self.assertEqual(first_step.step_size, 0.5)
        self.assertEqual(first_step.backtracks, 1)
        self.assertEqual(first_step.objective_evaluations, 3)
        self.assertEqual(first_step.gradient_evaluations, 2)
        self.assertEqual(result.stop_reason, "kkt_tolerance")
        np.testing.assert_allclose(result.image, [0.4], atol=1e-12, rtol=0)

    def test_budget_retains_last_accepted_iterate(self):
        result = solve_sp_nmpg(Quadratic(), [0.55], max_objective_evaluations=2)
        self.assertEqual(result.stop_reason, "evaluation_budget")
        self.assertEqual(result.objective_evaluations, 2)
        self.assertEqual(result.gradient_evaluations, 1)
        self.assertEqual(len(result.history), 1)
        np.testing.assert_array_equal(result.image, [0.55])

    def test_armijo_uses_recent_five_values_and_allows_an_increase(self):
        loss = RecordingRosenbrock()
        result = solve_sp_nmpg(loss, [0.25, 0.9], max_iterations=30)
        iterates = tuple(loss.iterates)
        self.assertEqual(len(result.history), len(iterates))
        self.assertTrue(np.all((np.array(iterates) >= 0) & (np.array(iterates) <= 1)))
        accepted_increase = False
        for i, row in enumerate(result.history[1:], start=1):
            previous = result.history[i - 1]
            accepted_increase |= row.objective > previous.objective
            reference = max(entry.objective for entry in result.history[max(0, i - 5) : i])
            slope = loss.gradient(iterates[i - 1]) @ (iterates[i] - iterates[i - 1])
            self.assertLessEqual(row.objective, reference + 1e-4 * slope + 1e-14)
        self.assertTrue(accepted_increase)


if __name__ == "__main__":
    unittest.main()
