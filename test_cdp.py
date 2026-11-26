"""Verification cases for the coded diffraction model."""

import unittest

import numpy as np

from cdp import (
    AmplitudeLoss,
    CDPOperator,
    add_intensity_noise,
    kkt_residual,
    out_of_bounds_fraction,
    project_box,
    psnr_db,
    relative_error,
    snr_db,
    ssim,
)


class CDPTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(25)
        self.operator = CDPOperator.random((3, 4), 3, self.rng)

    def test_complex_adjoint_identity(self):
        image = self.rng.normal(size=(3, 4)) + 1j * self.rng.normal(size=(3, 4))
        shape = self.operator.measurement_shape
        detector = self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)
        field = self.operator.forward(image)
        np.testing.assert_allclose(
            np.vdot(field, detector),
            np.vdot(image, self.operator.adjoint(detector)),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.operator.adjoint(field), shape[0] * image, rtol=0, atol=1e-12
        )

    def test_real_gradient_against_central_difference(self):
        truth = self.rng.random((3, 4))
        intensity = self.operator.intensity(truth)
        loss = AmplitudeLoss(self.operator, intensity, epsilon=0.15)
        image = self.rng.uniform(0.2, 0.8, size=(3, 4))
        value, gradient = loss.value_and_gradient(image)
        self.assertEqual(value, loss.value(image))
        np.testing.assert_allclose(gradient, loss.gradient(image), rtol=0, atol=0)
        for _ in range(8):
            direction = self.rng.normal(size=image.shape)
            direction /= np.linalg.norm(direction)
            step = 1e-5
            difference = (
                loss.value(image + step * direction) - loss.value(image - step * direction)
            ) / (2 * step)
            self.assertAlmostEqual(difference, float(np.sum(gradient * direction)), places=10)

    def test_noiseless_truth_has_zero_loss(self):
        truth = self.rng.random((3, 4))
        loss = AmplitudeLoss(self.operator, self.operator.intensity(truth), epsilon=0.05)
        value, gradient = loss.value_and_gradient(truth)
        self.assertLess(value, 1e-28)
        self.assertLess(kkt_residual(truth, gradient), 1e-13)

    def test_negative_intensity_is_clipped_for_loss(self):
        raw = self.rng.normal(size=self.operator.measurement_shape)
        original = raw.copy()
        image = self.rng.random((3, 4))
        raw_loss = AmplitudeLoss(self.operator, raw, epsilon=0.05)
        clipped_loss = AmplitudeLoss(self.operator, np.maximum(raw, 0), epsilon=0.05)
        self.assertEqual(raw_loss.value(image), clipped_loss.value(image))
        np.testing.assert_array_equal(raw_loss.gradient(image), clipped_loss.gradient(image))
        np.testing.assert_array_equal(raw, original)

    def test_zero_stationary_point_is_not_a_local_minimum(self):
        masks = np.array([[[1, 1], [1, 1]], [[1j, -1], [1, -1j]]])
        operator = CDPOperator(masks)
        truth = np.array([[1.0, 0.0], [0.0, 0.0]])
        intensity = operator.intensity(truth)
        np.testing.assert_allclose(intensity, 0.25, rtol=0, atol=1e-15)
        loss = AmplitudeLoss(operator, intensity, epsilon=0.25)
        zero = np.zeros_like(truth)
        value, gradient = loss.value_and_gradient(zero)
        self.assertAlmostEqual(value, (3 - np.sqrt(5)) / 16, places=14)
        self.assertEqual(kkt_residual(zero, gradient), 0)
        self.assertEqual(relative_error(truth, zero), 1)
        self.assertLess(loss.value(0.01 * truth), value)
        self.assertLess(loss.value(truth), 1e-28)


class EvaluationTests(unittest.TestCase):
    def test_box_projection_and_kkt_signs(self):
        np.testing.assert_array_equal(
            project_box([-1.0, 0.0, 0.4, 1.0, 2.0]), [0.0, 0.0, 0.4, 1.0, 1.0]
        )
        image = np.array([[0.0, 0.5, 1.0]])
        self.assertEqual(kkt_residual(image, [[0.4, 0.0, -0.2]]), 0.0)
        self.assertAlmostEqual(kkt_residual(image, [[-0.4, 0.1, 0.2]]), 0.4)

    def test_raw_noise_models_reach_requested_snr(self):
        clean = np.linspace(0, 2, 2 * 128 * 128).reshape(2, 128, 128)
        for model, target in (("gaussian", 30.0), ("poisson", 20.0)):
            with self.subTest(model=model):
                kwargs = {"model": model, "target_snr_db": target}
                observed = add_intensity_noise(clean, rng=np.random.default_rng(7), **kwargs)
                repeat = add_intensity_noise(clean, rng=np.random.default_rng(7), **kwargs)
                np.testing.assert_array_equal(observed, repeat)
                self.assertLess(abs(snr_db(clean, observed) - target), 0.2)
                if model == "poisson":
                    self.assertGreaterEqual(observed.min(), 0)
                    self.assertEqual(observed[0, 0, 0], 0)
                else:
                    self.assertLess(observed.min(), 0)

    def test_image_metrics(self):
        reference = np.full((12, 13), 0.5)
        estimate = np.full((12, 13), 0.75)
        self.assertEqual(relative_error(reference, estimate), 0.5)
        self.assertAlmostEqual(psnr_db(reference, estimate), 20 * np.log10(4))
        self.assertEqual(psnr_db(reference, reference), np.inf)
        self.assertAlmostEqual(ssim(reference, reference), 1)
        expected_ssim = (2 * 0.5 * 0.75 + 0.01**2) / (0.5**2 + 0.75**2 + 0.01**2)
        self.assertAlmostEqual(ssim(reference, estimate), expected_ssim)
        self.assertEqual(out_of_bounds_fraction([-0.1, 0, 0.4, 1, 1.1]), 0.4)


if __name__ == "__main__":
    unittest.main()
