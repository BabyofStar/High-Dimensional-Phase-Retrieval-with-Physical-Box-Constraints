"""Coded diffraction, the smoothed amplitude objective, and evaluation metrics."""

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray


def _real_array(values: ArrayLike, name: str) -> NDArray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    return array.astype(np.float64, copy=False)


class CDPOperator:
    """Unitary 2D FFTs with one unit-modulus phase mask per observation."""

    def __init__(self, masks: ArrayLike) -> None:
        phase_masks = np.array(masks, dtype=np.complex128, copy=True)
        if phase_masks.ndim != 3 or 0 in phase_masks.shape:
            raise ValueError("masks must have nonempty shape (count, height, width)")
        if not np.allclose(np.abs(phase_masks), 1.0):
            raise ValueError("masks must have unit modulus")
        self._masks = phase_masks
        self._conjugate_masks = phase_masks.conj()

    @classmethod
    def random(
        cls, shape: tuple[int, int], mask_count: int, rng: np.random.Generator
    ) -> "CDPOperator":
        """Draw independent phases uniformly on [0, 2π)."""
        height, width = shape
        if min(height, width, mask_count) <= 0:
            raise ValueError("image dimensions and mask count must be positive")
        phases = rng.uniform(0.0, 2 * np.pi, size=(mask_count, height, width))
        return cls(np.exp(1j * phases))

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._masks.shape[1:]

    @property
    def measurement_shape(self) -> tuple[int, int, int]:
        return self._masks.shape

    @property
    def phase_masks(self) -> NDArray:
        """Return a copy of the masks for saving or reproducing measurements."""
        return self._masks.copy()

    def forward(self, image: ArrayLike) -> NDArray:
        """Map a real or complex image to a stack of complex detector fields."""
        image = np.asarray(image)
        if image.shape != self.image_shape:
            raise ValueError(f"image must have shape {self.image_shape}")
        return np.fft.fft2(self._masks * image, axes=(-2, -1), norm="ortho")

    def adjoint(self, fields: ArrayLike) -> NDArray:
        """Apply the complex adjoint of forward."""
        fields = np.asarray(fields)
        if fields.shape != self.measurement_shape:
            raise ValueError(f"fields must have shape {self.measurement_shape}")
        spatial = np.fft.ifft2(fields, axes=(-2, -1), norm="ortho")
        return np.sum(self._conjugate_masks * spatial, axis=0)

    def intensity(self, image: ArrayLike) -> NDArray:
        """Simulate noiseless detector intensities."""
        return np.abs(self.forward(image)) ** 2


class AmplitudeLoss:
    """Smoothed amplitude least squares averaged over all mask and pixel pairs."""

    def __init__(
        self, operator: CDPOperator, intensity: ArrayLike, *, epsilon: float
    ) -> None:
        measured = _real_array(intensity, "intensity")
        if measured.shape != operator.measurement_shape:
            raise ValueError(f"intensity must have shape {operator.measurement_shape}")
        if not np.all(np.isfinite(measured)):
            raise ValueError("intensity must be finite")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.operator = operator
        self._epsilon = float(epsilon)
        self._observed_amplitude = np.hypot(np.sqrt(np.maximum(measured, 0)), epsilon)

    def _prediction(self, image: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        field = self.operator.forward(_real_array(image, "image"))
        amplitude = np.hypot(np.abs(field), self._epsilon)
        return field, amplitude, amplitude - self._observed_amplitude

    def _gradient(self, field: NDArray, amplitude: NDArray, residual: NDArray) -> NDArray:
        weighted_field = (residual / amplitude) * field
        return self.operator.adjoint(weighted_field).real / residual.size

    def value(self, image: ArrayLike) -> float:
        _, _, residual = self._prediction(image)
        return float(0.5 * np.mean(residual**2))

    def gradient(self, image: ArrayLike) -> NDArray:
        return self._gradient(*self._prediction(image))

    def value_and_gradient(self, image: ArrayLike) -> tuple[float, NDArray]:
        field, amplitude, residual = self._prediction(image)
        return float(0.5 * np.mean(residual**2)), self._gradient(field, amplitude, residual)


def project_box(image: ArrayLike) -> NDArray:
    """Project real pixels onto the closed interval [0, 1]."""
    return np.clip(_real_array(image, "image"), 0.0, 1.0)


def kkt_residual(image: ArrayLike, gradient: ArrayLike) -> float:
    """Infinity norm of the projected gradient mapping with a fixed unit step."""
    image, gradient = _matching_arrays(image, gradient)
    return float(np.max(np.abs(image - project_box(image - gradient))))


def add_intensity_noise(
    clean: ArrayLike,
    *,
    model: Literal["gaussian", "poisson"],
    target_snr_db: float,
    rng: np.random.Generator,
) -> NDArray:
    """Simulate raw noise at the requested ratio of signal to expected noise power."""
    clean = _real_array(clean, "clean intensity")
    if clean.size == 0 or not np.all(np.isfinite(clean) & (clean >= 0)):
        raise ValueError("clean intensity must be nonempty, finite, and nonnegative")
    if not np.isfinite(target_snr_db):
        raise ValueError("target SNR must be finite")
    peak = np.max(clean)
    if peak == 0:
        raise ValueError("a nonzero signal is required for finite target SNR")

    normalized = clean / peak
    signal_power = np.mean(normalized**2)
    with np.errstate(over="ignore", under="ignore"):
        noise_power = signal_power * np.power(10.0, -target_snr_db / 10.0)
    if not np.isfinite(noise_power) or noise_power <= 0:
        raise ValueError("target SNR is outside the representable range")

    if model == "gaussian":
        return clean + rng.normal(0.0, peak * np.sqrt(noise_power), size=clean.shape)
    if model == "poisson":
        # For counts with mean gain * intensity, dividing by gain gives
        # an unbiased intensity estimate with variance intensity / gain.
        gain = np.mean(normalized) / noise_power
        return (rng.poisson(gain * normalized) / gain) * peak
    raise ValueError(f"unknown noise model: {model}")


def _matching_arrays(reference: ArrayLike, estimate: ArrayLike) -> tuple[NDArray, NDArray]:
    reference = _real_array(reference, "reference")
    estimate = _real_array(estimate, "estimate")
    if reference.size == 0 or reference.shape != estimate.shape:
        raise ValueError("inputs must have the same nonempty shape")
    return reference, estimate


def snr_db(clean: ArrayLike, observed: ArrayLike) -> float:
    """Measure SNR in dB on the raw intensities, before any clipping."""
    clean, observed = _matching_arrays(clean, observed)
    signal = np.linalg.norm(clean)
    noise = np.linalg.norm(observed - clean)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(20 * (np.log10(signal) - np.log10(noise)))


def relative_error(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Relative L2 reconstruction error for a nonzero reference image."""
    reference, estimate = _matching_arrays(reference, estimate)
    reference_norm = np.linalg.norm(reference)
    if reference_norm == 0:
        raise ValueError("relative error requires a nonzero reference")
    return float(np.linalg.norm(estimate - reference) / reference_norm)


def psnr_db(reference: ArrayLike, estimate: ArrayLike) -> float:
    """PSNR in dB with the nominal image range fixed at [0, 1]."""
    reference, estimate = _matching_arrays(reference, estimate)
    mse = np.mean((estimate - reference) ** 2)
    with np.errstate(divide="ignore"):
        return float(-10 * np.log10(mse))


def ssim(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Mean SSIM over full 11×11 windows with a fixed unit image range.

    The Gaussian window (sigma 1.5) and constants follow Wang et al. (2004),
    https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
    """
    reference, estimate = _matching_arrays(reference, estimate)
    if reference.ndim != 2 or min(reference.shape) < 11:
        raise ValueError("SSIM requires 2D images at least 11 pixels along each axis")

    offsets = np.arange(-5, 6)
    weights = np.exp(-0.5 * (offsets / 1.5) ** 2)
    window = np.outer(weights, weights)
    window /= window.sum()

    def local_mean(values: NDArray) -> NDArray:
        patches = sliding_window_view(values, window.shape)
        return np.einsum("...ij,ij->...", patches, window)

    mean_ref = local_mean(reference)
    mean_est = local_mean(estimate)
    var_ref = np.maximum(local_mean(reference**2) - mean_ref**2, 0)
    var_est = np.maximum(local_mean(estimate**2) - mean_est**2, 0)
    covariance = local_mean(reference * estimate) - mean_ref * mean_est
    c1, c2 = 0.01**2, 0.03**2
    luminance = (2 * mean_ref * mean_est + c1) / (mean_ref**2 + mean_est**2 + c1)
    structure = (2 * covariance + c2) / (var_ref + var_est + c2)
    return float(np.mean(luminance * structure))


def out_of_bounds_fraction(image: ArrayLike) -> float:
    """Fraction of pixels strictly outside the closed unit interval."""
    image = _real_array(image, "image")
    return float(np.mean((image < 0) | (image > 1)))
