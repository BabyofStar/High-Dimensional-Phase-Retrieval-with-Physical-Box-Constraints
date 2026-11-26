"""Spectral initialization and nonmonotone projected gradient for CDP."""

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdp import AmplitudeLoss, CDPOperator, kkt_residual, project_box


def spectral_initialization(
    operator: CDPOperator, intensity: ArrayLike, rng: np.random.Generator
) -> NDArray:
    """Take 30 real power iterations with weights truncated at 9× mean intensity.

    The weights use the upper truncation of Chen and Candès,
    https://arxiv.org/abs/1505.05114. The real operator and energy scaling use
    the unitary FFT and unit-modulus masks of the CDP model.
    """
    observed = np.asarray(intensity)
    if np.iscomplexobj(observed) or observed.shape != operator.measurement_shape:
        raise ValueError(f"intensity must be real with shape {operator.measurement_shape}")
    if not np.all(np.isfinite(observed)):
        raise ValueError("intensity must be finite")

    observed = np.maximum(observed, 0.0)
    mean_intensity = float(np.mean(observed))
    if mean_intensity == 0:
        return np.zeros(operator.image_shape)

    # A positive scalar multiple of the spectral weights has the same eigenvectors.
    weights = np.where(observed <= 9 * mean_intensity, observed, 0.0) / mean_intensity
    if not np.any(weights):
        # Keep the spectral matrix informative when every positive count is rejected.
        weights = observed / mean_intensity

    direction = rng.standard_normal(operator.image_shape)
    direction /= np.linalg.norm(direction)
    for _ in range(30):
        direction = operator.adjoint(weights * operator.forward(direction)).real
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise FloatingPointError("spectral power iteration reached a zero vector")
        direction /= norm

    # Resolve the sign against the nonnegative orthant before projection.
    if np.linalg.norm(np.maximum(-direction, 0)) > np.linalg.norm(np.maximum(direction, 0)):
        direction = -direction
    energy_scale = np.sqrt(direction.size * mean_intensity)
    return project_box(energy_scale * direction)


@dataclass(frozen=True)
class IterationLog:
    iteration: int
    objective: float
    kkt_residual: float
    bb_step: float
    step_size: float
    backtracks: int
    objective_evaluations: int
    gradient_evaluations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SPNMPGResult:
    image: NDArray
    history: tuple[IterationLog, ...]
    stop_reason: Literal[
        "kkt_tolerance", "max_iterations", "evaluation_budget", "line_search_failed"
    ]
    objective_evaluations: int
    gradient_evaluations: int
    backtracks: int = 0


def solve_sp_nmpg(
    loss: AmplitudeLoss,
    initial: ArrayLike,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
    max_objective_evaluations: int | None = None,
) -> SPNMPGResult:
    """Minimize the CDP loss on the unit box with BB1 and SPG2 backtracking.

    The Armijo test uses the maximum of the last five accepted objectives.
    Each trial is a convex combination of the current point and its BB
    projected step. The logged step size is the BB step times the accepted
    line-search multiplier. The method follows Birgin, Martínez and Raydan,
    https://pages.cs.wisc.edu/~swright/726/handouts/SJE001196.pdf.
    """
    if not np.isfinite(tolerance) or tolerance < 0 or max_iterations < 0:
        raise ValueError("tolerance and max_iterations must be nonnegative")
    if max_objective_evaluations is not None and max_objective_evaluations < 1:
        raise ValueError("the objective evaluation budget must allow the initial evaluation")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial image must be finite")

    start = perf_counter()
    image = project_box(initial)
    value, gradient = loss.value_and_gradient(image)
    objective_evaluations = gradient_evaluations = 1
    residual = kkt_residual(image, gradient)
    history = [IterationLog(0, value, residual, 0.0, 0.0, 0, 1, 1, perf_counter() - start)]
    window = deque([value], maxlen=5)

    # The objective averages Ln measurements and A* A = L I.
    initial_step = float(image.size)
    min_step = 1e-3 * initial_step
    max_step = 1e3 * initial_step
    bb_step = initial_step
    armijo = 1e-4
    stop_reason = "max_iterations"
    total_backtracks = 0

    for iteration in range(1, max_iterations + 1):
        if residual <= tolerance:
            break
        direction = project_box(image - bb_step * gradient) - image
        slope = float(np.sum(gradient * direction))
        reference = max(window)
        multiplier = 1.0

        for backtracks in range(41):
            if objective_evaluations == max_objective_evaluations:
                stop_reason = "evaluation_budget"
                break
            candidate = image + multiplier * direction
            candidate_value = loss.value(candidate)
            objective_evaluations += 1
            sufficient_decrease = reference + armijo * multiplier * slope
            if np.isfinite(candidate_value) and candidate_value <= sufficient_decrease:
                break
            multiplier *= 0.5
        else:
            stop_reason = "line_search_failed"

        total_backtracks += backtracks
        if stop_reason != "max_iterations":
            break

        candidate_gradient = loss.gradient(candidate)
        gradient_evaluations += 1
        residual = kkt_residual(candidate, candidate_gradient)
        history.append(
            IterationLog(
                iteration,
                candidate_value,
                residual,
                bb_step,
                multiplier * bb_step,
                backtracks,
                objective_evaluations,
                gradient_evaluations,
                perf_counter() - start,
            )
        )
        window.append(candidate_value)

        step = candidate - image
        gradient_change = candidate_gradient - gradient
        curvature = float(np.sum(step * gradient_change))
        bb_step = initial_step
        if curvature > 0:
            bb_step = float(np.clip(np.sum(step**2) / curvature, min_step, max_step))
        image, gradient = candidate, candidate_gradient

    if residual <= tolerance:
        stop_reason = "kkt_tolerance"
    return SPNMPGResult(
        image, tuple(history), stop_reason, objective_evaluations, gradient_evaluations,
        total_backtracks,
    )
