"""Two-metric box-constrained L-BFGS with optional Powell damping."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdp import AmplitudeLoss, kkt_residual, project_box


@dataclass(frozen=True)
class _CurvaturePair:
    step: NDArray
    change: NDArray
    curvature: float


def _dot(left: NDArray, right: NDArray) -> float:
    return float(np.einsum("i,i->", left, right))


def _positive_pair(step: NDArray, change: NDArray) -> _CurvaturePair | None:
    curvature = _dot(step, change)
    size = np.sqrt(_dot(step, step)) * np.sqrt(_dot(change, change))
    if np.isfinite(curvature) and curvature > 1e-10 * size:
        return _CurvaturePair(step, change, curvature)
    return None


def _restrict_pairs(
    memory: deque[_CurvaturePair], free: NDArray
) -> list[_CurvaturePair]:
    # Restriction can change the sign of a previously positive inner product.
    pairs = []
    for stored in memory:
        pair = _positive_pair(stored.step[free], stored.change[free])
        if pair is not None:
            pairs.append(pair)
    return pairs


def _inverse_scale(pairs: list[_CurvaturePair], default: float) -> float:
    if not pairs:
        return default
    last = pairs[-1]
    scale = last.curvature / _dot(last.change, last.change)
    return float(np.clip(scale, 1e-3 * default, 1e3 * default))


def _inverse_product(
    vector: NDArray, pairs: list[_CurvaturePair], inverse_scale: float
) -> NDArray:
    """Apply the inverse Hessian by the L-BFGS two-loop recursion."""
    working = vector.copy()
    coefficients = []
    for pair in reversed(pairs):
        alpha = _dot(pair.step, working) / pair.curvature
        coefficients.append(alpha)
        working -= alpha * pair.change
    working *= inverse_scale
    for pair, alpha in zip(pairs, reversed(coefficients)):
        beta = _dot(pair.change, working) / pair.curvature
        working += (alpha - beta) * pair.step
    return working


def _hessian_product(
    vector: NDArray, pairs: list[_CurvaturePair], inverse_scale: float
) -> NDArray:
    """Apply the matching direct Hessian for Powell's B s without forming B."""
    # Each BFGS update adds one rank-one term and subtracts another.
    update_vectors: list[tuple[NDArray, NDArray]] = []

    def product(v: NDArray) -> NDArray:
        result = v / inverse_scale
        for subtract, add in update_vectors:
            result -= _dot(subtract, v) * subtract
            result += _dot(add, v) * add
        return result

    for pair in pairs:
        b_step = product(pair.step)
        model_curvature = _dot(pair.step, b_step)
        if not np.isfinite(model_curvature) or model_curvature <= 0:
            raise FloatingPointError("BFGS model lost positive curvature")
        update_vectors.append(
            (b_step / np.sqrt(model_curvature), pair.change / np.sqrt(pair.curvature))
        )
    return product(vector)


def _free_coordinates(image: NDArray, gradient: NDArray, diagonal: float) -> NDArray:
    distance = np.max(np.abs(image - project_box(image - diagonal * gradient)))
    margin = min(1e-3, distance)
    near_lower = (image <= margin) & (gradient > 0)
    near_upper = (image >= 1 - margin) & (gradient < 0)
    return ~(near_lower | near_upper)


@dataclass(frozen=True)
class _PairUpdate:
    pair: _CurvaturePair | None = None
    raw_curvature: float | None = None
    model_curvature: float | None = None
    damped: bool = False


def _update_pair(
    step: NDArray,
    change: NDArray,
    pairs: list[_CurvaturePair],
    diagonal: float,
    damping: bool,
) -> _PairUpdate:
    raw_curvature = _dot(step, change)
    if not np.any(step):
        return _PairUpdate(raw_curvature=raw_curvature)
    try:
        b_step = _hessian_product(step, pairs, _inverse_scale(pairs, diagonal))
    except FloatingPointError:
        return _PairUpdate(raw_curvature=raw_curvature)
    model_curvature = _dot(step, b_step)
    if not np.isfinite(model_curvature) or model_curvature <= 0:
        return _PairUpdate(raw_curvature=raw_curvature, model_curvature=model_curvature)
    damped = raw_curvature < 0.2 * model_curvature
    if damped:
        if not damping:
            return _PairUpdate(raw_curvature=raw_curvature, model_curvature=model_curvature)
        weight = 0.8 * model_curvature / (model_curvature - raw_curvature)
        change = weight * change + (1 - weight) * b_step
    return _PairUpdate(_positive_pair(step, change), raw_curvature, model_curvature, damped)


@dataclass(frozen=True)
class _LineSearch:
    image: NDArray | None
    objective: float
    step_size: float
    backtracks: int
    evaluations: int
    budget_exhausted: bool = False


def _projected_armijo(
    loss: AmplitudeLoss,
    image: NDArray,
    shape: tuple[int, ...],
    value: float,
    gradient: NDArray,
    direction: NDArray,
    project: Callable[[NDArray], NDArray],
    remaining_evaluations: int | None,
) -> _LineSearch:
    evaluations = 0
    step_size = 1.0
    for backtracks in range(41):
        if remaining_evaluations == evaluations:
            return _LineSearch(None, value, 0.0, backtracks, evaluations, True)
        trial = project(image + step_size * direction)
        slope = _dot(gradient, trial - image)
        if slope < 0:
            trial_value = loss.value(trial.reshape(shape))
            evaluations += 1
            if np.isfinite(trial_value) and trial_value <= value + 1e-4 * slope:
                return _LineSearch(trial, trial_value, step_size, backtracks, evaluations)
        step_size *= 0.5
    exhausted = remaining_evaluations == evaluations
    return _LineSearch(None, value, 0.0, 40, evaluations, exhausted)


@dataclass(frozen=True)
class LBFGSIteration:
    iteration: int
    objective: float
    kkt_residual: float
    step_size: float
    backtracks: int
    objective_evaluations: int
    gradient_evaluations: int
    elapsed_seconds: float
    active_variables: int
    memory_size: int = 0
    damped: bool = False
    pair_skipped: bool = False
    fallback: Literal["non_descent", "line_search_failed"] | None = None
    raw_curvature: float | None = None
    model_curvature: float | None = None
    stored_curvature: float | None = None


@dataclass(frozen=True)
class ProjectedLBFGSResult:
    image: NDArray
    history: tuple[LBFGSIteration, ...]
    stop_reason: Literal[
        "kkt_tolerance", "max_iterations", "evaluation_budget", "line_search_failed"
    ]
    objective_evaluations: int
    gradient_evaluations: int
    backtracks: int
    direction_fallbacks: int


def solve_projected_lbfgs(
    loss: AmplitudeLoss,
    initial: ArrayLike,
    *,
    damping: bool = True,
    box_constrained: bool = True,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
    max_objective_evaluations: int | None = None,
) -> ProjectedLBFGSResult:
    """Minimize a real loss on [0, 1] with a ten-pair L-BFGS memory.

    The active coordinates use a diagonal inverse metric, and the free block
    uses reduced L-BFGS. New secants use coordinates free at both endpoints.
    Apply Powell damping at s·y < 0.2 s·B s, or skip that pair with damping off.
    An unusable quasi-Newton direction is retried with the diagonal metric.
    The log reports active coordinates and stored pairs after each accepted step.
    The unconstrained comparison uses the same updates with identity projection
    and records the infinity norm of the gradient as its stationarity residual.

    Two-metric projection: https://web.mit.edu/people/dimitrib/ProjectedNewton.pdf
    Limited-memory BFGS: https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf
    """
    if not np.isfinite(tolerance) or tolerance < 0 or max_iterations < 0:
        raise ValueError("tolerance and max_iterations must be nonnegative")
    if max_objective_evaluations is not None and max_objective_evaluations < 1:
        raise ValueError("the objective evaluation budget must allow the initial evaluation")
    if np.iscomplexobj(initial) or not np.all(np.isfinite(initial)):
        raise ValueError("initial image must be real and finite")

    project = project_box if box_constrained else np.asarray

    def stationarity(image: NDArray, gradient: NDArray) -> float:
        if box_constrained:
            return kkt_residual(image, gradient)
        return float(np.max(np.abs(gradient)))

    start = perf_counter()
    initial_image = project(np.asarray(initial, dtype=np.float64))
    shape = initial_image.shape
    image = initial_image.ravel()
    value, gradient = loss.value_and_gradient(initial_image)
    gradient = gradient.ravel()
    residual = stationarity(image, gradient)
    diagonal = float(image.size)
    free = _free_coordinates(image, gradient, diagonal) if box_constrained else np.ones(
        image.shape, dtype=bool
    )
    memory: deque[_CurvaturePair] = deque(maxlen=10)
    evaluations = gradient_evaluations = 1
    total_backtracks = direction_fallbacks = 0
    history = [
        LBFGSIteration(
            iteration=0,
            objective=value,
            kkt_residual=residual,
            step_size=0.0,
            backtracks=0,
            objective_evaluations=1,
            gradient_evaluations=1,
            elapsed_seconds=perf_counter() - start,
            active_variables=image.size - np.count_nonzero(free),
        )
    ]
    stop_reason = "max_iterations"

    for iteration in range(1, max_iterations + 1):
        if residual <= tolerance:
            break
        pairs = _restrict_pairs(memory, free)
        diagonal_direction = -diagonal * gradient
        direction = diagonal_direction.copy()
        fallback = None
        if pairs:
            direction[free] = -_inverse_product(
                gradient[free], pairs, _inverse_scale(pairs, diagonal)
            )
            if not np.all(np.isfinite(direction)) or (
                _dot(gradient, project(image + direction) - image) >= 0
            ):
                direction = diagonal_direction
                fallback = "non_descent"

        remaining = (
            None if max_objective_evaluations is None else max_objective_evaluations - evaluations
        )
        search = _projected_armijo(loss, image, shape, value, gradient, direction, project, remaining)
        evaluations += search.evaluations
        backtracks = search.backtracks
        if search.image is None and pairs and fallback is None and not search.budget_exhausted:
            fallback = "line_search_failed"
            if remaining is not None:
                remaining -= search.evaluations
            search = _projected_armijo(
                loss, image, shape, value, gradient, diagonal_direction, project, remaining
            )
            evaluations += search.evaluations
            backtracks += search.backtracks
        total_backtracks += backtracks
        direction_fallbacks += int(fallback is not None)
        if search.image is None:
            stop_reason = "evaluation_budget" if search.budget_exhausted else "line_search_failed"
            break

        next_image = search.image
        next_gradient = loss.gradient(next_image.reshape(shape)).ravel()
        gradient_evaluations += 1
        residual = stationarity(next_image, next_gradient)
        next_free = _free_coordinates(next_image, next_gradient, diagonal) if box_constrained else free
        stable_free = free & next_free
        update_pairs = (
            pairs if np.array_equal(stable_free, free) else _restrict_pairs(memory, stable_free)
        )
        update = _update_pair(
            next_image[stable_free] - image[stable_free],
            next_gradient[stable_free] - gradient[stable_free],
            update_pairs,
            diagonal,
            damping,
        )
        stored_curvature = None
        if update.pair is not None:
            stored_step = np.zeros_like(image)
            stored_change = np.zeros_like(gradient)
            stored_step[stable_free] = update.pair.step
            stored_change[stable_free] = update.pair.change
            stored_curvature = update.pair.curvature
            memory.append(_CurvaturePair(stored_step, stored_change, stored_curvature))

        history.append(
            LBFGSIteration(
                iteration=iteration,
                objective=search.objective,
                kkt_residual=residual,
                step_size=search.step_size,
                backtracks=backtracks,
                objective_evaluations=evaluations,
                gradient_evaluations=gradient_evaluations,
                elapsed_seconds=perf_counter() - start,
                active_variables=image.size - np.count_nonzero(next_free),
                memory_size=len(memory),
                damped=update.damped,
                pair_skipped=update.pair is None,
                fallback=fallback,
                raw_curvature=update.raw_curvature,
                model_curvature=update.model_curvature,
                stored_curvature=stored_curvature,
            )
        )
        image, gradient, value, free = next_image, next_gradient, search.objective, next_free

    if residual <= tolerance:
        stop_reason = "kkt_tolerance"
    return ProjectedLBFGSResult(
        image.reshape(shape), tuple(history), stop_reason, evaluations, gradient_evaluations,
        total_backtracks, direction_fallbacks,
    )
