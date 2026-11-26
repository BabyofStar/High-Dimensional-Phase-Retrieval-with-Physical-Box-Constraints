"""Run the fixed CDP cases and save the data, reconstructions, and iteration logs."""

import argparse
import csv
import hashlib
import json
import platform
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import PIL
from numpy.typing import NDArray
from PIL import Image

from cdp import (
    AmplitudeLoss,
    CDPOperator,
    add_intensity_noise,
    out_of_bounds_fraction,
    project_box,
    psnr_db,
    relative_error,
    snr_db,
    ssim,
)
from projected_lbfgs import ProjectedLBFGSResult, solve_projected_lbfgs
from sp_nmpg import solve_sp_nmpg, spectral_initialization


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "experiment_protocol.toml"


@dataclass(frozen=True)
class SolverSettings:
    epsilon: float
    tolerance: float
    max_iterations: int
    max_objective_evaluations: int
    kkt_timing_threshold: float


@dataclass(frozen=True)
class CaseSettings:
    image: str
    image_sha256: str
    size: int
    mask_count: int
    mask_seed: int
    spectral_seed: int
    noise: Literal["none", "poisson", "gaussian"]
    noise_seed: int | None = None
    target_snr_db: float | None = None
    random_initial_seeds: tuple[int, ...] = ()
    compare_unconstrained: bool = False
    report_relative_threshold: float | None = None
    report_kkt_threshold: float | None = None


@dataclass(frozen=True)
class PreparedCase:
    truth: NDArray
    clean: NDArray
    observed: NDArray
    loss: AmplitudeLoss
    spectral_initial: NDArray
    spectral_seconds: float
    actual_snr_db: float


@dataclass(frozen=True)
class RunSpec:
    name: str
    algorithm: Literal["sp_nmpg", "lbfgs"]
    initial_seed: int | None = None
    initial_scale: float = 1.0
    box_constrained: bool = True
    damping: bool = True


def _load_image(case: CaseSettings) -> NDArray:
    path = PROJECT_ROOT / "data" / case.image
    if hashlib.sha256(path.read_bytes()).hexdigest() != case.image_sha256:
        raise ValueError(f"source image checksum differs from protocol: {path}")
    with Image.open(path) as source:
        pixels = np.asarray(source, dtype=np.float64) / 255
    if pixels.ndim == 3:
        pixels = pixels @ np.array([0.2125, 0.7154, 0.0721])
    if pixels.ndim != 2 or pixels.shape[0] != pixels.shape[1] or case.size < 1:
        raise ValueError("protocol requires a square source image and a positive target size")

    # Average pixel area over each output footprint, even for 400 -> 128.
    source_edges = np.arange(pixels.shape[0], dtype=np.float64)
    target_edges = np.linspace(0, pixels.shape[0], case.size + 1)
    left = np.maximum(target_edges[:-1, None], source_edges)
    right = np.minimum(target_edges[1:, None], source_edges + 1)
    weights = np.maximum(right - left, 0) / np.diff(target_edges)[:, None]
    return project_box(weights @ pixels @ weights.T)


def prepare_case(case: CaseSettings, solver: SolverSettings) -> PreparedCase:
    truth = _load_image(case)
    operator = CDPOperator.random(truth.shape, case.mask_count, np.random.default_rng(case.mask_seed))
    clean = operator.intensity(truth)
    observed = clean
    if case.noise != "none":
        if case.noise_seed is None or case.target_snr_db is None:
            raise ValueError("a noisy case requires noise_seed and target_snr_db")
        observed = add_intensity_noise(
            clean, model=case.noise, target_snr_db=case.target_snr_db,
            rng=np.random.default_rng(case.noise_seed),
        )
    loss = AmplitudeLoss(operator, observed, epsilon=solver.epsilon)
    start = perf_counter()
    spectral_initial = spectral_initialization(
        operator, observed, np.random.default_rng(case.spectral_seed)
    )
    spectral_seconds = perf_counter() - start
    return PreparedCase(
        truth, clean, observed, loss, spectral_initial, spectral_seconds, snr_db(clean, observed)
    )


def _run_specs(case: CaseSettings, ablation_initial_scale: float | None) -> list[RunSpec]:
    if ablation_initial_scale is not None:
        if not 0 < ablation_initial_scale <= 1:
            raise ValueError("ablation initial scale must be in (0, 1]")
        return [
            RunSpec("lbfgs_damped", "lbfgs", initial_scale=ablation_initial_scale),
            RunSpec("lbfgs_skip", "lbfgs", initial_scale=ablation_initial_scale, damping=False),
        ]
    specs = [RunSpec("sp_nmpg_spectral", "sp_nmpg"), RunSpec("lbfgs_spectral", "lbfgs")]
    specs.extend(
        RunSpec(f"sp_nmpg_random_{seed}", "sp_nmpg", initial_seed=seed)
        for seed in case.random_initial_seeds
    )
    if case.compare_unconstrained:
        specs.append(RunSpec("lbfgs_unconstrained", "lbfgs", box_constrained=False))
    return specs


def _json_number(value: float) -> float | str | None:
    if np.isnan(value):
        return None
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def solve_run(
    loss: AmplitudeLoss, initial: NDArray, spec: RunSpec, solver: SolverSettings
):
    kwargs = dict(
        tolerance=solver.tolerance,
        max_iterations=solver.max_iterations,
        max_objective_evaluations=solver.max_objective_evaluations,
    )
    if spec.algorithm == "sp_nmpg":
        return solve_sp_nmpg(loss, initial, **kwargs)
    return solve_projected_lbfgs(
        loss, initial, damping=spec.damping,
        box_constrained=spec.box_constrained, **kwargs,
    )


def _run(
    directory: Path,
    spec: RunSpec,
    prepared: PreparedCase,
    case: CaseSettings,
    solver: SolverSettings,
) -> dict:
    run_dir = directory / spec.name
    run_dir.mkdir()
    initial = prepared.spectral_initial
    initialization_seconds = prepared.spectral_seconds
    initialization = "spectral"
    if spec.initial_seed is not None:
        start = perf_counter()
        initial = np.random.default_rng(spec.initial_seed).random(prepared.truth.shape)
        initialization_seconds = perf_counter() - start
        initialization = "uniform"
    elif spec.initial_scale != 1:
        start = perf_counter()
        initial = spec.initial_scale * initial
        initialization_seconds += perf_counter() - start
        initialization = "spectral_scaled"

    start = perf_counter()
    result = solve_run(prepared.loss, initial, spec, solver)
    solve_seconds = perf_counter() - start
    first, last = result.history[0], result.history[-1]
    time_to_threshold = next(
        (initialization_seconds + row.elapsed_seconds for row in result.history
         if row.kkt_residual < solver.kkt_timing_threshold),
        None,
    )
    error = relative_error(prepared.truth, result.image)
    summary = {
        **asdict(spec),
        "initialization": initialization,
        "stationarity": "box_projected_gradient_inf" if spec.box_constrained else "gradient_inf",
        "actual_snr_db": _json_number(prepared.actual_snr_db),
        "initialization_seconds": initialization_seconds,
        "solve_seconds": solve_seconds,
        "seconds_to_stationarity_threshold": time_to_threshold,
        "iterations": last.iteration,
        "initial_objective": first.objective,
        "initial_stationarity_residual": first.kkt_residual,
        "initial_relative_error": relative_error(prepared.truth, initial),
        "objective": last.objective,
        "stationarity_residual": last.kkt_residual,
        "relative_error": error,
        "psnr_db": _json_number(psnr_db(prepared.truth, result.image)),
        "ssim": ssim(prepared.truth, result.image),
        "out_of_bounds_fraction": out_of_bounds_fraction(result.image),
        "objective_evaluations": result.objective_evaluations,
        "gradient_evaluations": result.gradient_evaluations,
        "backtracks": result.backtracks,
        "stop_reason": result.stop_reason,
    }
    if isinstance(result, ProjectedLBFGSResult):
        summary.update(
            damping_events=sum(row.damped for row in result.history),
            skipped_pairs=sum(row.pair_skipped for row in result.history),
            direction_fallbacks=result.direction_fallbacks,
        )
    if case.report_relative_threshold is not None:
        summary["relative_threshold_met"] = error < case.report_relative_threshold
    if case.report_kkt_threshold is not None:
        summary["kkt_threshold_met"] = last.kkt_residual < case.report_kkt_threshold

    rows = [asdict(row) for row in result.history]
    with (run_dir / "iterations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(run_dir / "reconstruction.npz", initial=initial, estimate=result.image)
    return summary


def run_group(
    directory: Path,
    case_name: str,
    case: CaseSettings,
    solver: SolverSettings,
    prepared: PreparedCase,
    *,
    ablation_initial_scale: float | None = None,
) -> list[dict]:
    specs = _run_specs(case, ablation_initial_scale)
    directory.mkdir(parents=True)
    arrays = {
        "truth": prepared.truth, "clean": prepared.clean, "observed": prepared.observed,
        "spectral_initial": prepared.spectral_initial,
        "masks": prepared.loss.operator.phase_masks,
    }
    np.savez(directory / "inputs.npz", **arrays)
    rows = []
    for spec in specs:
        summary = _run(directory, spec, prepared, case, solver)
        rows.append(summary)
        print(
            f"{directory.name}/{spec.name}: {summary['stop_reason']}, "
            f"iterations={summary['iterations']}, "
            f"stationarity={summary['stationarity_residual']:.3e}",
            flush=True,
        )
    report = {
        "case": case_name,
        "ablation": ablation_initial_scale is not None,
        "case_settings": asdict(case),
        "solver_settings": asdict(solver),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "platform": platform.platform(),
        },
        "input_sha256": {
            name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in arrays.items()
        },
        "runs": rows,
    }
    (directory / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("case1", "case2", "case3", "ablation", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    solver = SolverSettings(**config["solver"])
    ablation_case = config["ablation"]["case"]
    groups = (("case1", "case2", "case3", "ablation") if args.case == "all" else (args.case,))
    required = {ablation_case if group == "ablation" else group for group in groups}
    if missing := required - config["cases"].keys():
        parser.error(f"no settings for {', '.join(sorted(missing))}")
    if any((args.output / group).exists() for group in groups):
        parser.error("output already contains a selected group; choose a new output path")

    for name, parameters in config["cases"].items():
        is_regular = name in groups
        is_ablation = "ablation" in groups and name == ablation_case
        if not (is_regular or is_ablation):
            continue
        case = CaseSettings(**parameters)
        prepared = prepare_case(case, solver)
        if is_regular:
            run_group(args.output / name, name, case, solver, prepared)
        if is_ablation:
            run_group(
                args.output / "ablation", name, case, solver, prepared,
                ablation_initial_scale=config["ablation"]["initial_scale"],
            )


if __name__ == "__main__":
    main()
