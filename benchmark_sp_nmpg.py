"""Run a deterministic 128×128 SP-NMPG benchmark and save its iteration log."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from cdp import AmplitudeLoss, CDPOperator, relative_error
from sp_nmpg import solve_sp_nmpg, spectral_initialization


def ellipse_phantom() -> NDArray:
    """Generate a synthetic transmission image with smooth and flat regions."""
    coordinates = (np.arange(128) + 0.5) / 64 - 1
    x, y = np.meshgrid(coordinates, coordinates)
    radius_squared = (x / 0.75) ** 2 + (y / 0.9) ** 2
    image = np.where(radius_squared <= 1, 0.55 - 0.2 * radius_squared, 0.0)
    image[((x + 0.24) / 0.17) ** 2 + ((y + 0.16) / 0.30) ** 2 <= 1] = 0.85
    image[((x - 0.27) / 0.19) ** 2 + ((y + 0.10) / 0.24) ** 2 <= 1] = 0.17
    image[((x - 0.06) / 0.10) ** 2 + ((y - 0.39) / 0.14) ** 2 <= 1] = 1.0
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("sp_nmpg_benchmark.csv"))
    args = parser.parse_args()

    truth = ellipse_phantom()
    operator = CDPOperator.random(truth.shape, 6, np.random.default_rng(25))
    observed = operator.intensity(truth)
    loss = AmplitudeLoss(operator, observed, epsilon=0.01)

    start = perf_counter()
    initial = spectral_initialization(operator, observed, np.random.default_rng(26))
    initialization_seconds = perf_counter() - start
    result = solve_sp_nmpg(loss, initial, tolerance=1e-12)

    rows = [asdict(row) for row in result.history]
    with args.log.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first, last = result.history[0], result.history[-1]
    print(f"objective: {first.objective:.6e} -> {last.objective:.6e}")
    print(f"KKT residual: {first.kkt_residual:.6e} -> {last.kkt_residual:.6e}")
    initial_error = relative_error(truth, initial)
    final_error = relative_error(truth, result.image)
    print(f"relative error: {initial_error:.6e} -> {final_error:.6e}")
    total_backtracks = sum(row.backtracks for row in result.history)
    print(f"iterations: {last.iteration}, backtracks: {total_backtracks}")
    print(
        f"objective/gradient evaluations: "
        f"{result.objective_evaluations}/{result.gradient_evaluations}"
    )
    print(f"initialization: {initialization_seconds:.3f} s, solve: {last.elapsed_seconds:.3f} s")
    print(f"stop: {result.stop_reason}")
    print(f"log: {args.log}")


if __name__ == "__main__":
    main()
