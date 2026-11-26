"""Rebuild the report figures from experiment snapshots and iteration CSVs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


BLUE = "#0072B2"
ORANGE = "#C54B12"
IMAGE_STYLE = {"cmap": "gray", "vmin": 0, "vmax": 1, "interpolation": "nearest"}
SPECTRAL_RUNS = (
    ("sp_nmpg_spectral", "SP-NMPG", BLUE, "-"),
    ("lbfgs_spectral", "Projected L-BFGS", ORANGE, "--"),
)
ABLATION_RUNS = (
    ("lbfgs_damped", "Powell damping", BLUE, "-"),
    ("lbfgs_skip", "Skip weak pair", ORANGE, "--"),
)


def _report(directory):
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def _reconstruction(directory, run):
    with np.load(directory / run / "reconstruction.npz") as data:
        return data["initial"], data["estimate"]


def _show(ax, image, title, **style):
    ax.imshow(image, **style)
    ax.set_title(title)
    ax.set_axis_off()


def _save(fig, destination):
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def _initialization_figure(directory, destination):
    report = _report(directory)
    runs = {row["name"]: row for row in report["runs"]}
    names = ["sp_nmpg_spectral", *(
        name for name in runs if name.startswith("sp_nmpg_random_")
    )]
    fig, axes = plt.subplots(2, len(names) + 1, figsize=(13, 4.3), layout="constrained")
    fig.suptitle("Case 1 | Initializations and reconstructions", fontweight="semibold")
    with np.load(directory / "inputs.npz") as data:
        _show(axes[0, 0], data["truth"], "Ground truth", **IMAGE_STYLE)
    _, estimate = _reconstruction(directory, "lbfgs_spectral")
    error = runs["lbfgs_spectral"]["relative_error"]
    _show(axes[1, 0], estimate, f"L-BFGS (spectral)\nError {error:.2e}", **IMAGE_STYLE)
    for column, name in enumerate(names, start=1):
        row = runs[name]
        seed = row["initial_seed"]
        initial, estimate = _reconstruction(directory, name)
        source = "Spectral start" if seed is None else f"Uniform seed {seed}"
        _show(axes[0, column], initial, source, **IMAGE_STYLE)
        _show(
            axes[1, column], estimate,
            f"SP-NMPG\nError {row['relative_error']:.2e}", **IMAGE_STYLE,
        )
    _save(fig, destination)


def _four_panel(directory, destination):
    report = _report(directory)
    runs = {row["name"]: row for row in report["runs"]}
    bounded = runs["lbfgs_spectral"]
    unbounded = runs["lbfgs_unconstrained"]
    _, boxed_image = _reconstruction(directory, bounded["name"])
    _, unboxed_image = _reconstruction(directory, unbounded["name"])
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.4), layout="constrained")
    case = report["case"][-1]
    fig.suptitle(
        f"Case {case} | Reconstruction at {bounded['actual_snr_db']:.2f} dB measured SNR",
        fontweight="semibold",
    )
    with np.load(directory / "inputs.npz") as data:
        _show(axes[0], data["truth"], "Ground truth", **IMAGE_STYLE)
        intensity = np.log1p(np.maximum(np.fft.fftshift(data["observed"][0]), 0))
    _show(
        axes[1], intensity, "Log diffraction intensity\nMask 1, centered",
        cmap="magma", interpolation="nearest",
    )
    _show(
        axes[2], unboxed_image, f"Unconstrained L-BFGS\nPSNR {unbounded['psnr_db']:.2f} dB",
        **IMAGE_STYLE,
    )
    _show(
        axes[3], boxed_image, f"Box-constrained L-BFGS\nPSNR {bounded['psnr_db']:.2f} dB",
        **IMAGE_STYLE,
    )
    _save(fig, destination)


def _curve_figure(directory, destination, methods, title, *, mark_curvature=False):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.75), layout="constrained")
    fig.suptitle(title, fontweight="semibold")
    metrics = (("objective", "Objective"), ("kkt_residual", "KKT residual"))
    for name, label, color, linestyle in methods:
        with (directory / name / "iterations.csv").open(encoding="utf-8", newline="") as data:
            rows = list(csv.DictReader(data))
        evaluations = [int(row["objective_evaluations"]) for row in rows]
        for ax, (key, axis_label) in zip(axes, metrics):
            values = np.array([float(row[key]) for row in rows])
            ax.plot(evaluations, values, label=label, color=color, ls=linestyle, lw=2.1)
            if mark_curvature:
                event = "damped" if name == "lbfgs_damped" else "pair_skipped"
                indices = [i for i, row in enumerate(rows) if row[event] == "True"]
                symbol = "o" if event == "damped" else "x"
                ax.scatter(
                    [evaluations[i] for i in indices], values[indices], marker=symbol,
                    color=color, s=44, linewidths=1.4, zorder=4,
                    label="Pair damped" if event == "damped" else "Pair skipped",
                )
            ax.set(xlabel="Objective evaluations", ylabel=axis_label, yscale="log")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(visible=True, which="major", color="#dce3eb", linewidth=0.8)
    axes[0].legend(loc="best", frameon=False)
    _save(fig, destination)


def _write_summary(results, groups, destination):
    metrics = (
        "name", "initialization", "actual_snr_db", "objective", "relative_error",
        "stationarity_residual", "psnr_db", "ssim", "out_of_bounds_fraction",
        "iterations", "objective_evaluations", "gradient_evaluations",
        "initialization_seconds", "solve_seconds", "seconds_to_stationarity_threshold",
        "relative_threshold_met", "kkt_threshold_met", "damping_events", "skipped_pairs",
        "direction_fallbacks", "backtracks", "stop_reason",
    )
    columns = ("group", "case", *metrics, "peak_mib", "median_gradient_ms")
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for group in groups:
            directory = results / group
            report = _report(directory)
            profile_file = directory / "profile.json"
            profile = json.loads(profile_file.read_text(encoding="utf-8")) if profile_file.exists() else None
            for run in report["runs"]:
                row = {"group": group, "case": report["case"]}
                row.update({field: run.get(field) for field in metrics})
                if profile is not None:
                    row["peak_mib"] = profile["runs"][run["name"]]["peak_mib"]
                    row["median_gradient_ms"] = profile["gradient"]["median_ms"]
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("case1", "case2", "case3", "ablation", "all"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        if args.case in ("case1", "all"):
            case = args.results / "case1"
            _initialization_figure(case, args.output / "case1_initialization.png")
            _curve_figure(
                case, args.output / "case1_convergence.png", SPECTRAL_RUNS,
                "Case 1 | Noiseless convergence from the shared spectral start",
            )
        if args.case in ("case2", "all"):
            _four_panel(args.results / "case2", args.output / "case2_reconstruction.png")
        if args.case in ("case3", "all"):
            case = args.results / "case3"
            _four_panel(case, args.output / "case3_reconstruction.png")
            _curve_figure(
                case, args.output / "case3_convergence.png", SPECTRAL_RUNS,
                "Case 3 | Convergence at 65,536 image variables",
            )
        if args.case in ("ablation", "all"):
            _curve_figure(
                args.results / "ablation", args.output / "powell_ablation.png", ABLATION_RUNS,
                "Powell damping ablation | Case 2 with a scaled spectral start",
                mark_curvature=True,
            )
    groups = ("case1", "case2", "case3", "ablation") if args.case == "all" else (args.case,)
    _write_summary(args.results, groups, args.output / "run_summary.csv")
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()
