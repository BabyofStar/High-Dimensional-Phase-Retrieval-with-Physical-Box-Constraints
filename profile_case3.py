"""Measure process peak memory and standalone gradient time for a saved Case 3."""

import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from experiments import CaseSettings, RunSpec, SolverSettings, prepare_case, solve_run


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get_process = kernel.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = kernel.K32GetProcessMemoryInfo
        get_memory.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD
        )
        get_memory.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize

    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum) if sys.platform == "darwin" else int(maximum) * 1024


def _measure_gradient(prepared) -> dict:
    warmups, repeats = 3, 20
    for _ in range(warmups):
        prepared.loss.gradient(prepared.spectral_initial)
    timings_ms = []
    for _ in range(repeats):
        start = perf_counter_ns()
        prepared.loss.gradient(prepared.spectral_initial)
        timings_ms.append((perf_counter_ns() - start) / 1e6)
    q25, median, q75 = np.percentile(timings_ms, [25, 50, 75])
    return {
        "warmups": warmups,
        "repeats": repeats,
        "median_ms": float(median),
        "q25_ms": float(q25),
        "q75_ms": float(q75),
        "samples_ms": timings_ms,
    }


def _worker(directory: Path, report: dict, worker: str) -> dict:
    solver = SolverSettings(**report["solver_settings"])
    prepared = prepare_case(CaseSettings(**report["case_settings"]), solver)
    if worker == "_gradient":
        return _measure_gradient(prepared)

    entry = next(row for row in report["runs"] if row["name"] == worker)
    spec = RunSpec(
        worker, entry["algorithm"], box_constrained=entry["box_constrained"], damping=entry["damping"]
    )
    if entry["initialization"] != "spectral":
        raise ValueError("Case 3 profiling expects the common spectral start")
    initialization_peak = peak_rss_bytes()
    result = solve_run(prepared.loss, prepared.spectral_initial, spec, solver)
    peak = peak_rss_bytes()
    with np.load(directory / worker / "reconstruction.npz") as saved:
        np.testing.assert_array_equal(saved["initial"], prepared.spectral_initial)
        np.testing.assert_array_equal(saved["estimate"], result.image)
    return {
        "peak_mib": peak / 2**20,
        "peak_at_initialization_mib": initialization_peak / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.directory.resolve()
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report["case"] != "case3":
        parser.error("directory must contain the Case 3 result")

    if args.worker is not None:
        print(json.dumps(_worker(directory, report, args.worker), allow_nan=False))
        return

    def isolated(worker: str) -> dict:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), str(directory), "--worker", worker]
        return json.loads(subprocess.check_output(command, text=True))

    profile = {
        "peak_measurement": "isolated process resident memory high-water mark",
        "runs": {row["name"]: isolated(row["name"]) for row in report["runs"]},
        "gradient": isolated("_gradient"),
    }
    output = directory / "profile.json"
    output.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Case 3 profile saved to {output}")


if __name__ == "__main__":
    main()
