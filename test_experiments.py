"""Check the fixed acquisition and saved comparison inputs."""

import json
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments import DEFAULT_PROTOCOL, CaseSettings, SolverSettings, prepare_case, run_group


class ExperimentProtocolTests(unittest.TestCase):
    def test_saved_comparisons_share_inputs_and_reproduce_seeded_data(self):
        with DEFAULT_PROTOCOL.open("rb") as stream:
            config = tomllib.load(stream)
        case_name = config["ablation"]["case"]
        case = CaseSettings(**config["cases"][case_name])
        solver = replace(SolverSettings(**config["solver"]), max_iterations=1)
        first = prepare_case(case, solver)
        repeat = prepare_case(case, solver)
        np.testing.assert_array_equal(first.loss.operator.phase_masks, repeat.loss.operator.phase_masks)
        np.testing.assert_array_equal(first.observed, repeat.observed)
        np.testing.assert_array_equal(first.spectral_initial, repeat.spectral_initial)
        self.assertLess(abs(first.actual_snr_db - case.target_snr_db), 0.2)

        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            ordinary, ablation = root / case_name, root / "ablation"
            run_group(ordinary, case_name, case, solver, first)
            scale = config["ablation"]["initial_scale"]
            rows = run_group(ablation, case_name, case, solver, repeat, ablation_initial_scale=scale)

            def report(path):
                return json.loads((path / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(report(ordinary)["input_sha256"], report(ablation)["input_sha256"])
            self.assertEqual(report(ordinary)["solver_settings"], report(ablation)["solver_settings"])
            for row in report(ordinary)["runs"]:
                with np.load(ordinary / row["name"] / "reconstruction.npz") as data:
                    np.testing.assert_array_equal(data["initial"], first.spectral_initial)
            self.assertEqual([row["damping"] for row in rows], [True, False])
            for row in rows:
                with np.load(ablation / row["name"] / "reconstruction.npz") as data:
                    np.testing.assert_array_equal(data["initial"], scale * repeat.spectral_initial)


if __name__ == "__main__":
    unittest.main()
