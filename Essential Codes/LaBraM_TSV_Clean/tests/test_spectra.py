import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from labram_tsv.config import AnalysisConfig
from labram_tsv.geometry import compute_svd
from labram_tsv.io import RunRecord
from labram_tsv import spectra


SMALL_SHAPES = {"L00/Q": (4, 4), "L00/K": (4, 4)}


def fake_run(task, replicate):
    root = Path("/unused") / task / str(replicate)
    return RunRecord(
        task=task,
        replicate=replicate,
        run_dir=root,
        metadata_path=root / "metadata.json",
        checkpoint_path=root / "checkpoint.pth",
        delta_path=root / "delta.npz",
        success_path=root / "success.json",
        metadata={"num_classes": 2},
    )


class FakeEvaluator:
    full_bacc = 0.9

    def __init__(self, full_energy=1.0):
        self.full_energy = full_energy

    @staticmethod
    def _metrics(bacc):
        return {"balanced_accuracy": bacc, "accuracy": bacc, "macro_f1": bacc}

    def evaluate_q(self, q):
        return self._metrics(0.5 + 0.4 * float(q))

    def evaluate_global_order(self, order, count):
        return self._metrics(0.5 + 0.4 * count / max(len(order), 1))

    def evaluate_updates(self, updates):
        energy = sum(float(np.sum(np.asarray(value) ** 2)) for value in updates.values())
        fraction = min(1.0, np.sqrt(energy / max(self.full_energy, 1e-12)))
        return self._metrics(0.5 + 0.4 * fraction)


class SixSpectraTests(unittest.TestCase):
    def setUp(self):
        self.candidate = fake_run("Rest", 1)
        self.context = fake_run("Motor", 1)
        diagonal = np.diag([4.0, 3.0, 2.0, 1.0])
        second = np.diag([3.5, 2.5, 1.5, 0.5])
        self.decompositions = {}
        for run in (self.candidate, self.context):
            self.decompositions[(run.task, run.replicate, "L00/Q")] = compute_svd(diagonal)
            self.decompositions[(run.task, run.replicate, "L00/K")] = compute_svd(second)
        self.analysis = replace(
            AnalysisConfig(),
            coarse_fractions=(0.0, 0.5, 1.0),
            fine_step=0.25,
            energy_vertical_step=0.25,
            bacc_vertical_step=0.20,
            shared_random_n=2,
        )
        self.full_energy = float(np.sum(diagonal**2) + np.sum(second**2))

    def test_individual_energy_and_functional_endpoints(self):
        with patch.object(spectra, "EXPECTED_MODULE_SHAPES", SMALL_SHAPES):
            energy, functional, cutoff = spectra.individual_spectra_for_run(
                self.candidate,
                self.decompositions,
                self.analysis,
                FakeEvaluator(self.full_energy),
            )
        endpoint_energy = max(energy, key=lambda row: row["q"])
        endpoint_functional = max(functional, key=lambda row: row["q"])
        self.assertAlmostEqual(endpoint_energy["individual_energy"], 1.0)
        self.assertAlmostEqual(endpoint_functional["raw_bacc"], 0.9)
        self.assertEqual(cutoff["q_star"], 1.0)

    def test_shared_pair_has_exact_functional_endpoint(self):
        cutoffs = {
            self.candidate.key: {
                "q_star": 1.0,
                "K_star": 8.0,
                "B_ind_at_qstar": 0.9,
                "B0": 0.5,
                "Bfull": 0.9,
                "q_energy_99": 1.0,
                "d_energy_99": 8.0,
            },
            self.context.key: {
                "q_star": 1.0,
                "K_star": 8.0,
                "B_ind_at_qstar": 0.9,
                "B0": 0.5,
                "Bfull": 0.9,
                "q_energy_99": 1.0,
                "d_energy_99": 8.0,
            },
        }
        with patch.object(spectra, "EXPECTED_MODULE_SHAPES", SMALL_SHAPES):
            energy, components = spectra.shared_energy_spectrum(
                self.candidate,
                [self.context],
                "single",
                cutoffs,
                self.decompositions,
                self.analysis,
            )
            functional = spectra.shared_functional_spectrum(
                self.candidate,
                [self.context],
                "single",
                components,
                energy,
                cutoffs,
                self.analysis,
                FakeEvaluator(self.full_energy),
            )
            principal = spectra.principal_angle_spectrum(
                self.candidate,
                [self.context],
                "replicate",
                cutoffs,
                self.decompositions,
                self.analysis,
            )
            overlap = spectra.overlapped_functional_spectrum(
                self.candidate,
                [self.context],
                "replicate",
                cutoffs,
                self.decompositions,
                self.analysis,
                FakeEvaluator(self.full_energy),
            )
        self.assertAlmostEqual(max(row["shared_energy_spectrum"] for row in energy), 1.0)
        self.assertAlmostEqual(
            max(functional, key=lambda row: row["component_count"])["raw_bacc"], 0.9
        )
        self.assertTrue(all(abs(row["cos_theta"] - 1.0) < 1e-10 for row in principal))
        self.assertAlmostEqual(
            max(overlap, key=lambda row: row["principal_dimensions"])[
                "functional_retention"
            ],
            1.0,
            places=10,
        )

    def test_cross_task_principal_random_reference_is_cached(self):
        cutoffs = {
            run.key: {
                "q_star": 1.0,
                "K_star": 8.0,
                "B_ind_at_qstar": 0.9,
                "B0": 0.5,
                "Bfull": 0.9,
                "q_energy_99": 1.0,
                "d_energy_99": 8.0,
            }
            for run in (self.candidate, self.context)
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            with patch.object(spectra, "EXPECTED_MODULE_SHAPES", SMALL_SHAPES):
                first = spectra.principal_angle_spectrum(
                    self.candidate,
                    [self.context],
                    "single",
                    cutoffs,
                    self.decompositions,
                    self.analysis,
                    cache,
                )
                second = spectra.principal_angle_spectrum(
                    self.candidate,
                    [self.context],
                    "single",
                    cutoffs,
                    self.decompositions,
                    self.analysis,
                    cache,
                )
            self.assertEqual(len(list(cache.glob("principal_*.npz"))), 1)
        self.assertEqual(
            [row["matched_random_upper"] for row in first],
            [row["matched_random_upper"] for row in second],
        )


if __name__ == "__main__":
    unittest.main()
