import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from labram_tsv.config import TASK_ORDER
from labram_tsv.geometry import compute_svd
from labram_tsv.io import RunRecord
from labram_tsv import sti
from labram_tsv.sti import sti_from_parts


class STITests(unittest.TestCase):
    def test_orthogonal_task_bases_have_zero_sti(self):
        identity = np.eye(4)
        left = [identity[:, :2], identity[:, 2:]]
        right = [identity[:, :2], identity[:, 2:]]
        singulars = [np.asarray([2.0, 1.0]), np.asarray([3.0, 0.5])]
        self.assertAlmostEqual(sti_from_parts(left, singulars, right), 0.0, places=12)

    def test_identical_task_bases_interfere(self):
        identity = np.eye(4)
        basis = identity[:, :2]
        value = sti_from_parts(
            [basis, basis],
            [np.asarray([2.0, 1.0]), np.asarray([3.0, 0.5])],
            [basis, basis],
        )
        self.assertGreater(value, 0.0)

    def test_rank_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            sti_from_parts(
                [np.eye(3)[:, :1], np.eye(3)[:, :2]],
                [np.ones(1), np.ones(2)],
                [np.eye(3)[:, :1], np.eye(3)[:, :2]],
            )

    def test_exact_32_combination_grid_and_random_null(self):
        runs = []
        decompositions = {}
        rng = np.random.default_rng(41)
        for task in TASK_ORDER:
            for replicate in (1, 2):
                root = Path("/unused") / task / str(replicate)
                run = RunRecord(
                    task,
                    replicate,
                    root,
                    root / "metadata.json",
                    root / "checkpoint.pth",
                    root / "delta.npz",
                    root / "success.json",
                    {"num_classes": 2},
                )
                runs.append(run)
                decompositions[(task, replicate, "L00/Q")] = compute_svd(
                    rng.standard_normal((5, 5))
                )
        with patch.object(sti, "EXPECTED_MODULE_SHAPES", {"L00/Q": (5, 5)}):
            combinations = sti.replicate_combinations(runs)
            observed = sti.observed_sti_rows(runs, decompositions)
            random_row, draws = sti.random_sti_for_module(
                "L00/Q", combinations, decompositions, 3, 123, save_draws=True
            )
        self.assertEqual(len(combinations), 32)
        self.assertEqual(len(observed), 32)
        self.assertEqual(random_row["k_paper"], 1)
        self.assertEqual(len(draws), 3)


if __name__ == "__main__":
    unittest.main()
