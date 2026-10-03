import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from labram_tsv import heatmap
from labram_tsv.io import RunRecord


class HeatmapTests(unittest.TestCase):
    def test_relative_update_energy_and_linear_median(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for replicate, scale in ((1, 1.0), (2, 3.0)):
                path = root / f"rep{replicate}.npz"
                np.savez(path, L00__Q=np.eye(2) * scale)
                runs.append(
                    RunRecord(
                        "Rest",
                        replicate,
                        root,
                        root / "meta.json",
                        root / "checkpoint.pth",
                        path,
                        root / "success.json",
                        {},
                    )
                )
            with patch.object(heatmap, "EXPECTED_MODULE_SHAPES", {"L00/Q": (2, 2)}):
                rows = heatmap.compute_heatmap_rows(runs, {"L00/Q": np.eye(2)})
                summary = heatmap.summarize_replicates(rows)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["relative_update_energy"], 1.0)
        self.assertAlmostEqual(rows[1]["relative_update_energy"], 9.0)
        self.assertAlmostEqual(summary[0]["relative_update_energy_median"], 5.0)
        self.assertAlmostEqual(
            summary[0]["log10_median_relative_update_energy"], np.log10(5.0)
        )


if __name__ == "__main__":
    unittest.main()
