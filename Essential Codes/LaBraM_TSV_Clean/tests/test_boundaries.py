import ast
import unittest
from pathlib import Path

from labram_tsv.config import EXPECTED_MODULE_SHAPES, N_MODULES, TASK_ORDER
from labram_tsv.spectra import SPECTRUM_NAMES


PACKAGE = Path(__file__).resolve().parents[1] / "labram_tsv"


def imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
            roots.add(node.module.split(".")[-1])
    return roots


class BoundaryTests(unittest.TestCase):
    def test_exact_experiment_grid(self):
        self.assertEqual(TASK_ORDER, ("Rest", "Motor", "P300", "SSS", "TS"))
        self.assertEqual(N_MODULES, 72)
        self.assertEqual(len(EXPECTED_MODULE_SHAPES), 72)

    def test_spectra_contract_has_exactly_six_names(self):
        self.assertEqual(
            SPECTRUM_NAMES,
            (
                "Individual Energy",
                "Individual Functional",
                "Shared Energy",
                "Shared Functional",
                "Principal Angle",
                "Overlapped Functional",
            ),
        )

    def test_spectra_does_not_import_plot_heatmap_or_sti(self):
        roots = imported_roots(PACKAGE / "spectra.py")
        self.assertFalse({"matplotlib", "heatmap", "sti"} & roots)

    def test_plotting_does_not_import_geometry_or_model(self):
        roots = imported_roots(PACKAGE / "plotting.py")
        self.assertFalse({"geometry", "model", "spectra", "heatmap", "sti"} & roots)


if __name__ == "__main__":
    unittest.main()
