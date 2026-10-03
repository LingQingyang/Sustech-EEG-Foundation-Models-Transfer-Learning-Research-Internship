import unittest

import numpy as np

from labram_tsv.geometry import (
    compute_svd,
    principal_decomposition_rank_one,
    projected_candidate_update,
    projection_fractions_rank_one,
    synthetic_self_test,
)


class GeometryTests(unittest.TestCase):
    def test_bundled_exact_self_test(self):
        result = synthetic_self_test()
        self.assertGreater(result["identical_projection_min"], 1 - 1e-12)
        self.assertLess(result["orthogonal_projection_max"], 1e-12)
        self.assertLess(result["explicit_projection_max_error"], 1e-12)
        self.assertLess(result["principal_identity_max_error"], 1e-12)

    def test_rectangular_svd_reconstruction(self):
        rng = np.random.default_rng(17)
        matrix = rng.standard_normal((9, 5))
        record = compute_svd(matrix)
        np.testing.assert_allclose(record.reconstruct_prefix(record.rank), matrix, atol=1e-12)
        self.assertLess(record.reconstruction_relative_error, 1e-12)

    def test_identical_principal_projection_reconstructs_update(self):
        rng = np.random.default_rng(23)
        left, _ = np.linalg.qr(rng.standard_normal((8, 4)), mode="reduced")
        right, _ = np.linalg.qr(rng.standard_normal((7, 4)), mode="reduced")
        singulars = np.asarray([4.0, 2.0, 1.0, 0.5])
        decomposition = principal_decomposition_rank_one(
            left, right, singulars, [(left, right)], 8 * 7
        )
        reconstructed = projected_candidate_update(
            decomposition, range(decomposition.rank), (8, 7)
        )
        expected = left @ (singulars[:, None] * right.T)
        np.testing.assert_allclose(reconstructed, expected, atol=1e-11)
        np.testing.assert_allclose(decomposition.cosines, 1.0, atol=1e-12)

    def test_projection_fractions_are_bounded(self):
        rng = np.random.default_rng(31)
        candidate_left, _ = np.linalg.qr(rng.standard_normal((11, 4)), mode="reduced")
        candidate_right, _ = np.linalg.qr(rng.standard_normal((9, 4)), mode="reduced")
        context_left, _ = np.linalg.qr(rng.standard_normal((11, 5)), mode="reduced")
        context_right, _ = np.linalg.qr(rng.standard_normal((9, 5)), mode="reduced")
        values, rank, _ = projection_fractions_rank_one(
            candidate_left,
            candidate_right,
            [(context_left, context_right)],
            11 * 9,
        )
        self.assertEqual(rank, 5)
        self.assertTrue(np.all(values >= 0))
        self.assertTrue(np.all(values <= 1))


if __name__ == "__main__":
    unittest.main()
