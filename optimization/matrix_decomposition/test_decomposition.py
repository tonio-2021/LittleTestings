import unittest

import numpy as np

from optimization.matrix_decomposition.decomposition import (
    decompose,
    quadratic_value,
)


class MatrixDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array(
            [
                [2, -1, 4],
                [3, 0, 5],
                [-2, 1, 7],
            ],
            dtype=float,
        )

    def test_parts_reconstruct_the_matrix(self):
        symmetric, antisymmetric = decompose(self.matrix)

        np.testing.assert_allclose(symmetric + antisymmetric, self.matrix)

    def test_parts_have_the_right_transposes(self):
        symmetric, antisymmetric = decompose(self.matrix)

        np.testing.assert_allclose(symmetric.T, symmetric)
        np.testing.assert_allclose(antisymmetric.T, -antisymmetric)

    def test_symmetric_matrix_has_no_antisymmetric_part(self):
        matrix = np.array([[2, -1], [-1, 3]], dtype=float)
        symmetric, antisymmetric = decompose(matrix)

        np.testing.assert_allclose(symmetric, matrix)
        np.testing.assert_allclose(antisymmetric, np.zeros((2, 2)))

    def test_antisymmetric_quadratic_value_is_zero(self):
        _, antisymmetric = decompose(self.matrix)

        for vector in ([1, 0, 0], [1, -2, 0.5], [-3, 4, 2]):
            with self.subTest(vector=vector):
                self.assertAlmostEqual(quadratic_value(antisymmetric, vector), 0.0)

    def test_one_by_one_matrix(self):
        symmetric, antisymmetric = decompose([[5]])

        np.testing.assert_allclose(symmetric, [[5]])
        np.testing.assert_allclose(antisymmetric, [[0]])

    def test_bad_shapes_and_values_are_rejected(self):
        with self.assertRaises(ValueError):
            decompose([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            decompose([[1, np.nan], [2, 3]])
        with self.assertRaises(ValueError):
            quadratic_value(np.eye(2), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
