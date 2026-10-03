import unittest

import numpy as np

from optimization.change_of_variables.gradients import (
    automatic_gradient,
    chain_rule_gradient,
    comparison_error,
)


class ChangeOfVariablesTests(unittest.TestCase):
    def test_identity_matrix_leaves_base_gradient(self):
        point = np.array([0.2, -0.6, 1.1])
        gradient = chain_rule_gradient(point, np.eye(3))

        np.testing.assert_allclose(gradient, np.cos(point) + point)

    def test_square_matrix_matches_jax(self):
        rng = np.random.default_rng(4)
        point = rng.normal(size=3)
        matrix = rng.normal(size=(3, 3))

        np.testing.assert_allclose(
            chain_rule_gradient(point, matrix),
            automatic_gradient(point, matrix),
            atol=1e-5,
        )

    def test_rectangular_matrix_matches_jax(self):
        rng = np.random.default_rng(8)
        point = rng.normal(size=2)
        matrix = rng.normal(size=(4, 2))

        self.assertLess(comparison_error(point, matrix), 1e-5)

    def test_gradient_has_same_size_as_original_point(self):
        point = np.array([0.3, -0.2, 0.7])
        matrix = np.ones((2, 3))

        self.assertEqual(automatic_gradient(point, matrix).shape, point.shape)

    def test_bad_shapes_and_values_are_rejected(self):
        bad_inputs = [
            ([1.0, 2.0], np.ones((2, 3))),
            ([[1.0, 2.0]], np.eye(2)),
            ([1.0, np.nan], np.eye(2)),
            ([1.0, 2.0], [[1.0, np.inf], [0.0, 1.0]]),
        ]

        for point, matrix in bad_inputs:
            with self.subTest(point=point, matrix=matrix):
                with self.assertRaises(ValueError):
                    chain_rule_gradient(point, matrix)


if __name__ == "__main__":
    unittest.main()
