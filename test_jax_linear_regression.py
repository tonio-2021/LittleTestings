import unittest

import jax
import jax.numpy as jnp

from jax_linear_regression import direct_fit, fit_with_jax, squared_error


class JaxLinearRegressionTests(unittest.TestCase):
    def test_jax_gradient_on_two_points(self):
        x = jnp.array([0.0, 1.0])
        y = jnp.array([1.0, 3.0])
        gradient = jax.grad(squared_error)(jnp.array([0.0, 0.0]), x, y)

        self.assertAlmostEqual(float(gradient[0]), -3.0)
        self.assertAlmostEqual(float(gradient[1]), -4.0)

    def test_learning_gets_close_to_direct_fit(self):
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = jnp.array([-2.9, -1.1, 1.0, 3.1, 4.9])

        learned, losses = fit_with_jax(x, y)
        direct = direct_fit(x, y)

        self.assertEqual(len(losses), 81)
        self.assertLess(losses[-1], losses[0])
        self.assertLess(float(jnp.max(jnp.abs(learned - direct))), 0.001)

    def test_direct_fit_on_an_exact_line(self):
        slope, intercept = direct_fit([0, 1, 2], [1, 3, 5])

        self.assertAlmostEqual(float(slope), 2.0)
        self.assertAlmostEqual(float(intercept), 1.0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            fit_with_jax([0, 1], [1])
        with self.assertRaises(ValueError):
            direct_fit([1, 1], [2, 3])
        with self.assertRaises(ValueError):
            fit_with_jax([0, 1], [1, 3], step_size=0)
        with self.assertRaises(ValueError):
            fit_with_jax([0, 1], [1, 3], steps=-1)


if __name__ == "__main__":
    unittest.main()
