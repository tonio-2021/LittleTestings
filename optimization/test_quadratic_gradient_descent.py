import unittest

from optimization.quadratic_gradient_descent import gradient_descent, quadratic_value


class QuadraticGradientDescentTests(unittest.TestCase):
    def test_minimum_has_value_zero(self):
        self.assertEqual(quadratic_value(1.0), 0.0)

    def test_small_step_moves_towards_the_minimum(self):
        positions, values = gradient_descent(-4.0, 0.2, 40)

        self.assertAlmostEqual(positions[-1], 1.0, places=6)
        self.assertTrue(
            all(next_value <= value for value, next_value in zip(values, values[1:]))
        )

    def test_step_near_the_limit_jumps_across_the_minimum(self):
        positions, _ = gradient_descent(-4.0, 0.9, 5)
        offsets = [x - 1.0 for x in positions]

        self.assertTrue(
            all(left * right < 0 for left, right in zip(offsets, offsets[1:]))
        )

    def test_step_above_the_limit_moves_away(self):
        _, values = gradient_descent(-4.0, 1.05, 8)

        self.assertGreater(values[-1], values[0])

    def test_zero_iterations_only_returns_the_start(self):
        positions, values = gradient_descent(3.0, 0.2, 0)

        self.assertEqual(positions, [3.0])
        self.assertEqual(values, [quadratic_value(3.0)])

    def test_bad_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            gradient_descent(0.0, 0.0, 5)
        with self.assertRaises(ValueError):
            gradient_descent(0.0, 0.2, -1)
        with self.assertRaises(ValueError):
            gradient_descent(0.0, 0.2, 5, curvature=0.0)


if __name__ == "__main__":
    unittest.main()
