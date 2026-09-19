import math
import unittest

from optimization.finite_difference_accuracy.differences import (
    centered_difference,
    error_table,
    forward_difference,
)


class FiniteDifferenceTests(unittest.TestCase):
    def test_centered_difference_on_a_quadratic(self):
        estimate = centered_difference(lambda x: x**2, 2.0, 0.001)

        self.assertAlmostEqual(estimate, 4.0, places=10)

    def test_centered_is_more_accurate_for_a_reasonable_step(self):
        step = 0.001
        exact = math.cos(1.0)
        forward_error = abs(forward_difference(math.sin, 1.0, step) - exact)
        centered_error = abs(centered_difference(math.sin, 1.0, step) - exact)

        self.assertLess(centered_error, forward_error)

    def test_tiny_steps_show_roundoff(self):
        rows = error_table(
            math.sin,
            math.cos,
            1.0,
            [1e-2, 1e-5, 1e-8, 1e-12, 1e-16],
        )
        centered_errors = [row[2] for row in rows]

        self.assertLess(min(centered_errors[:-1]), centered_errors[-1])

    def test_error_table_keeps_the_steps(self):
        steps = [0.1, 0.01, 0.001]
        rows = error_table(math.exp, math.exp, 0.0, steps)

        self.assertEqual([row[0] for row in rows], steps)
        self.assertTrue(all(row[1] >= 0 and row[2] >= 0 for row in rows))

    def test_bad_steps_are_rejected(self):
        for step in (0, -0.1, math.inf, math.nan, True, "small"):
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    forward_difference(math.sin, 1.0, step)


if __name__ == "__main__":
    unittest.main()
