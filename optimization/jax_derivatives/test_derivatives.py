import math
import unittest

from optimization.jax_derivatives.derivatives import (
    finite_difference_derivatives,
    jax_derivatives,
)


class DerivativeTests(unittest.TestCase):
    def test_simple_point_with_known_answer(self):
        gradient, trace = jax_derivatives((0.0, 0.0))

        self.assertAlmostEqual(gradient[0], 0.0)
        self.assertAlmostEqual(gradient[1], 0.0)
        self.assertAlmostEqual(trace, 2.0)

    def test_finite_differences_are_close_at_two_points(self):
        for point in ((0.7, -0.4), (-1.2, 0.5)):
            with self.subTest(point=point):
                gradient, trace = jax_derivatives(point)
                rough_gradient, rough_trace = finite_difference_derivatives(point)

                for exact, rough in zip(gradient, rough_gradient):
                    self.assertAlmostEqual(exact, rough, delta=0.0002)
                self.assertAlmostEqual(trace, rough_trace, delta=0.0002)

    def test_bad_point_or_step(self):
        for point in ((1,), (1, 2, 3), (math.nan, 0)):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    jax_derivatives(point)
        for step in (0, -0.1, math.inf):
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    finite_difference_derivatives((0, 0), step)


if __name__ == "__main__":
    unittest.main()
