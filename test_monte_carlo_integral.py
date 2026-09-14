import math
import random
import unittest

from monte_carlo_integral import estimate_integral, midpoint_integral


class MonteCarloIntegralTests(unittest.TestCase):
    def test_constant_height_has_no_sampling_error(self):
        estimate, standard_error = estimate_integral(
            lambda x: 2, 1, 4, 20, random.Random(1)
        )

        self.assertEqual(estimate, 6)
        self.assertEqual(standard_error, 0)

    def test_log_three_is_close_to_the_estimate(self):
        estimate, standard_error = estimate_integral(
            lambda x: 1 / x, 1, 3, 10_000, random.Random(42)
        )

        self.assertLess(abs(estimate - math.log(3)), 3 * standard_error)
        self.assertGreater(standard_error, 0)

    def test_same_seed_repeats_the_result(self):
        first = estimate_integral(lambda x: x * x, 0, 2, 50, random.Random(7))
        second = estimate_integral(lambda x: x * x, 0, 2, 50, random.Random(7))

        self.assertEqual(first, second)

    def test_midpoint_rule_on_a_line(self):
        self.assertAlmostEqual(midpoint_integral(lambda x: x, 0, 2, 10), 2)

    def test_midpoint_rule_for_log_three(self):
        result = midpoint_integral(lambda x: 1 / x, 1, 3, 1_000)

        self.assertAlmostEqual(result, math.log(3), places=6)

    def test_bad_ranges_and_sample_counts(self):
        with self.assertRaises(ValueError):
            estimate_integral(lambda x: x, 2, 1, 10)
        with self.assertRaises(ValueError):
            estimate_integral(lambda x: x, 0, 1, 1)
        with self.assertRaises(ValueError):
            midpoint_integral(lambda x: x, 1, 1, 10)
        with self.assertRaises(ValueError):
            midpoint_integral(lambda x: x, 0, 1, 0)


if __name__ == "__main__":
    unittest.main()
