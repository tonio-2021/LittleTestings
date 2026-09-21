import math
import random
import unittest

from monte_carlo.product_stopping_time.simulation import (
    simulate_counts,
    stopping_count,
    summarize_counts,
    theoretical_summary,
)


class ProductStoppingTimeTests(unittest.TestCase):
    def test_count_is_always_positive(self):
        counts = simulate_counts(100, 0.2, random.Random(4))

        self.assertTrue(all(count >= 1 for count in counts))

    def test_fixed_seed_repeats_the_same_experiment(self):
        first = simulate_counts(20, 0.1, random.Random(12))
        second = simulate_counts(20, 0.1, random.Random(12))

        self.assertEqual(first, second)

    def test_theory_for_exp_minus_two(self):
        expected_mean, expected_variance, probabilities = theoretical_summary(
            math.exp(-2)
        )

        self.assertAlmostEqual(expected_mean, 3.0)
        self.assertAlmostEqual(expected_variance, 2.0)
        self.assertAlmostEqual(probabilities[1], math.exp(-2))
        self.assertAlmostEqual(probabilities[2], 2 * math.exp(-2))

    def test_simulation_is_close_to_theory(self):
        threshold = math.exp(-2)
        counts = simulate_counts(30_000, threshold, random.Random(4212))
        sample_mean, sample_variance, observed = summarize_counts(counts)
        exact_mean, exact_variance, expected = theoretical_summary(threshold)

        self.assertAlmostEqual(sample_mean, exact_mean, delta=0.04)
        self.assertAlmostEqual(sample_variance, exact_variance, delta=0.10)
        for count in range(1, 6):
            self.assertAlmostEqual(observed[count], expected[count], delta=0.012)

    def test_summary_of_known_counts(self):
        sample_mean, sample_variance, probabilities = summarize_counts([1, 2, 2, 3])

        self.assertEqual(sample_mean, 2)
        self.assertEqual(sample_variance, 0.5)
        self.assertEqual(probabilities[2], 0.5)

    def test_bad_inputs_are_rejected(self):
        for threshold in (0, 1, -0.1, math.inf, math.nan, True):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    stopping_count(threshold)
        with self.assertRaises(ValueError):
            simulate_counts(0, 0.2)
        with self.assertRaises(ValueError):
            summarize_counts([])


if __name__ == "__main__":
    unittest.main()
