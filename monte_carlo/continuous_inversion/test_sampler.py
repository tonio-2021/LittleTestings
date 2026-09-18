import math
import random
from statistics import quantiles
import unittest

from monte_carlo.continuous_inversion.sampler import (
    exponential_inverse,
    power_inverse,
    sample_inverse,
)


class ContinuousInversionTests(unittest.TestCase):
    def test_known_inverse_values(self):
        self.assertEqual(exponential_inverse(0, rate=2), 0)
        self.assertAlmostEqual(exponential_inverse(0.5, rate=2), math.log(2) / 2)
        self.assertAlmostEqual(power_inverse(0.25, shape=2), 0.5)

    def test_same_seed_gives_same_samples(self):
        first = sample_inverse(power_inverse, 20, random.Random(7))
        second = sample_inverse(power_inverse, 20, random.Random(7))

        self.assertEqual(first, second)

    def test_sample_quartiles_are_close(self):
        examples = (
            lambda number: exponential_inverse(number, rate=2),
            lambda number: power_inverse(number, shape=3),
        )
        for inverse_cdf in examples:
            with self.subTest(inverse_cdf=inverse_cdf):
                draws = sample_inverse(inverse_cdf, 10_000, random.Random(42))
                observed = quantiles(draws, n=4, method="inclusive")
                expected = [inverse_cdf(p) for p in (0.25, 0.5, 0.75)]

                for sample_quartile, target in zip(observed, expected):
                    self.assertLess(abs(sample_quartile - target), 0.025)

    def test_bad_inputs(self):
        for number in (-0.1, 1.0, math.nan):
            with self.subTest(number=number):
                with self.assertRaises(ValueError):
                    exponential_inverse(number)
                with self.assertRaises(ValueError):
                    power_inverse(number)

        with self.assertRaises(ValueError):
            exponential_inverse(0.5, rate=0)
        with self.assertRaises(ValueError):
            power_inverse(0.5, shape=-1)
        with self.assertRaises(ValueError):
            sample_inverse(power_inverse, 0)
        with self.assertRaises(ValueError):
            sample_inverse(power_inverse, 2.5)


if __name__ == "__main__":
    unittest.main()
