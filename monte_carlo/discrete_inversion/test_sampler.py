import math
import random
import unittest

from monte_carlo.discrete_inversion.sampler import DiscreteSampler


class DiscreteSamplerTests(unittest.TestCase):
    def test_boundaries_and_zero_probability(self):
        sampler = DiscreteSampler([0.2, 0.0, 0.5, 0.3])
        examples = [(0.0, 0), (0.199, 0), (0.2, 2), (0.699, 2), (0.7, 3)]

        for number, expected in examples:
            self.assertEqual(sampler.pick_linear(number), expected)
            self.assertEqual(sampler.pick_binary(number), expected)

    def test_both_searches_agree_on_random_draws(self):
        sampler = DiscreteSampler([0.1, 0.2, 0.3, 0.4])
        rng = random.Random(42)

        for _ in range(10_000):
            number = rng.random()
            self.assertEqual(sampler.pick_linear(number), sampler.pick_binary(number))

    def test_frequencies_are_close_to_the_table(self):
        probabilities = [0.1, 0.2, 0.3, 0.4]
        sampler = DiscreteSampler(probabilities)
        rng = random.Random(42)
        draws = [sampler.pick_binary(rng.random()) for _ in range(10_000)]

        for index, target in enumerate(probabilities):
            observed = draws.count(index) / len(draws)
            self.assertLess(abs(observed - target), 0.02)

    def test_bad_probability_tables(self):
        for probabilities in ([], [0.4, 0.4], [0.5, -0.5, 1.0], [math.nan, 1.0]):
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    DiscreteSampler(probabilities)

    def test_draw_must_be_in_unit_interval(self):
        sampler = DiscreteSampler([0.5, 0.5])
        for number in (-0.1, 1.0, math.nan):
            with self.subTest(number=number):
                with self.assertRaises(ValueError):
                    sampler.pick_linear(number)
                with self.assertRaises(ValueError):
                    sampler.pick_binary(number)


if __name__ == "__main__":
    unittest.main()
