import tempfile
import unittest
from pathlib import Path

from monte_carlo.congruential_generator.generator import (
    cycle_period,
    multiplicative_numbers,
    save_pair_plot,
)


class CongruentialGeneratorTests(unittest.TestCase):
    def test_first_states_follow_the_recurrence(self):
        numbers = multiplicative_numbers(1, 3, 31, 5)
        expected_states = [3, 9, 27, 19, 26]

        self.assertEqual(numbers, [state / 31 for state in expected_states])

    def test_same_parameters_repeat_the_sequence(self):
        first = multiplicative_numbers(7, 3, 31, 20)
        second = multiplicative_numbers(7, 3, 31, 20)

        self.assertEqual(first, second)

    def test_values_stay_inside_unit_interval(self):
        numbers = multiplicative_numbers(1, 3, 31, 100)

        self.assertTrue(all(0 < value < 1 for value in numbers))

    def test_multiplier_choice_changes_the_period(self):
        self.assertEqual(cycle_period(1, 3, 31), 30)
        self.assertEqual(cycle_period(1, 2, 31), 5)

    def test_plot_is_saved(self):
        good = multiplicative_numbers(1, 3, 31, 40)
        poor = multiplicative_numbers(1, 2, 31, 40)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.png"
            returned = save_pair_plot(good, poor, path)

            self.assertEqual(returned, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_bad_parameters_are_rejected(self):
        bad_parameters = [
            (0, 3, 31, 5),
            (1, 0, 31, 5),
            (1, 3, 1, 5),
            (1, 2, 10, 5),
            (1, 3, 31, 0),
        ]

        for seed, multiplier, modulus, count in bad_parameters:
            with self.subTest(
                seed=seed,
                multiplier=multiplier,
                modulus=modulus,
                count=count,
            ):
                with self.assertRaises(ValueError):
                    multiplicative_numbers(seed, multiplier, modulus, count)


if __name__ == "__main__":
    unittest.main()
