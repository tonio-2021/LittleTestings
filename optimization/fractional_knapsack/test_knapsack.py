import math
import random
import unittest

from scipy.optimize import linprog

from optimization.fractional_knapsack.knapsack import fractional_knapsack


class FractionalKnapsackTests(unittest.TestCase):
    def test_partial_last_item_and_pick_order(self):
        items = [("small", 2, 12), ("medium", 3, 15), ("large", 5, 20)]
        fractions, used_weight, total_value, choices = fractional_knapsack(items, 4)

        self.assertEqual(fractions[0], 1.0)
        self.assertAlmostEqual(fractions[1], 2 / 3)
        self.assertEqual(fractions[2], 0.0)
        self.assertAlmostEqual(used_weight, 4)
        self.assertAlmostEqual(total_value, 22)
        self.assertEqual([name for name, _, _ in choices], ["small", "medium"])

    def test_matches_linear_program_on_small_random_bags(self):
        rng = random.Random(7)
        for _ in range(12):
            items = [
                (f"item {index}", rng.randint(1, 8), rng.randint(1, 20))
                for index in range(4)
            ]
            capacity = rng.randint(1, 15)
            fractions, used_weight, total_value, _ = fractional_knapsack(items, capacity)
            lp = linprog(
                [-value for _, _, value in items],
                A_ub=[[weight for _, weight, _ in items]],
                b_ub=[capacity],
                bounds=(0, 1),
            )

            self.assertTrue(lp.success)
            self.assertLessEqual(used_weight, capacity + 1e-9)
            self.assertTrue(all(0 <= fraction <= 1 for fraction in fractions))
            self.assertAlmostEqual(total_value, -lp.fun, places=7)

    def test_empty_and_zero_capacity(self):
        self.assertEqual(fractional_knapsack([], 3), ([], 0.0, 0.0, []))
        self.assertEqual(
            fractional_knapsack([("one", 2, 4)], 0),
            ([0.0], 0.0, 0.0, []),
        )

    def test_no_need_to_take_a_zero_value_item(self):
        fractions, used_weight, total_value, _ = fractional_knapsack(
            [("worthless", 1, 0), ("useful", 2, 4)], 10
        )

        self.assertEqual(fractions, [0.0, 1.0])
        self.assertEqual(used_weight, 2)
        self.assertEqual(total_value, 4)

    def test_bad_items_and_capacity(self):
        for items, capacity in (
            ([("x", 1, 2)], -1),
            ([("x", 1, 2)], math.inf),
            ([("x", 0, 2)], 5),
            ([("x", -1, 2)], 5),
            ([("x", 1, -2)], 5),
            ([("x", 1, math.nan)], 5),
        ):
            with self.subTest(items=items, capacity=capacity):
                with self.assertRaises(ValueError):
                    fractional_knapsack(items, capacity)


if __name__ == "__main__":
    unittest.main()
