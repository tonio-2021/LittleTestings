import random
import unittest

from optimization.random_cut.random_cut import (
    cut_weight,
    exact_max_cut,
    random_cut,
    try_random_cuts,
)


class RandomCutTests(unittest.TestCase):
    def setUp(self):
        self.edges = [
            ("a", "b", 3),
            ("b", "c", 2),
            ("c", "d", 4),
            ("d", "a", 1),
            ("a", "c", 2),
            ("b", "d", 1),
        ]

    def test_cut_weight_only_counts_crossing_edges(self):
        self.assertEqual(cut_weight(self.edges, {"a", "c"}), 10.0)
        self.assertEqual(cut_weight(self.edges, set()), 0.0)

    def test_random_cut_returns_a_matching_weight(self):
        side, weight = random_cut(self.edges, random.Random(4))

        self.assertLessEqual(side, {"a", "b", "c", "d"})
        self.assertEqual(weight, cut_weight(self.edges, side))

    def test_exact_cut_finds_small_graph_optimum(self):
        side, weight = exact_max_cut(self.edges)

        self.assertEqual(weight, 10.0)
        self.assertEqual(cut_weight(self.edges, side), weight)

    def test_random_average_is_close_to_half_the_total_weight(self):
        result = try_random_cuts(self.edges, trials=10000, seed=12)

        self.assertAlmostEqual(result.average_weight, 6.5, delta=0.15)

    def test_repeated_cuts_can_find_the_small_optimum(self):
        result = try_random_cuts(self.edges, trials=200, seed=3)
        _, exact_weight = exact_max_cut(self.edges)

        self.assertEqual(result.best_weight, exact_weight)

    def test_empty_graph_and_bad_inputs(self):
        result = try_random_cuts([], trials=5, seed=1)
        self.assertEqual(result.average_weight, 0.0)
        self.assertEqual(result.best_side, frozenset())

        with self.assertRaises(ValueError):
            try_random_cuts(self.edges, trials=0)
        with self.assertRaises(ValueError):
            cut_weight([("a", "a", 1)], {"a"})
        with self.assertRaises(ValueError):
            cut_weight([("a", "b", -1)], {"a"})


if __name__ == "__main__":
    unittest.main()
