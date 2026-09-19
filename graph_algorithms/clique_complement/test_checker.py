import unittest
from itertools import combinations

from graph_algorithms.clique_complement.checker import (
    complement_edges,
    is_clique,
    maximum_clique,
)
from graph_algorithms.independent_set_vertex_cover import is_independent_set


class CliqueComplementTests(unittest.TestCase):
    def setUp(self):
        self.vertices = ["a", "b", "c", "d", "e"]
        self.edges = [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
        ]

    def test_complement_contains_exactly_the_missing_pairs(self):
        complement = complement_edges(self.vertices, self.edges)
        original = {frozenset(edge) for edge in self.edges}
        missing = {frozenset(edge) for edge in complement}
        all_pairs = {frozenset(pair) for pair in combinations(self.vertices, 2)}

        self.assertFalse(original & missing)
        self.assertEqual(original | missing, all_pairs)

    def test_cliques_become_independent_sets(self):
        complement = complement_edges(self.vertices, self.edges)

        for size in range(len(self.vertices) + 1):
            for group in combinations(self.vertices, size):
                self.assertEqual(
                    is_clique(self.edges, group),
                    is_independent_set(complement, group),
                )

    def test_finds_the_triangle(self):
        clique = maximum_clique(self.vertices, self.edges)

        self.assertEqual(clique, {"a", "b", "c"})
        self.assertTrue(is_clique(self.edges, clique))

    def test_complementing_twice_gets_the_graph_back(self):
        first = complement_edges(self.vertices, self.edges)
        second = complement_edges(self.vertices, first)

        self.assertEqual(
            {frozenset(edge) for edge in second},
            {frozenset(edge) for edge in self.edges},
        )

    def test_invalid_edges_are_rejected(self):
        with self.assertRaises(ValueError):
            complement_edges(["a", "b"], [("a", "missing")])
        with self.assertRaises(ValueError):
            maximum_clique(["a"], [("a", "a")])


if __name__ == "__main__":
    unittest.main()
