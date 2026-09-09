import unittest
from itertools import combinations

from independent_set_vertex_cover import (
    is_independent_set,
    maximum_independent_set,
    minimum_vertex_cover,
)
from maximal_matching_vertex_cover import is_vertex_cover


class IndependentSetVertexCoverTests(unittest.TestCase):
    def setUp(self):
        self.vertices = ["a", "b", "c", "d", "e", "f"]
        self.edges = [
            ("a", "b"),
            ("a", "c"),
            ("b", "d"),
            ("c", "d"),
            ("d", "e"),
            ("e", "f"),
        ]

    def test_every_independent_set_has_a_cover_as_its_complement(self):
        all_vertices = set(self.vertices)

        for size in range(len(self.vertices) + 1):
            for group in combinations(self.vertices, size):
                group = set(group)
                self.assertEqual(
                    is_independent_set(self.edges, group),
                    is_vertex_cover(self.edges, all_vertices - group),
                )

    def test_best_sizes_add_up_to_number_of_vertices(self):
        independent_set = maximum_independent_set(self.vertices, self.edges)
        vertex_cover = minimum_vertex_cover(self.vertices, self.edges)

        self.assertTrue(is_independent_set(self.edges, independent_set))
        self.assertTrue(is_vertex_cover(self.edges, vertex_cover))
        self.assertEqual(
            len(independent_set) + len(vertex_cover),
            len(self.vertices),
        )

    def test_isolated_vertex_can_stay_in_the_independent_set(self):
        independent_set = maximum_independent_set(
            ["a", "b", "alone"],
            [("a", "b")],
        )

        self.assertIn("alone", independent_set)
        self.assertEqual(len(independent_set), 2)

    def test_empty_graph(self):
        self.assertEqual(maximum_independent_set([], []), set())
        self.assertEqual(minimum_vertex_cover([], []), set())

    def test_unknown_edge_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            maximum_independent_set(["a"], [("a", "b")])


if __name__ == "__main__":
    unittest.main()
