import unittest

from maximal_matching_vertex_cover import is_vertex_cover
from tree_vertex_cover import minimum_tree_vertex_cover


class TreeVertexCoverTests(unittest.TestCase):
    def test_path_needs_two_vertices(self):
        edges = [(1, 2), (2, 3), (3, 4)]

        cover = minimum_tree_vertex_cover(edges)

        self.assertTrue(is_vertex_cover(edges, cover))
        self.assertEqual(len(cover), 2)

    def test_star_only_needs_the_centre(self):
        edges = [
            ("centre", "a"),
            ("centre", "b"),
            ("centre", "c"),
            ("centre", "d"),
        ]

        cover = minimum_tree_vertex_cover(edges)

        self.assertEqual(cover, {"centre"})

    def test_empty_tree(self):
        self.assertEqual(minimum_tree_vertex_cover([]), set())

    def test_cycle_is_rejected(self):
        edges = [(1, 2), (2, 3), (3, 1)]

        with self.assertRaises(ValueError):
            minimum_tree_vertex_cover(edges)


if __name__ == "__main__":
    unittest.main()
