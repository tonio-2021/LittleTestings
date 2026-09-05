import unittest

from maximal_matching_vertex_cover import (
    is_vertex_cover,
    maximal_matching_vertex_cover,
)


class VertexCoverTests(unittest.TestCase):
    def test_result_covers_all_edges(self):
        edges = [
            ("a", "b"),
            ("a", "c"),
            ("b", "d"),
            ("c", "d"),
            ("d", "e"),
            ("e", "f"),
        ]

        cover, _ = maximal_matching_vertex_cover(edges)

        self.assertTrue(is_vertex_cover(edges, cover))

    def test_matching_edges_do_not_share_vertices(self):
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

        _, matching = maximal_matching_vertex_cover(edges)
        used_vertices = [vertex for edge in matching for vertex in edge]

        self.assertEqual(len(used_vertices), len(set(used_vertices)))

    def test_empty_graph(self):
        cover, matching = maximal_matching_vertex_cover([])

        self.assertEqual(cover, set())
        self.assertEqual(matching, [])


if __name__ == "__main__":
    unittest.main()
