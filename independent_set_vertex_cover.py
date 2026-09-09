"""A small check of how independent sets and vertex covers fit together."""

from itertools import combinations

from maximal_matching_vertex_cover import is_vertex_cover


def is_independent_set(edges, vertices):
    """Check that no edge has both endpoints in the chosen set."""
    chosen = set(vertices)
    return all(not (u in chosen and v in chosen) for u, v in edges)


def _prepare_graph(vertices, edges):
    # Keeping the order makes the result a bit easier to read in the example.
    vertices = list(dict.fromkeys(vertices))
    vertex_set = set(vertices)
    edges = list(edges)

    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Every edge needs two endpoints")
        if edge[0] not in vertex_set or edge[1] not in vertex_set:
            raise ValueError("Every edge endpoint needs to be in vertices")

    return vertices, edges


def maximum_independent_set(vertices, edges):
    """Find one largest independent set by checking all subsets."""
    vertices, edges = _prepare_graph(vertices, edges)

    # This is slow for big graphs, but fine for the little examples here.
    for size in range(len(vertices), -1, -1):
        for group in combinations(vertices, size):
            if is_independent_set(edges, group):
                return set(group)

    return set()


def minimum_vertex_cover(vertices, edges):
    """Find one smallest vertex cover by checking all subsets."""
    vertices, edges = _prepare_graph(vertices, edges)

    for size in range(len(vertices) + 1):
        for group in combinations(vertices, size):
            if is_vertex_cover(edges, set(group)):
                return set(group)

    return set()


def run_example():
    vertices = ["a", "b", "c", "d", "e", "f"]
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
    ]

    independent_set = maximum_independent_set(vertices, edges)
    vertex_cover = minimum_vertex_cover(vertices, edges)

    print("Vertices:", vertices)
    print("Edges:", edges)
    print("Largest independent set:", sorted(independent_set))
    print("Smallest vertex cover:", sorted(vertex_cover))
    print()
    print("Complement of the independent set:", sorted(set(vertices) - independent_set))
    print("Is that a vertex cover?", is_vertex_cover(edges, set(vertices) - independent_set))
    print("Do the two best sizes add up?", len(independent_set) + len(vertex_cover) == len(vertices))


if __name__ == "__main__":
    run_example()
