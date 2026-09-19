"""Check the link between cliques and independent sets in a complement graph."""

from itertools import combinations

from graph_algorithms.independent_set_vertex_cover import is_independent_set


def _prepare_graph(vertices, edges):
    vertices = list(dict.fromkeys(vertices))
    vertex_set = set(vertices)
    edge_keys = set()

    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Every edge needs two endpoints")
        left, right = edge
        if left not in vertex_set or right not in vertex_set:
            raise ValueError("Every edge endpoint needs to be in vertices")
        if left == right:
            raise ValueError("Self-loops are not used in this example")
        edge_keys.add(frozenset((left, right)))

    return vertices, edge_keys


def complement_edges(vertices, edges):
    """Return the missing edges of a small undirected graph."""
    vertices, edge_keys = _prepare_graph(vertices, edges)

    # Looking at every pair is enough because the graph is undirected.
    return [
        (left, right)
        for left, right in combinations(vertices, 2)
        if frozenset((left, right)) not in edge_keys
    ]


def is_clique(edges, vertices):
    """Check whether all pairs in vertices have an edge between them."""
    edge_keys = {frozenset(edge) for edge in edges}
    return all(
        frozenset(pair) in edge_keys
        for pair in combinations(set(vertices), 2)
    )


def maximum_clique(vertices, edges):
    """Find one largest clique by trying all subsets."""
    vertices, edge_keys = _prepare_graph(vertices, edges)
    clean_edges = [tuple(edge) for edge in edge_keys]

    # This grows badly, but it lets me see the transformation on tiny graphs.
    for size in range(len(vertices), -1, -1):
        for group in combinations(vertices, size):
            if is_clique(clean_edges, group):
                return set(group)

    return set()


def run_example():
    vertices = ["a", "b", "c", "d", "e"]
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
    ]

    missing_edges = complement_edges(vertices, edges)
    clique = maximum_clique(vertices, edges)

    print("Original graph edges:", edges)
    print("Complement graph edges:", missing_edges)
    print("\nLargest clique in the original:", sorted(clique))
    print("Same vertices in the complement:", sorted(clique))
    print("Independent there?", is_independent_set(missing_edges, clique))


if __name__ == "__main__":
    run_example()
