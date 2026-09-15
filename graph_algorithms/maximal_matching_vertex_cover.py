"""A small vertex cover approximation using a maximal matching."""


def maximal_matching_vertex_cover(edges):
    """Return a vertex cover and the matching used to build it."""
    remaining_edges = []

    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Every edge needs two endpoints")

        u, v = edge
        if u == v:
            raise ValueError("This example only works with simple graphs")

        remaining_edges.append((u, v))

    cover = set()
    matching = []

    while remaining_edges:
        # For now I just take the first edge I can find.
        u, v = remaining_edges[0]
        matching.append((u, v))

        # I take both sides, even though sometimes one would already be enough.
        cover.add(u)
        cover.add(v)

        # Anything touching these vertices is covered now and can be ignored.
        remaining_edges = [
            edge
            for edge in remaining_edges
            if u not in edge and v not in edge
        ]

    return cover, matching


def is_vertex_cover(edges, vertices):
    """Check whether every edge has at least one endpoint in the set."""
    return all(u in vertices or v in vertices for u, v in edges)


def run_example():
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
    ]

    cover, matching = maximal_matching_vertex_cover(edges)

    print("Edges:", edges)
    print("Matching I ended up with:", matching)
    print("Vertex cover:", sorted(cover))
    print("Does it cover every edge?", is_vertex_cover(edges, cover))
    print()
    print("The cover is not necessarily the smallest one.")
    print("The useful part is that the matching gives a simple 2-approximation.")


if __name__ == "__main__":
    run_example()
