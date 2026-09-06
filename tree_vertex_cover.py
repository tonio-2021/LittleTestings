"""Finding a minimum vertex cover when the graph is a tree."""

from maximal_matching_vertex_cover import (
    is_vertex_cover,
    maximal_matching_vertex_cover,
)


def minimum_tree_vertex_cover(edges):
    """Return one smallest vertex cover of a tree."""
    edges = list(edges)
    if not edges:
        return set()

    graph = {}
    seen_edges = set()

    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Every edge needs two endpoints")

        u, v = edge
        if u == v:
            raise ValueError("A tree cannot have a self-loop")

        edge_key = frozenset((u, v))
        if edge_key in seen_edges:
            raise ValueError("The same edge appears more than once")
        seen_edges.add(edge_key)

        graph.setdefault(u, []).append(v)
        graph.setdefault(v, []).append(u)

    root = edges[0][0]
    parent = {root: None}
    order = [root]

    # I root the tree first because the parent and child idea is easier that way.
    for vertex in order:
        for neighbour in graph[vertex]:
            if neighbour == parent[vertex]:
                continue
            if neighbour in parent:
                raise ValueError("The edges contain a cycle")

            parent[neighbour] = vertex
            order.append(neighbour)

    if len(order) != len(graph):
        raise ValueError("The graph is not connected")

    included = {}
    excluded = {}

    # Starting at the end means the children are already done when I need them.
    for vertex in reversed(order):
        children = [
            neighbour
            for neighbour in graph[vertex]
            if parent.get(neighbour) == vertex
        ]

        # When this vertex is used, the children can choose their cheaper option.
        included[vertex] = 1 + sum(
            min(included[child], excluded[child]) for child in children
        )

        # When I skip it, every child has to cover the edge between them.
        excluded[vertex] = sum(included[child] for child in children)

    cover = set()
    take_root = included[root] <= excluded[root]
    choices = [(root, take_root)]

    while choices:
        vertex, take_vertex = choices.pop()
        if take_vertex:
            cover.add(vertex)

        for neighbour in graph[vertex]:
            if parent.get(neighbour) != vertex:
                continue

            if take_vertex:
                # The parent is already in, so I just use the cheaper child choice.
                take_child = included[neighbour] <= excluded[neighbour]
            else:
                # Otherwise this child has to be used to cover their edge.
                take_child = True

            choices.append((neighbour, take_child))

    return cover


def run_example():
    tree_edges = [(1, 2), (2, 3), (3, 4)]

    exact_cover = minimum_tree_vertex_cover(tree_edges)
    approximate_cover, matching = maximal_matching_vertex_cover(tree_edges)

    print("Tree edges:", tree_edges)
    print("Matching used by the approximation:", matching)
    print("Approximate cover:", sorted(approximate_cover))
    print("Exact tree cover:", sorted(exact_cover))
    print("Exact result covers every edge:", is_vertex_cover(tree_edges, exact_cover))
    print()
    print("The tree structure gives more information than the general method.")
    print("Here the dynamic program finds a cover with two vertices instead of four.")


if __name__ == "__main__":
    run_example()
