"""Try random cuts on small weighted graphs."""

import math
import random
from dataclasses import dataclass
from itertools import product
from numbers import Real


@dataclass(frozen=True)
class CutExperiment:
    average_weight: float
    best_weight: float
    best_side: frozenset


def _check_edges(edges):
    checked = []

    for edge in edges:
        try:
            left, right, weight = edge
        except (TypeError, ValueError) as error:
            raise ValueError("each edge needs two vertices and a weight") from error

        try:
            hash(left)
            hash(right)
        except TypeError as error:
            raise ValueError("vertex names need to be hashable") from error

        if left == right:
            raise ValueError("self-loops are not useful for this cut")
        if not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
            raise ValueError("edge weights need to be finite and nonnegative")

        checked.append((left, right, float(weight)))

    return checked


def _vertices(edges):
    vertices = {vertex for left, right, _ in edges for vertex in (left, right)}
    return sorted(vertices, key=repr)


def cut_weight(edges, one_side):
    """Add the weights of edges whose endpoints are on different sides."""
    edges = _check_edges(edges)
    one_side = set(one_side)
    return sum(
        weight
        for left, right, weight in edges
        if (left in one_side) != (right in one_side)
    )


def random_cut(edges, rng=None):
    """Put every vertex on either side with equal probability."""
    edges = _check_edges(edges)
    rng = rng or random.Random()
    one_side = {vertex for vertex in _vertices(edges) if rng.random() < 0.5}
    return frozenset(one_side), cut_weight(edges, one_side)


def try_random_cuts(edges, trials=1000, seed=None):
    """Repeat the random choice and keep its average and best result."""
    edges = _check_edges(edges)
    if not isinstance(trials, int) or isinstance(trials, bool) or trials <= 0:
        raise ValueError("trials needs to be a positive integer")

    rng = random.Random(seed)
    total = 0.0
    best_weight = -1.0
    best_side = frozenset()

    for _ in range(trials):
        one_side, weight = random_cut(edges, rng)
        total += weight
        if weight > best_weight:
            best_weight = weight
            best_side = one_side

    return CutExperiment(total / trials, best_weight, best_side)


def exact_max_cut(edges):
    """Check every cut, which is only meant for small comparisons."""
    edges = _check_edges(edges)
    vertices = _vertices(edges)
    if not vertices:
        return frozenset(), 0.0

    # A cut and its flipped copy have the same weight, so one vertex can stay fixed.
    free_vertices = vertices[1:]
    best_side = frozenset()
    best_weight = -1.0

    for choices in product((False, True), repeat=len(free_vertices)):
        one_side = frozenset(
            vertex
            for vertex, chosen in zip(free_vertices, choices)
            if chosen
        )
        weight = cut_weight(edges, one_side)
        if weight > best_weight:
            best_side = one_side
            best_weight = weight

    return best_side, best_weight


def run_example():
    edges = [
        ("a", "b", 3),
        ("b", "c", 2),
        ("c", "d", 4),
        ("d", "a", 1),
        ("a", "c", 2),
        ("b", "d", 1),
    ]

    total_weight = sum(weight for _, _, weight in edges)
    experiment = try_random_cuts(edges, trials=5000, seed=7)
    exact_side, exact_weight = exact_max_cut(edges)

    print(f"Total edge weight: {total_weight:.1f}")
    print(f"Half of that:      {total_weight / 2:.3f}")
    print(f"Random average:    {experiment.average_weight:.3f}")
    print(
        f"Best random cut:   {experiment.best_weight:.1f} "
        f"with one side {sorted(experiment.best_side)}"
    )
    print(f"Exact best cut:    {exact_weight:.1f} with one side {sorted(exact_side)}")


if __name__ == "__main__":
    run_example()
