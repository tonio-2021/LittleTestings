"""Fill a knapsack when taking part of an item is allowed."""

import math


def fractional_knapsack(items, capacity):
    """Return fractions, used weight, value, and the choices in pick order."""
    items = list(items)
    if not math.isfinite(capacity) or capacity < 0:
        raise ValueError("capacity must be finite and nonnegative")

    for name, weight, value in items:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("item weights must be finite and positive")
        if not math.isfinite(value) or value < 0:
            raise ValueError("item values must be finite and nonnegative")

    # Try the most value per unit of weight first.
    order = sorted(
        range(len(items)),
        key=lambda index: items[index][2] / items[index][1],
        reverse=True,
    )
    fractions = [0.0] * len(items)
    choices = []
    remaining = capacity

    for index in order:
        if remaining <= 0:
            break
        name, weight, value = items[index]
        if value == 0:
            break

        fraction = min(1.0, remaining / weight)
        fractions[index] = fraction
        choices.append((name, value / weight, fraction))
        remaining = max(0.0, remaining - fraction * weight)

    used_weight = math.fsum(
        fraction * item[1] for fraction, item in zip(fractions, items)
    )
    total_value = math.fsum(
        fraction * item[2] for fraction, item in zip(fractions, items)
    )
    return fractions, used_weight, total_value, choices


def run_example():
    from scipy.optimize import linprog

    items = [("small", 2, 12), ("medium", 3, 15), ("large", 5, 20)]
    capacity = 4
    fractions, used_weight, total_value, choices = fractional_knapsack(items, capacity)

    print(f"Bag capacity: {capacity}")
    print("Pick order (value per weight, fraction taken):")
    for name, density, fraction in choices:
        print(f"  {name:7} {density:.1f}   {fraction:.3f}")
    print(f"Fractions in original item order: {[round(f, 3) for f in fractions]}")
    print(f"Used weight: {used_weight:.3f}")
    print(f"Greedy value: {total_value:.3f}")

    # The LP is just a check on this tiny example, not how the greedy pick works.
    lp = linprog(
        [-value for _, _, value in items],
        A_ub=[[weight for _, weight, _ in items]],
        b_ub=[capacity],
        bounds=(0, 1),
    )
    if not lp.success:
        raise RuntimeError(lp.message)
    print(f"LP value:     {-lp.fun:.3f}")


if __name__ == "__main__":
    run_example()
