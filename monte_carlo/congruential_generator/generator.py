"""A small multiplicative congruential random-number experiment."""

import math
from pathlib import Path


def _check_parameters(seed, multiplier, modulus):
    values = (seed, multiplier, modulus)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("seed, multiplier, and modulus need to be integers")
    if modulus <= 1:
        raise ValueError("modulus needs to be larger than one")
    if not 0 < seed < modulus or not 0 < multiplier < modulus:
        raise ValueError("seed and multiplier need to be between zero and modulus")
    if math.gcd(seed, modulus) != 1 or math.gcd(multiplier, modulus) != 1:
        raise ValueError("seed and multiplier need to be coprime with the modulus")


def multiplicative_numbers(seed, multiplier, modulus, count):
    """Generate count values using x[n] = multiplier * x[n-1] mod modulus."""
    _check_parameters(seed, multiplier, modulus)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count needs to be a positive integer")

    state = seed
    numbers = []
    for _ in range(count):
        state = multiplier * state % modulus
        numbers.append(state / modulus)

    return numbers


def cycle_period(seed, multiplier, modulus):
    """Follow the states until one repeats and return the cycle length."""
    _check_parameters(seed, multiplier, modulus)
    state = seed
    first_seen = {}
    step = 0

    while state not in first_seen:
        first_seen[state] = step
        state = multiplier * state % modulus
        step += 1

    return step - first_seen[state]


def _successive_pairs(numbers):
    numbers = list(numbers)
    if len(numbers) < 2:
        raise ValueError("at least two numbers are needed for successive pairs")
    if any(not math.isfinite(value) or not 0 < value < 1 for value in numbers):
        raise ValueError("numbers need to be finite and between zero and one")
    return numbers[:-1], numbers[1:]


def save_pair_plot(good_numbers, poor_numbers, filename=None):
    """Plot (u[n], u[n+1]) for the two parameter choices."""
    from matplotlib import pyplot as plt

    good_x, good_y = _successive_pairs(good_numbers)
    poor_x, poor_y = _successive_pairs(poor_numbers)
    if filename is None:
        filename = Path(__file__).with_name("successive_pairs.png")

    figure, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for axis, x_values, y_values, title in (
        (axes[0], good_x, good_y, "multiplier 3"),
        (axes[1], poor_x, poor_y, "multiplier 2"),
    ):
        axis.scatter(x_values, y_values, s=24, alpha=0.75)
        axis.set(
            title=title,
            xlabel="current value",
            ylabel="next value",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.2)

    figure.suptitle("Successive pairs from two small generators")
    figure.tight_layout()
    figure.savefig(filename, dpi=150)
    plt.close(figure)
    return Path(filename)


def run_example():
    seed = 1
    modulus = 31
    count = 60
    good_multiplier = 3
    poor_multiplier = 2

    good_numbers = multiplicative_numbers(
        seed, good_multiplier, modulus, count
    )
    poor_numbers = multiplicative_numbers(
        seed, poor_multiplier, modulus, count
    )

    print(f"Using seed {seed} and modulus {modulus}")
    print(
        f"Multiplier {good_multiplier}: period "
        f"{cycle_period(seed, good_multiplier, modulus)}"
    )
    print(
        f"Multiplier {poor_multiplier}: period "
        f"{cycle_period(seed, poor_multiplier, modulus)}"
    )
    print("First five values with multiplier 3:")
    print("  " + ", ".join(f"{value:.4f}" for value in good_numbers[:5]))

    plot_path = save_pair_plot(good_numbers, poor_numbers)
    print(f"Saved the successive-pairs plot as {plot_path.name}")
    print("This little generator is for learning, not for real random numbers.")


if __name__ == "__main__":
    run_example()
