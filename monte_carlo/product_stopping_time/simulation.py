"""Multiply uniforms until the running product crosses a threshold."""

import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pvariance


def _check_threshold(threshold):
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold needs to be between 0 and 1")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold needs to be between 0 and 1")


def stopping_count(threshold, rng=None):
    """Return how many uniform draws are needed to cross the threshold."""
    _check_threshold(threshold)
    if rng is None:
        rng = random.Random()

    product = 1.0
    count = 0

    while product > threshold:
        product *= rng.random()
        count += 1

    return count


def simulate_counts(trials, threshold, rng=None):
    """Repeat the stopping experiment a number of times."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials needs to be a positive integer")
    _check_threshold(threshold)
    if rng is None:
        rng = random.Random()

    return [stopping_count(threshold, rng) for _ in range(trials)]


def summarize_counts(counts, largest_count=6):
    """Calculate a mean, variance, and the first few observed probabilities."""
    counts = list(counts)
    invalid_count = any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in counts
    )
    if not counts or invalid_count:
        raise ValueError("counts needs to contain positive integers")
    if (
        isinstance(largest_count, bool)
        or not isinstance(largest_count, int)
        or largest_count < 1
    ):
        raise ValueError("largest_count needs to be a positive integer")

    frequencies = Counter(counts)
    probabilities = {
        count: frequencies[count] / len(counts)
        for count in range(1, largest_count + 1)
    }
    return mean(counts), pvariance(counts), probabilities


def theoretical_summary(threshold, largest_count=6):
    """Return the matching mean, variance, and probabilities."""
    _check_threshold(threshold)
    if (
        isinstance(largest_count, bool)
        or not isinstance(largest_count, int)
        or largest_count < 1
    ):
        raise ValueError("largest_count needs to be a positive integer")

    rate = -math.log(threshold)
    probabilities = {
        count: threshold * rate ** (count - 1) / math.factorial(count - 1)
        for count in range(1, largest_count + 1)
    }
    return rate + 1, rate, probabilities


def save_probability_plot(observed, expected, filename=None):
    """Put the simulated and calculated probabilities beside each other."""
    counts = sorted(observed)
    if not counts or set(observed) != set(expected):
        raise ValueError("observed and expected need the same counts")

    all_values = [*observed.values(), *expected.values()]
    if any(not math.isfinite(value) or value < 0 for value in all_values):
        raise ValueError("probabilities need to be finite and nonnegative")

    from matplotlib import pyplot as plt

    if filename is None:
        filename = Path(__file__).with_name("stopping_count_probabilities.png")

    positions = list(range(len(counts)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        [position - width / 2 for position in positions],
        [observed[count] for count in counts],
        width,
        label="simulated",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [expected[count] for count in counts],
        width,
        label="calculated",
    )
    axis.set_xticks(positions, counts)
    axis.set(
        xlabel="stopping count",
        ylabel="probability",
        title="How many uniforms are multiplied before stopping?",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return Path(filename)


def run_example():
    threshold = math.exp(-2)
    counts = simulate_counts(20_000, threshold)
    sample_mean, sample_variance, observed = summarize_counts(
        counts,
        largest_count=8,
    )
    exact_mean, exact_variance, expected = theoretical_summary(
        threshold,
        largest_count=8,
    )

    print(f"Stopping when the product falls below exp(-2) = {threshold:.4f}")
    print(f"Simulated mean:     {sample_mean:.4f}")
    print(f"Calculated mean:    {exact_mean:.4f}")
    print(f"Simulated variance: {sample_variance:.4f}")
    print(f"Calculated variance: {exact_variance:.4f}\n")
    print(" count   observed   calculated")
    for count in observed:
        print(f"{count:6d}   {observed[count]:8.4f}   {expected[count]:10.4f}")

    plot_path = save_probability_plot(observed, expected)
    print(
        "\nThe numbers move a little each run because the example does not fix a seed."
    )
    print(f"Saved the probability comparison as {plot_path.name}")


if __name__ == "__main__":
    run_example()
