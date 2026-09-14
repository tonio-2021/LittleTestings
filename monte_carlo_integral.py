"""Estimating an integral from random heights instead of random dots."""

import math
import random


def estimate_integral(function, left, right, samples, rng=None):
    """Return a Monte Carlo estimate and its estimated standard error."""
    if right <= left:
        raise ValueError("right has to be larger than left")
    if samples < 2:
        raise ValueError("at least two samples are needed for the standard error")

    if rng is None:
        rng = random.Random()

    width = right - left
    heights = [function(rng.uniform(left, right)) for _ in range(samples)]
    average = sum(heights) / samples

    # The spread of the sampled heights gives a rough idea of the error.
    variance = sum((height - average) ** 2 for height in heights) / (samples - 1)
    estimate = width * average
    standard_error = width * math.sqrt(variance / samples)
    return estimate, standard_error


def midpoint_integral(function, left, right, steps):
    """A simple non-random comparison using the middle of each interval."""
    if right <= left:
        raise ValueError("right has to be larger than left")
    if steps < 1:
        raise ValueError("steps has to be positive")

    width = (right - left) / steps
    return width * sum(
        function(left + (index + 0.5) * width) for index in range(steps)
    )


def run_example():
    function = lambda x: 1 / x
    exact = math.log(3)
    rng = random.Random(42)

    print("Estimating the area under 1/x from 1 to 3")
    print(f"math.log(3): {exact:.6f}\n")
    print("samples    estimate    rough SE    absolute error")

    for samples in (100, 1_000, 10_000):
        estimate, standard_error = estimate_integral(function, 1, 3, samples, rng)
        print(
            f"{samples:>7,}    {estimate:.6f}    {standard_error:.6f}"
            f"    {abs(estimate - exact):.6f}"
        )

    midpoint = midpoint_integral(function, 1, 3, 1_000)
    print(f"\nMidpoint rule (1,000 steps): {midpoint:.6f}")
    print("One random run can wobble, even when the sample size gets bigger.")


if __name__ == "__main__":
    run_example()
