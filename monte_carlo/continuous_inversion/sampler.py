"""Turn uniform random numbers into samples from other distributions."""

import math
import random
from statistics import quantiles


def _check_uniform(number):
    if not math.isfinite(number) or not 0 <= number < 1:
        raise ValueError("number must be in [0, 1)")


def exponential_inverse(number, rate=1.0):
    """Inverse CDF for an exponential distribution with the given rate."""
    _check_uniform(number)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")
    return -math.log1p(-number) / rate


def power_inverse(number, shape=2.0):
    """Inverse CDF for F(x) = x**shape on [0, 1]."""
    _check_uniform(number)
    if not math.isfinite(shape) or shape <= 0:
        raise ValueError("shape must be finite and positive")
    return number ** (1 / shape)


def sample_inverse(inverse_cdf, count, rng=None):
    """Use one inverse CDF to transform a batch of uniform draws."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if rng is None:
        rng = random.Random()
    # The drawing loop stays the same when I swap the inverse formula.
    return [inverse_cdf(rng.random()) for _ in range(count)]


def run_example():
    examples = [
        ("exponential (rate 2)", lambda number: exponential_inverse(number, rate=2)),
        ("power (shape 3)", lambda number: power_inverse(number, shape=3)),
    ]

    print("Comparing quartiles from 10,000 transformed uniform draws")
    for name, inverse_cdf in examples:
        draws = sample_inverse(inverse_cdf, 10_000, random.Random(42))
        observed = quantiles(draws, n=4, method="inclusive")
        print(name)
        for label, probability, sample_quartile in zip(
            ("Q1", "median", "Q3"), (0.25, 0.5, 0.75), observed
        ):
            expected = inverse_cdf(probability)
            print(f"  {label:6} sample={sample_quartile:.3f}  target={expected:.3f}")


if __name__ == "__main__":
    run_example()
