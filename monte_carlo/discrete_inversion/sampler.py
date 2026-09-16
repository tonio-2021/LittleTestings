"""Pick an item from a small probability table."""

from bisect import bisect_right
import math
import random


class DiscreteSampler:
    def __init__(self, probabilities):
        probabilities = list(probabilities)
        if not probabilities or any(
            not math.isfinite(probability) or probability < 0
            for probability in probabilities
        ):
            raise ValueError("probabilities must be finite and nonnegative")
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("probabilities must add up to 1")

        self.cumulative = []
        total = 0.0
        for probability in probabilities:
            total += probability
            self.cumulative.append(total)
        self.cumulative[-1] = 1.0  # just in case the sum rounded a little

    def pick_linear(self, number):
        """Return the index whose interval contains a number in [0, 1)."""
        self._check_number(number)
        for index, upper_end in enumerate(self.cumulative):
            if number < upper_end:
                return index

    def pick_binary(self, number):
        """Find the same interval with binary search."""
        self._check_number(number)
        # At an exact cutoff I want the next interval, not the previous one.
        return bisect_right(self.cumulative, number)

    @staticmethod
    def _check_number(number):
        if not 0 <= number < 1:
            raise ValueError("number must be between 0 (included) and 1 (excluded)")


def run_example():
    choices = ["apple", "banana", "cherry", "date"]
    probabilities = [0.1, 0.2, 0.3, 0.4]
    sampler = DiscreteSampler(probabilities)
    rng = random.Random(42)
    numbers = [rng.random() for _ in range(10_000)]

    # Reusing these numbers makes it easier to check the two searches agree.
    linear_picks = [sampler.pick_linear(number) for number in numbers]
    binary_picks = [sampler.pick_binary(number) for number in numbers]

    print("Picking from a small probability table")
    print("Same answers from both searches?", linear_picks == binary_picks)
    print("choice    target    observed")
    for index, choice in enumerate(choices):
        observed = linear_picks.count(index) / len(linear_picks)
        print(f"{choice:7}   {probabilities[index]:.3f}     {observed:.3f}")


if __name__ == "__main__":
    run_example()
