"""Compare a few greedy rules for choosing non-overlapping intervals."""

import math
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Interval:
    name: str
    start: float
    end: float


def _check_intervals(intervals):
    intervals = list(intervals)
    names = set()

    for interval in intervals:
        if not isinstance(interval, Interval):
            raise ValueError("intervals need to be Interval objects")
        if not interval.name or interval.name in names:
            raise ValueError("interval names need to be unique and non-empty")
        if not math.isfinite(interval.start) or not math.isfinite(interval.end):
            raise ValueError("interval endpoints need to be finite")
        if interval.start >= interval.end:
            raise ValueError("an interval needs to end after it starts")
        names.add(interval.name)

    return intervals


def _compatible(intervals):
    ordered = sorted(intervals, key=lambda interval: (interval.start, interval.end))
    return all(
        current.end <= following.start
        for current, following in zip(ordered, ordered[1:])
    )


def are_compatible(intervals):
    """Check that no pair of intervals overlaps."""
    return _compatible(_check_intervals(intervals))


def greedy_schedule(intervals, rule="earliest_finish"):
    """Choose intervals using one of three simple ordering rules."""
    intervals = _check_intervals(intervals)
    keys = {
        "earliest_finish": lambda interval: (interval.end, interval.start),
        "earliest_start": lambda interval: (interval.start, interval.end),
        "shortest_duration": lambda interval: (
            interval.end - interval.start,
            interval.end,
        ),
    }
    if rule not in keys:
        raise ValueError("unknown scheduling rule")

    selected = []
    for interval in sorted(intervals, key=keys[rule]):
        if _compatible([*selected, interval]):
            selected.append(interval)

    return sorted(selected, key=lambda interval: interval.start)


def maximum_schedule(intervals):
    """Find an optimum by checking subsets, only for small comparisons."""
    intervals = _check_intervals(intervals)

    for size in range(len(intervals), -1, -1):
        for group in combinations(intervals, size):
            if _compatible(group):
                return list(group)

    return []


def _names(intervals):
    return [interval.name for interval in intervals]


def run_example():
    earliest_start_trap = [
        Interval("long", 0, 10),
        Interval("a", 1, 2),
        Interval("b", 2, 3),
        Interval("c", 3, 4),
        Interval("d", 4, 5),
    ]
    shortest_trap = [
        Interval("left", 0, 3),
        Interval("short middle", 2, 4),
        Interval("right", 3, 6),
    ]

    print("Example where picking the earliest start gets stuck:")
    for rule in ("earliest_finish", "earliest_start"):
        chosen = greedy_schedule(earliest_start_trap, rule)
        print(f"  {rule:18} -> {_names(chosen)}")

    print("\nExample where picking the shortest interval gets stuck:")
    for rule in ("earliest_finish", "shortest_duration"):
        chosen = greedy_schedule(shortest_trap, rule)
        print(f"  {rule:18} -> {_names(chosen)}")

    optimum = maximum_schedule(shortest_trap)
    print(f"\nBrute-force best for the second example: {_names(optimum)}")


if __name__ == "__main__":
    run_example()
