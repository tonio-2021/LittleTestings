# Picking non-overlapping intervals

This is a small greedy scheduling example. The goal is to keep as many
non-overlapping activities as possible. The useful rule is to repeatedly take
the available interval that finishes first, because it leaves the most room
for whatever comes next.

I also tried sorting by earliest start and by shortest duration. Both sound
fairly sensible, but the script includes a tiny case where each of them gets
stuck with fewer activities. A brute-force search is included just to check
the greedy result on small inputs.

From the main folder:

```bash
python3 -m optimization.interval_scheduling.scheduling
python3 -m unittest optimization.interval_scheduling.test_scheduling
```

The exhaustive check is not part of the fast algorithm. It is only practical
for these little examples because it tries subsets of the intervals.
