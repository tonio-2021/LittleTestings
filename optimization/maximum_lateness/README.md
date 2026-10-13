# Scheduling by deadline

This is a small single-machine scheduling example. Each job has a processing
time and a deadline. Finishing after the deadline gives positive lateness,
while finishing early gives negative lateness.

The greedy rule is to process the jobs in increasing deadline order. The
script returns the order, completion times, individual lateness, and the
largest lateness. I also check all possible orders for the tiny example so
there is something to compare the greedy answer with.

For the four toy jobs, the original order has maximum lateness `3`. Sorting
by deadline lowers it to `1`, which is also the brute-force optimum.

From the main folder:

```bash
python3 -m optimization.maximum_lateness.scheduling
python3 -m unittest optimization.maximum_lateness.test_scheduling
```

The exhaustive check grows very quickly, so the code limits it to nine jobs.
