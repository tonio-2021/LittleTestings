# Picking from a probability table

This is a small discrete inversion sampler. I give four choices different
chances, add those chances up, and use a random number between 0 and 1 to pick
the interval it lands in.

There are two ways to look up the interval here: walk through the cumulative
probabilities, or use binary search. I feed both versions the same random
numbers so it is easy to see whether they disagree.

From the repository's main folder:

```bash
python3 -m monte_carlo.discrete_inversion.sampler
python3 -m unittest monte_carlo.discrete_inversion.test_sampler
```

With 10,000 draws, the observed shares should be near the probabilities in
the table, though they will not match exactly.
