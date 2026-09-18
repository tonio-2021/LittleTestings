# Continuous inversion sampling

This continues the probability-table experiment, but now the results are
continuous numbers. I start with a uniform random number and put it through
an inverse CDF. The two examples are an exponential distribution and a power
distribution with `F(x) = x³` on `[0, 1]`.

The same `sample_inverse` function works for either formula. The script prints
sample quartiles next to the quartiles from the inverse formulas. They are
close for 10,000 draws with seed 42, but a different run will move them a bit.

From the main folder:

```bash
python3 -m monte_carlo.continuous_inversion.sampler
python3 -m unittest monte_carlo.continuous_inversion.test_sampler
```
