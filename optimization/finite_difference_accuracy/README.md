# How small should a finite-difference step be?

This checks two simple ways of estimating the derivative of `sin(x)` at
`x = 1`. The forward difference uses the current point and one point just to
its right. The centered difference looks the same distance in both directions.

For an ordinary-sized step, the centered version is much more accurate here.
Making the step smaller helps both methods at first. It does not help forever,
because eventually the two function values are so close that floating-point
rounding matters more than the formula's approximation error.

From the main folder:

```bash
python3 -m optimization.finite_difference_accuracy.differences
python3 -m unittest optimization.finite_difference_accuracy.test_differences
```

The script prints the errors for steps from `1e-1` to `1e-16` and saves a
log-log plot. The best step is not the smallest one, which was the main thing
I wanted to see in this example.
