# Checking JAX derivatives

I tried JAX's automatic differentiation on a two-input function:
`sin(x*y) + x² + y³/2`. It has a mix of terms but is still small enough to
check by changing one input at a time.

The script prints the gradient (the two slopes) and the Hessian trace (the
sum of the two diagonal second derivatives). It puts JAX's answers beside
finite-difference estimates from small nudges. They should be close, but the
finite-difference numbers depend a little on the chosen step size.

From the main folder:

```bash
python3 -m optimization.jax_derivatives.derivatives
python3 -m unittest optimization.jax_derivatives.test_derivatives
```
