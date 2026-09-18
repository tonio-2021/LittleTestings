# Checking JAX derivatives

I tried JAX's automatic differentiation on a two-input function:
`sin(x*y) + x² + y³/2`. It has a mix of terms but is still small enough to
check by changing one input at a time.

The script prints the gradient (the two slopes) and the Hessian trace (the
sum of the two diagonal second derivatives). It puts JAX's answers beside
finite-difference estimates from small nudges. They should be close, but the
finite-difference numbers depend a little on the chosen step size.

## What JAX is doing here

Automatic differentiation (often shortened to *autodiff* or *autograd*) is
different from guessing a slope by nudging an input. JAX follows the
operations in the function—multiply, `sin`, power, add—and applies the chain
rule to them. `jax.grad` works backward from the one-number output to get
both input slopes. `jax.hessian` differentiates again to get the second
derivatives. The trace I print is just the sum of the Hessian's diagonal.

That is useful because I can change the function without working out a new
derivative by hand or choosing a finite-difference step. The derivative is
still subject to floating-point arithmetic, but it does not have the same
step-size tradeoff as finite differences. This gets more helpful when a model
has lots of parameters. JAX can also compile the derivative calculation with
`jax.jit` so repeated calls reuse the compiled work.

## The little timing test

The script now times repeated calls to the two methods and prints which one
was quicker **on that run**. It warms JAX up first, so its one-time JIT
compilation is not included, then reports the median time per call from three
runs. The JAX function returns ordinary Python numbers, which makes the
timing wait for its work to finish.

This is only a two-input function on one machine. Python's finite differences
may well be faster here; they are a few arithmetic operations and an
approximation, while JAX has its own call and array overhead. So the timing
is a curiosity, not a claim that one method is always faster. Initial JAX
compilation would make a one-off call slower still.

I used the [JAX autodiff notes](https://docs.jax.dev/en/latest/jacobian-vector-products.html)
and [benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html) to
check the explanation and timing setup.

From the main folder:

```bash
python3 -m optimization.jax_derivatives.derivatives
python3 -m unittest optimization.jax_derivatives.test_derivatives
```
