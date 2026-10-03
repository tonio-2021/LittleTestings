# A chain-rule check with JAX

This is a small check for a function written as `G(x) = F(Mx)`. I used

```text
F(y) = sum(sin(y) + y² / 2)
```

so its gradient is still easy to write down. Applying the chain rule gives
`M.T @ grad F(Mx)`. The script compares this calculation with the gradient
JAX gets by differentiating the whole composed function.

I tried both a square matrix and a rectangular one. The rectangular case was
useful for seeing why the transpose is needed: `Mx` can live in a different
dimension, but the final gradient has to match the original `x`.

For the fixed examples, the differences between the two gradients were
`1.53e-07` for the square matrix and `7.94e-08` for the rectangular one.
That is small enough for the floating-point calculation here.

From the main folder:

```bash
python3 -m optimization.change_of_variables.gradients
python3 -m unittest optimization.change_of_variables.test_gradients
```
