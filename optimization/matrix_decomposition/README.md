# Splitting a matrix into two pieces

Any real square matrix can be split into a symmetric part and an
antisymmetric part:

```text
symmetric     = (A + A.T) / 2
antisymmetric = (A - A.T) / 2
```

Adding the parts gives the original matrix again. The script checks the two
transpose rules and also calculates `x.T @ K @ x` for the antisymmetric part
`K`. This value is zero apart from floating-point noise, because transposing
the scalar makes it equal to its own negative.

From the main folder:

```bash
python3 -m optimization.matrix_decomposition.decomposition
python3 -m unittest optimization.matrix_decomposition.test_decomposition
```

This is mostly a small sanity check around the formulas. The example uses a
three-by-three matrix so both parts are still easy to inspect by eye.
