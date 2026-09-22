"""Split a square matrix into symmetric and antisymmetric pieces."""

import numpy as np


def _square_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix needs to be square")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix entries need to be finite")
    return matrix


def decompose(matrix):
    """Return the symmetric and antisymmetric parts of a square matrix."""
    matrix = _square_matrix(matrix)
    symmetric = (matrix + matrix.T) / 2
    antisymmetric = (matrix - matrix.T) / 2
    return symmetric, antisymmetric


def quadratic_value(matrix, vector):
    """Calculate x-transpose matrix x for a matching vector."""
    matrix = _square_matrix(matrix)
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or len(vector) != len(matrix):
        raise ValueError("vector size needs to match the matrix")
    if not np.isfinite(vector).all():
        raise ValueError("vector entries need to be finite")
    return float(vector @ matrix @ vector)


def run_example():
    matrix = np.array(
        [
            [2, -1, 4],
            [3, 0, 5],
            [-2, 1, 7],
        ],
        dtype=float,
    )
    vector = np.array([1, -2, 0.5])
    symmetric, antisymmetric = decompose(matrix)

    print("Original matrix:\n", matrix)
    print("\nSymmetric part:\n", symmetric)
    print("\nAntisymmetric part:\n", antisymmetric)

    # Adding the two pieces should give back exactly what I started with.
    reconstruction_error = np.max(np.abs(matrix - symmetric - antisymmetric))
    print(f"\nLargest reconstruction error: {reconstruction_error:.2e}")
    print(f"x.T @ antisymmetric @ x: {quadratic_value(antisymmetric, vector):.2e}")


if __name__ == "__main__":
    run_example()
