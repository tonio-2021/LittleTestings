"""Check the chain rule after a linear change of variables."""

import numpy as np
import jax
import jax.numpy as jnp


def base_value(values):
    """A small scalar function F(y) used in the comparison."""
    values = jnp.asarray(values)
    return jnp.sum(jnp.sin(values) + 0.5 * values**2)


def transformed_value(point, matrix):
    """Evaluate G(x) = F(Mx)."""
    return base_value(jnp.asarray(matrix) @ jnp.asarray(point))


def _check_inputs(point, matrix):
    try:
        point = np.asarray(point, dtype=float)
        matrix = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("point and matrix need to contain numbers") from error

    if point.ndim != 1 or point.size == 0:
        raise ValueError("point needs to be a non-empty vector")
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("matrix needs to be two-dimensional and non-empty")
    if matrix.shape[1] != point.size:
        raise ValueError("matrix columns need to match the point size")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(matrix)):
        raise ValueError("point and matrix need to be finite")

    return point, matrix


def chain_rule_gradient(point, matrix):
    """Calculate M.T @ grad F(Mx) by hand."""
    point, matrix = _check_inputs(point, matrix)
    inside = matrix @ point
    base_gradient = np.cos(inside) + inside

    # The transpose brings the slopes back to the coordinates of x.
    return matrix.T @ base_gradient


def automatic_gradient(point, matrix):
    """Let JAX differentiate the composed function directly."""
    point, matrix = _check_inputs(point, matrix)
    gradient = jax.grad(transformed_value, argnums=0)(
        jnp.asarray(point), jnp.asarray(matrix)
    )
    return np.asarray(gradient, dtype=float)


def comparison_error(point, matrix):
    by_hand = chain_rule_gradient(point, matrix)
    automatic = automatic_gradient(point, matrix)
    return float(np.linalg.norm(by_hand - automatic))


def run_example():
    examples = {
        "square 2 by 2": (
            np.array([0.4, -0.7]),
            np.array([[1.0, 0.5], [-0.25, 2.0]]),
        ),
        "rectangular 3 by 2": (
            np.array([0.4, -0.7]),
            np.array([[1.0, 0.0], [0.5, -1.0], [-0.2, 0.8]]),
        ),
    }

    for name, (point, matrix) in examples.items():
        by_hand = chain_rule_gradient(point, matrix)
        automatic = automatic_gradient(point, matrix)
        error = np.linalg.norm(by_hand - automatic)

        print(name)
        print(f"  chain rule: {np.round(by_hand, 6)}")
        print(f"  JAX:        {np.round(automatic, 6)}")
        print(f"  difference: {error:.2e}")


if __name__ == "__main__":
    run_example()
