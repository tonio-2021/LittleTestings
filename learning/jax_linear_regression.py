"""A first little learning loop with JAX: fit a line to toy data."""

import jax
import jax.numpy as jnp


def squared_error(params, x, y):
    predictions = params[0] * x + params[1]
    return jnp.mean((predictions - y) ** 2)


def check_data(x, y):
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)

    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("x and y need to be equally sized, one-dimensional arrays")
    if float(jnp.sum((x - jnp.mean(x)) ** 2)) == 0:
        raise ValueError("the x values cannot all be the same")

    return x, y


def fit_with_jax(x, y, step_size=0.1, steps=80):
    """Learn the slope and intercept with gradient descent."""
    x, y = check_data(x, y)
    if step_size <= 0 or steps < 0:
        raise ValueError("step_size must be positive and steps cannot be negative")

    params = jnp.array([0.0, 0.0])  # slope, then intercept
    losses = [float(squared_error(params, x, y))]
    gradient = jax.grad(squared_error)

    for _ in range(steps):
        # JAX gets the derivatives; I still choose how big each update is.
        params = params - step_size * gradient(params, x, y)
        losses.append(float(squared_error(params, x, y)))

    return params, losses


def direct_fit(x, y):
    """Fit the same line directly, just to check the learned answer."""
    x, y = check_data(x, y)
    x_offset = x - jnp.mean(x)
    y_offset = y - jnp.mean(y)
    slope = jnp.sum(x_offset * y_offset) / jnp.sum(x_offset**2)
    intercept = jnp.mean(y) - slope * jnp.mean(x)
    return jnp.array([slope, intercept])


def run_example():
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = jnp.array([-2.9, -1.1, 1.0, 3.1, 4.9])

    learned, losses = fit_with_jax(x, y)
    direct = direct_fit(x, y)

    print("Learning a line from five toy points")
    print(f"Starting squared error: {losses[0]:.4f}")
    print(f"After 10 updates:       {losses[10]:.4f}")
    print(f"After 80 updates:       {losses[-1]:.4f}\n")
    print(f"JAX loop:   slope={learned[0]:.3f}, intercept={learned[1]:.3f}")
    print(f"Direct fit: slope={direct[0]:.3f}, intercept={direct[1]:.3f}")


if __name__ == "__main__":
    run_example()
