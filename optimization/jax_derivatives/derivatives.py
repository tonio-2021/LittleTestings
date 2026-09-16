"""Compare JAX derivatives with small changes to the inputs."""

import math

import jax
import jax.numpy as jnp


def toy_function(point):
    x, y = point
    return jnp.sin(x * y) + x**2 + 0.5 * y**3


def _check_point(point):
    point = tuple(point)
    if len(point) != 2 or any(not math.isfinite(value) for value in point):
        raise ValueError("point needs two finite numbers")
    return point


def jax_derivatives(point):
    point = jnp.asarray(_check_point(point), dtype=float)
    gradient = jax.grad(toy_function)(point)
    hessian = jax.hessian(toy_function)(point)
    return tuple(float(value) for value in gradient), float(jnp.trace(hessian))


def finite_difference_derivatives(point, step=0.001):
    point = _check_point(point)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")

    def plain_value(values):
        x, y = values
        return math.sin(x * y) + x**2 + 0.5 * y**3

    middle = plain_value(point)
    gradient = []
    trace = 0.0

    for index in range(2):
        left = list(point)
        right = list(point)
        left[index] -= step
        right[index] += step
        left_value = plain_value(left)
        right_value = plain_value(right)

        # First differences estimate the slope; second differences add up to the trace.
        gradient.append((right_value - left_value) / (2 * step))
        trace += (right_value - 2 * middle + left_value) / step**2

    return tuple(gradient), trace


def run_example():
    point = (0.7, -0.4)
    jax_gradient, jax_trace = jax_derivatives(point)
    rough_gradient, rough_trace = finite_difference_derivatives(point)

    print(f"Checking derivatives at {point}")
    print(f"JAX gradient:        ({jax_gradient[0]:.5f}, {jax_gradient[1]:.5f})")
    print(f"Finite difference:   ({rough_gradient[0]:.5f}, {rough_gradient[1]:.5f})")
    print(f"JAX Hessian trace:   {jax_trace:.5f}")
    print(f"Finite difference:   {rough_trace:.5f}")


if __name__ == "__main__":
    run_example()
