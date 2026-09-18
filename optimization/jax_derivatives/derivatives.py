"""Compare JAX derivatives with small changes to the inputs."""

import math
from statistics import median
from time import perf_counter

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


@jax.jit
def _compiled_derivatives(point):
    gradient = jax.grad(toy_function)(point)
    hessian = jax.hessian(toy_function)(point)
    return gradient, jnp.trace(hessian)


def jax_derivatives(point):
    point = jnp.asarray(_check_point(point), dtype=float)
    gradient, trace = _compiled_derivatives(point)
    return tuple(float(value) for value in gradient), float(trace)


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


def benchmark_derivatives(point=(0.7, -0.4), repeats=200):
    """Median seconds per call after JAX has compiled the small function."""
    point = _check_point(point)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    # The first JAX call includes compilation, so I leave it out of this timing.
    jax_derivatives(point)
    finite_difference_derivatives(point)
    methods = {
        "JAX": jax_derivatives,
        "finite differences": finite_difference_derivatives,
    }
    measurements = {name: [] for name in methods}

    for _ in range(3):
        for name, method in methods.items():
            start = perf_counter()
            for _ in range(repeats):
                method(point)
            measurements[name].append((perf_counter() - start) / repeats)

    return {name: median(values) for name, values in measurements.items()}


def run_example():
    point = (0.7, -0.4)
    jax_gradient, jax_trace = jax_derivatives(point)
    rough_gradient, rough_trace = finite_difference_derivatives(point)

    print(f"Checking derivatives at {point}")
    print(f"JAX gradient:        ({jax_gradient[0]:.5f}, {jax_gradient[1]:.5f})")
    print(f"Finite difference:   ({rough_gradient[0]:.5f}, {rough_gradient[1]:.5f})")
    print(f"JAX Hessian trace:   {jax_trace:.5f}")
    print(f"Finite difference:   {rough_trace:.5f}")

    timings = benchmark_derivatives(point)
    print("\nRepeated calls after JAX warm-up (median of 3 runs):")
    for name, seconds in timings.items():
        print(f"  {name:18} {seconds * 1_000_000:.1f} microseconds per call")
    faster = min(timings, key=timings.get)
    print(f"Faster for this tiny example: {faster}")


if __name__ == "__main__":
    run_example()
